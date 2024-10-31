from dictionary_learning.dictionary import CrossCoder
from collections import defaultdict
import torch as th
from tools.split_gemma import split_gemma
from argparse import ArgumentParser
from tqdm.auto import tqdm
import pandas as pd
import json
import wandb
from dlabutils import model_path
from pathlib import Path
from torch.nn.functional import kl_div
from torchmetrics.aggregation import MeanMetric
from setup_to_eval import *

CHAT_TEMPLATE = open("templates/gemma_chat_template.jinja").read()


@th.inference_mode()
def evaluate_interventions(
    base_model,
    instruct_model,
    dataset,
    preprocess_before_last_half_fns: dict[str, HalfStepPreprocessFn],
    layer_to_stop=13,
    batch_size=8,
    device="cuda",
    max_seq_len=1024,
    log_every=100,
    checkpoint_every=2000,
    save_path: Path | None = None,
):
    MeanMetricToDevice = lambda: MeanMetric().to(device)
    nlls = defaultdict(MeanMetricToDevice)
    perplexity = defaultdict(MeanMetricToDevice)
    instruct_kl = defaultdict(MeanMetricToDevice)
    base_kl = defaultdict(MeanMetricToDevice)
    nlls_wrt_instruct_pred = defaultdict(MeanMetricToDevice)
    perplexity_wrt_instruct_pred = defaultdict(MeanMetricToDevice)
    num_samples = defaultdict(int)
    base_model = split_gemma(base_model)
    instruct_model = split_gemma(instruct_model)

    def compute_result():
        ppx = {
            fn_name: perplexity[fn_name].compute().item()
            for fn_name in preprocess_before_last_half_fns
        }
        nll = {
            fn_name: nlls[fn_name].compute().item()
            for fn_name in preprocess_before_last_half_fns
        }
        it_kl = {
            fn_name: instruct_kl[fn_name].compute().item()
            for fn_name in preprocess_before_last_half_fns
        }
        b_kl = {
            fn_name: base_kl[fn_name].compute().item()
            for fn_name in preprocess_before_last_half_fns
        }
        nll_wrt_it = {
            fn_name: nlls_wrt_instruct_pred[fn_name].compute().item()
            for fn_name in preprocess_before_last_half_fns
        }
        ppx_wrt_it = {
            fn_name: perplexity_wrt_instruct_pred[fn_name].compute().item()
            for fn_name in preprocess_before_last_half_fns
        }
        return {
            "loss": nll,
            "perplexity": ppx,
            "kl-instruct": it_kl,
            "kl-base": b_kl,
            "loss_wrt_instruct_pred": nll_wrt_it,
            "perplexity_wrt_instruct_pred": ppx_wrt_it,
        }

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        batch_tokens = instruct_model.tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            return_assistant_tokens_mask=True,
            chat_template=CHAT_TEMPLATE,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        batch_tokens["input_ids"] = batch_tokens["input_ids"][:, -max_seq_len:].to(
            device
        )
        batch_tokens["attention_mask"] = batch_tokens["attention_mask"][
            :, -max_seq_len:
        ].to(device)
        batch_tokens["assistant_masks"] = (
            th.tensor(batch_tokens["assistant_masks"])[:, -max_seq_len:]
            .bool()
            .to(device)
        )
        # manual truncation because return_assistant_tokens_mask + truncate = 🤮
        base_activations, *base_other_outputs = base_model.first_half_forward(
            input_ids=batch_tokens["input_ids"],
            attention_mask=batch_tokens["attention_mask"],
            layer_idx=layer_to_stop,
        )
        instruct_activations, *instruct_other_outputs = (
            instruct_model.first_half_forward(
                input_ids=batch_tokens["input_ids"],
                attention_mask=batch_tokens["attention_mask"],
                layer_idx=layer_to_stop,
            )
        )
        base_logits = (
            base_model.second_half_forward(
                base_activations,
                *base_other_outputs,
                layer_idx=layer_to_stop,
                return_dict=True,
            )
            .logits[batch_tokens["assistant_masks"]]
            .float()
        )
        base_log_probs = th.log_softmax(base_logits, dim=-1)
        instruct_logits = instruct_model.second_half_forward(
            instruct_activations,
            *instruct_other_outputs,
            layer_idx=layer_to_stop,
            return_dict=True,
        ).logits
        instruct_preds = th.argmax(instruct_logits, dim=-1)[
            batch_tokens["assistant_masks"]
        ]
        instruct_logits = instruct_logits[
            batch_tokens["assistant_masks"]
        ].float()  # num_mask=1, d_vocab
        instruct_log_probs = th.log_softmax(instruct_logits, dim=-1)
        for (
            fn_name,
            preprocess_before_last_half_fn,
        ) in preprocess_before_last_half_fns.items():
            base_activations_edited, instruct_activations_edited, mask = (
                preprocess_before_last_half_fn(base_activations, instruct_activations)
            )
            if mask is not None:
                
            final_out = None
            if base_activations_edited is not None:
                final_out = base_model.second_half_forward(
                    base_activations_edited,
                    *base_other_outputs,
                    layer_idx=layer_to_stop,
                    return_dict=True,
                )
            elif instruct_activations_edited is not None:
                final_out = instruct_model.second_half_forward(
                    instruct_activations_edited,
                    *instruct_other_outputs,
                    layer_idx=layer_to_stop,
                    return_dict=True,
                )
            if final_out is not None:
                logits = final_out.logits[batch_tokens["assistant_masks"]].float()
                log_probs = th.log_softmax(logits, dim=-1)
                it_kl = kl_div(
                    log_probs,
                    instruct_log_probs,
                    log_target=True,
                    reduction="none",
                ).sum(dim=-1)
                b_kl = kl_div(
                    log_probs,
                    base_log_probs,
                    log_target=True,
                    reduction="none",
                ).sum(dim=-1)
                loss = instruct_model.compute_loss(
                    logits, batch_tokens["input_ids"][batch_tokens["assistant_masks"]]
                )
                loss_wrt_instruct_pred = instruct_model.compute_loss(
                    logits, instruct_preds, already_shifted=True
                )
                wandb.log(
                    {
                        f"loss_wrt_instruct_pred/{fn_name}": loss_wrt_instruct_pred.item()
                    },
                    step=i,
                )
                wandb.log(
                    {
                        f"perplexity_wrt_instruct_pred/{fn_name}": th.exp(
                            loss_wrt_instruct_pred
                        ).item()
                    },
                    step=i,
                )
                wandb.incr

                wandb.log({f"loss/{fn_name}": loss.item()}, step=i)
                wandb.log({f"perplexity/{fn_name}": th.exp(loss).item()}, step=i)
                n_pred = log_probs.shape[0]
                assert log_probs.dim() == 2
                wandb.log(
                    {f"kl-instruct/{fn_name}": (it_kl.sum() / n_pred).item()}, step=i
                )
                wandb.log({f"kl-base/{fn_name}": (b_kl.sum() / n_pred).item()}, step=i)
                nlls[fn_name].update(loss)
                perplexity[fn_name].update(th.exp(loss))
                instruct_kl[fn_name].update(it_kl)
                base_kl[fn_name].update(b_kl)
                nlls_wrt_instruct_pred[fn_name].update(loss_wrt_instruct_pred)
                perplexity_wrt_instruct_pred[fn_name].update(
                    th.exp(loss_wrt_instruct_pred)
                )
                num_samples[fn_name] += n_pred
                wandb.log({f"num_samples/{fn_name}": num_samples[fn_name]}, step=i)
            if i % log_every == 0:
                wandb.log(
                    {
                        f"perplexity_running/{fn_name}": perplexity[fn_name]
                        .compute()
                        .item()
                    },
                    step=i,
                )
                wandb.log(
                    {f"loss_running/{fn_name}": nlls[fn_name].compute().item()},
                    step=i,
                )
                wandb.log(
                    {
                        f"kl-instruct_running/{fn_name}": instruct_kl[fn_name]
                        .compute()
                        .item()
                    },
                    step=i,
                )
                wandb.log(
                    {f"kl-base_running/{fn_name}": base_kl[fn_name].compute().item()},
                    step=i,
                )
                wandb.log(
                    {
                        f"loss_wrt_instruct_pred_running/{fn_name}": nlls_wrt_instruct_pred[
                            fn_name
                        ]
                        .compute()
                        .item()
                    },
                    step=i,
                )
                wandb.log(
                    {
                        f"perplexity_wrt_instruct_pred_running/{fn_name}": perplexity_wrt_instruct_pred[
                            fn_name
                        ]
                        .compute()
                        .item()
                    },
                    step=i,
                )
                if save_path is not None:
                    with open(
                        save_path / f"{wandb.run.name}_latest_result.json", "w"
                    ) as f:
                        json.dump(compute_result(), f)
            if i % checkpoint_every == 0 and save_path is not None and i != 0:
                with open(save_path / f"{wandb.run.name}_{i}_result.json", "w") as f:
                    json.dump(compute_result(), f)
    return compute_result()


from datasets import load_from_disk

# from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer-to-stop", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/dlabscratch1/jminder/repositories/representation-structure-comparison/datasets/test/lmsys_chat",
    )
    parser.add_argument(
        "--crosscoder-path",
        type=str,
        default="/dlabscratch1/jminder/repositories/representation-structure-comparison/checkpoints/l13-mu4.1e-02-lr1e-04/ae_final.pt",
    )
    parser.add_argument(
        "--feature-df-path",
        type=Path,
        default=Path(
            "/dlabscratch1/cdumas/representation-structure-comparison/notebooks/results/eval_crosscoder/l13-mu4.1e-02-lr1e-04_ae_final/data/feature_df.csv"
        ),
    )
    parser.add_argument("--it-only-feature-list-path", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-path", type=Path, default=Path("results-runai"))
    args = parser.parse_args()
    instruct_model = AutoModelForCausalLM.from_pretrained(
        model_path("google/gemma-2-2b-it"),
        torch_dtype=th.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path("google/gemma-2-2b-it"))
    instruct_model.tokenizer = tokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path("google/gemma-2-2b"),
        device_map="cuda",
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
    )
    dataset = load_from_disk(args.dataset_path)
    if args.test:
        dataset = dataset.select(range(300))
    device = (
        args.device
        if args.device != "auto"
        else "cuda" if th.cuda.is_available() else "cpu"
    )
    crosscoder = CrossCoder.from_pretrained(args.crosscoder_path, device=device)
    df = pd.read_csv(args.feature_df_path)
    it_only_features = df[df["tag"] == "IT only"].index.tolist()
    base_only_features = df[df["tag"] == "Base only"].index.tolist()
    if args.it_only_feature_list_path is not None:
        it_only_features = pd.read_json(args.it_only_feature_list_path).index.tolist()
        print(f"Using IT only features: {it_only_features}")
    if args.name is None and args.test:
        args.name = "test"
    wandb.init(project="perplexity-comparison", name=args.name)
    seeds = list(range(10))
    fn_dict = {}
    # fn_dict = create_half_fn_dict_main(crosscoder, it_only_features, base_only_features)
    # fn_dict.update(create_half_fn_dict_no_cross())
    # fn_dict.update(create_half_fn_dict_seeds(crosscoder, seeds, len(it_only_features)))
    # fn_dict.update(
    #     create_half_fn_dict_secondary(crosscoder, it_only_features, base_only_features)
    # )
    # fn_dict.update(
    #     create_it_only_ft_fn_dict(crosscoder, it_only_features, base_only_features)
    # )
    fn_dict.update(create_half_fn_thresholded_features(crosscoder))
    fn_dict.update(
        create_half_fn_dict_steer_seeds(
            crosscoder, seeds, len(INTERESTING_FEATURES), threshold=10
        )
    )
    fn_dict.update(
        create_half_fn_dict_remove_seeds(
            crosscoder, seeds, len(INTERESTING_FEATURES), threshold=10
        )
    )
    fn_dict.update(create_half_fn_dict_steer_seeds(crosscoder, seeds, 1, threshold=10))
    fn_dict.update(create_half_fn_dict_remove_seeds(crosscoder, seeds, 1, threshold=10))
    result = evaluate_interventions(
        base_model,
        instruct_model,
        dataset["conversation"],
        # create_half_fn_dict_main(crosscoder, it_only_features, base_only_features),
        fn_dict,
        layer_to_stop=args.layer_to_stop,
        batch_size=args.batch_size,
        device=device,
        max_seq_len=args.max_seq_len,
        log_every=args.log_every,
        save_path=args.save_path,
    )
    args.save_path.mkdir(parents=True, exist_ok=True)
    wdb_name = wandb.run.name
    with open(args.save_path / f"{wdb_name}_result.json", "w") as f:
        json.dump(result, f)
