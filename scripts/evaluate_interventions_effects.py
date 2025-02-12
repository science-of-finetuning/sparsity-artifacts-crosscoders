import sys
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from datasets import load_dataset
from torch.nn.functional import kl_div
import torch as th

sys.path.append(str(Path(__file__).parent.parent))
from tools.compute_utils import RunningMeanStd
from tools.setup_to_eval import HalfStepPreprocessFn
from tools.split_gemma import split_gemma
from tools.tokenization_utils import tokenize_with_ctrl_mask, gemma_chat_template
from tools.utils import mask_k_first_ones_vec
from tools.setup_to_eval import create_acl_half_fns
from tools.cc_utils import load_crosscoder


def compute_metrics_for_subset(
    logits, labels, base_logits, instruct_logits, subset_mask, chat_model
):
    """Compute all metrics for a subset of tokens."""
    base_log_probs = th.log_softmax(base_logits[subset_mask].float(), dim=-1)
    instruct_preds = th.argmax(instruct_logits, dim=-1)[subset_mask]
    instruct_log_probs = th.log_softmax(instruct_logits[subset_mask].float(), dim=-1)
    logits = logits[:, :-1][subset_mask[:, 1:]].float()
    labels = labels[:, 1:][subset_mask[:, 1:]]
    log_probs = th.log_softmax(logits, dim=-1)

    loss = chat_model.compute_loss(logits, labels, already_shifted=True)
    loss_wrt_instruct = chat_model.compute_loss(
        logits, instruct_preds, already_shifted=True
    )
    kl_instruct = kl_div(
        log_probs, instruct_log_probs, log_target=True, reduction="none"
    ).sum(dim=-1)
    kl_base = kl_div(log_probs, base_log_probs, log_target=True, reduction="none").sum(
        dim=-1
    )

    return {
        "loss": loss,
        "kl_instruct": kl_instruct,
        "kl_base": kl_base,
        "loss_wrt_instruct": loss_wrt_instruct,
        "num_samples": logits.shape[0],
    }


def update_metrics(metrics_dict, metrics, fn_name, prefix=""):
    """Update running metrics with new values."""
    metrics_dict["nlls"][fn_name].update(metrics["loss"])
    metrics_dict["instruct_kl"][fn_name].update(metrics["kl_instruct"])
    metrics_dict["base_kl"][fn_name].update(metrics["kl_base"])
    metrics_dict["nlls_wrt_instruct"][fn_name].update(metrics["loss_wrt_instruct"])
    metrics_dict["num_samples"][fn_name] += metrics["num_samples"]


def log_metrics(metrics, fn_name, step, prefix=""):
    """Log metrics to wandb with proper prefixing."""
    wandb.log(
        {
            f"{prefix}loss/{fn_name}": metrics["loss"].mean().item(),
            f"{prefix}perplexity/{fn_name}": th.exp(metrics["loss"].mean()).item(),
            f"{prefix}kl-instruct/{fn_name}": metrics["kl_instruct"].mean().item(),
            f"{prefix}kl-base/{fn_name}": metrics["kl_base"].mean().item(),
            f"{prefix}loss_wrt_instruct/{fn_name}": metrics["loss_wrt_instruct"]
            .mean()
            .item(),
            f"{prefix}perplexity_wrt_instruct/{fn_name}": th.exp(
                metrics["loss_wrt_instruct"].mean()
            ).item(),
        },
        step=step,
    )


def create_metrics_dict():
    """Create a dictionary of metric trackers."""
    return {
        "nlls": defaultdict(RunningMeanStd),
        "instruct_kl": defaultdict(RunningMeanStd),
        "base_kl": defaultdict(RunningMeanStd),
        "nlls_wrt_instruct": defaultdict(RunningMeanStd),
        "num_samples": defaultdict(int),
    }


def compute_final_metrics(metrics_dict, preprocess_before_last_half_fns):
    """Compute final metrics from running metrics."""
    return {
        "loss": {
            fn_name: metrics_dict["nlls"][fn_name].compute(return_dict=True)
            for fn_name in preprocess_before_last_half_fns
        },
        "kl-instruct": {
            fn_name: metrics_dict["instruct_kl"][fn_name].compute(return_dict=True)
            for fn_name in preprocess_before_last_half_fns
        },
        "kl-base": {
            fn_name: metrics_dict["base_kl"][fn_name].compute(return_dict=True)
            for fn_name in preprocess_before_last_half_fns
        },
        "loss_wrt_instruct_pred": {
            fn_name: metrics_dict["nlls_wrt_instruct"][fn_name].compute(
                return_dict=True
            )
            for fn_name in preprocess_before_last_half_fns
        },
    }


@th.inference_mode()
def evaluate_interventions(
    base_model,
    chat_model,
    dataset,
    preprocess_before_last_half_fns: dict[str, HalfStepPreprocessFn],
    layer_to_stop=13,
    batch_size=8,
    device="cuda",
    max_seq_len=1024,
    log_every=100,
    checkpoint_every=2000,
    k_first: int = 10,
    save_path: Path | None = None,
):
    # Initialize metric trackers for each subset
    metrics_all = create_metrics_dict()
    metrics_kfirst = create_metrics_dict()
    metrics_post_k_first = create_metrics_dict()

    base_model = split_gemma(base_model)
    chat_model = split_gemma(chat_model)

    def compute_result():
        return {
            "all": compute_final_metrics(metrics_all, preprocess_before_last_half_fns),
            "k_first": compute_final_metrics(
                metrics_kfirst, preprocess_before_last_half_fns
            ),
            "post_k_first": compute_final_metrics(
                metrics_post_k_first, preprocess_before_last_half_fns
            ),
        }

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        batch_tokens = tokenize_with_ctrl_mask(
            batch,
            tokenizer=chat_model.tokenizer,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = batch_tokens["input_ids"].to(device)
        attn_mask = batch_tokens["attention_mask"].to(device)
        assistant_mask = batch_tokens["assistant_masks"].to(device)
        if not assistant_mask.any():
            print("Got an empty batch, skipping...")
            continue
        ctrl_mask = batch_tokens["ctrl_mask"].to(device)
        _k_first_ass_toks_mask = mask_k_first_ones_vec(assistant_mask, k_first).to(
            device
        )
        # shift left by 1 to have the token at which you make the prediction rather than the token you need to predict
        k_first_pred_toks_mask = th.zeros_like(_k_first_ass_toks_mask)
        k_first_pred_toks_mask[:, :-1] = _k_first_ass_toks_mask[:, 1:]
        next_ass_toks_mask = assistant_mask & ~k_first_pred_toks_mask

        (
            base_activations,
            base_causal_mask_raw,
            base_position_ids,
            base_cache_position,
        ) = base_model.first_half_forward(
            input_ids=input_ids,
            attention_mask=attn_mask,
            layer_idx=layer_to_stop,
        )
        if base_cache_position is None:
            base_cache_position = th.arange(
                base_activations.shape[1], device=base_activations.device
            )

        (
            instruct_activations,
            instruct_causal_mask_raw,
            instruct_position_ids,
            instruct_cache_position,
        ) = chat_model.first_half_forward(
            input_ids=input_ids,
            attention_mask=attn_mask,
            layer_idx=layer_to_stop,
        )
        if instruct_cache_position is None:
            instruct_cache_position = th.arange(
                instruct_activations.shape[1], device=instruct_activations.device
            )

        base_logits_raw = base_model.second_half_forward(
            base_activations,
            base_causal_mask_raw,
            base_position_ids,
            layer_idx=layer_to_stop,
            return_dict=True,
            cache_position=base_cache_position,
        ).logits
        instruct_logits_raw = chat_model.second_half_forward(
            instruct_activations,
            instruct_causal_mask_raw,
            instruct_position_ids,
            layer_idx=layer_to_stop,
            return_dict=True,
            cache_position=instruct_cache_position,
        ).logits

        for fn_name, preprocess_fn in preprocess_before_last_half_fns.items():
            base_activations_edited, instruct_activations_edited, mask = preprocess_fn(
                base_activations,
                instruct_activations,
                ctrl_mask=ctrl_mask,
                assistant_mask=assistant_mask,
                next_ass_toks_mask=next_ass_toks_mask,
                k_first_pred_toks_mask=k_first_pred_toks_mask,
            )

            if mask is not None:
                base_logits = base_logits_raw[mask]
                instruct_logits = instruct_logits_raw[mask]
                base_causal_mask = base_causal_mask_raw[mask]
                instruct_causal_mask = instruct_causal_mask_raw[mask]
                effective_assistant_mask = assistant_mask[mask]
                effective_k_first = k_first_pred_toks_mask[mask]
                effective_post_k_first = next_ass_toks_mask[mask]
                labels = input_ids[mask]
            else:
                base_logits = base_logits_raw
                instruct_logits = instruct_logits_raw
                base_causal_mask = base_causal_mask_raw
                instruct_causal_mask = instruct_causal_mask_raw
                effective_assistant_mask = assistant_mask
                effective_k_first = k_first_pred_toks_mask
                effective_post_k_first = next_ass_toks_mask
                labels = input_ids

            final_out = None
            if base_activations_edited is not None:
                final_out = base_model.second_half_forward(
                    base_activations_edited,
                    base_causal_mask,
                    base_position_ids,
                    layer_idx=layer_to_stop,
                    return_dict=True,
                    cache_position=base_cache_position,
                )
            elif instruct_activations_edited is not None:
                final_out = chat_model.second_half_forward(
                    instruct_activations_edited,
                    instruct_causal_mask,
                    instruct_position_ids,
                    layer_idx=layer_to_stop,
                    return_dict=True,
                    cache_position=instruct_cache_position,
                )

            if final_out is not None:
                # Compute metrics for all tokens
                for metrics_dict, prefix, mask in [
                    (metrics_all, "", effective_assistant_mask),
                    (metrics_kfirst, "k_first_", effective_k_first),
                    (metrics_post_k_first, "post_k_first_", effective_post_k_first),
                ]:
                    if mask.any():
                        metrics = compute_metrics_for_subset(
                            final_out.logits,
                            labels,
                            base_logits,
                            instruct_logits,
                            mask,
                            chat_model,
                        )
                        update_metrics(metrics_dict, metrics, fn_name, prefix)
                        log_metrics(metrics, fn_name, i, prefix)
                    else:
                        print(f"Skipping {fn_name} and {prefix} because mask is empty")

            if i % log_every == 0:
                # Log running metrics
                for metrics_dict, prefix in [
                    (metrics_all, ""),
                    (metrics_kfirst, "k_first_"),
                    (metrics_post_k_first, "post_k_first_"),
                ]:
                    wandb.log(
                        {
                            f"{prefix}perplexity_running/{fn_name}": th.exp(
                                metrics_dict["nlls"][fn_name].compute()[0]
                            ).item(),
                            f"{prefix}loss_running/{fn_name}": metrics_dict["nlls"][
                                fn_name
                            ]
                            .compute()[0]
                            .item(),
                            f"{prefix}kl-instruct_running/{fn_name}": metrics_dict[
                                "instruct_kl"
                            ][fn_name]
                            .compute()[0]
                            .item(),
                            f"{prefix}kl-base_running/{fn_name}": metrics_dict[
                                "base_kl"
                            ][fn_name]
                            .compute()[0]
                            .item(),
                            f"{prefix}loss_wrt_instruct_pred_running/{fn_name}": metrics_dict[
                                "nlls_wrt_instruct"
                            ][
                                fn_name
                            ]
                            .compute()[0]
                            .item(),
                            f"{prefix}perplexity_wrt_instruct_pred_running/{fn_name}": th.exp(
                                metrics_dict["nlls_wrt_instruct"][fn_name].compute()[0]
                            ).item(),
                            f"{prefix}num_samples/{fn_name}": metrics_dict[
                                "num_samples"
                            ][fn_name],
                        },
                        step=i,
                    )

            if i % checkpoint_every == 0 and save_path is not None and i != 0:
                with open(save_path / f"{wandb.run.name}_{i}_result.json", "w") as f:
                    json.dump(compute_result(), f)

    return compute_result()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer-to-stop", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="science-of-finetuning/ultrachat_200k_gemma-2-2b-it-generated",
    )
    parser.add_argument("--dataset-col", type=str, default="messages")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--base-device", type=str, default="cuda")
    parser.add_argument("--chat-device", type=str, default="cuda")
    parser.add_argument("--crosscoder", type=str, default="l13_crosscoder")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-path", type=Path, default=Path("results"))
    args = parser.parse_args()
    chat_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=th.bfloat16,
        device_map=args.chat_device,
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    tokenizer.chat_template = gemma_chat_template
    chat_model.tokenizer = tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map=args.base_device,
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
    )
    dataset = load_dataset(args.dataset, split=args.split)
    if args.test:
        dataset = dataset.select(range(20))
    else:
        dataset = dataset.select(range(min(30_000, len(dataset))))
    device = (
        args.device
        if args.device != "auto"
        else "cuda" if th.cuda.is_available() else "cpu"
    )
    crosscoder = load_crosscoder(args.crosscoder).to(device)
    if args.name is None and args.test:
        args.name = "test"
    project = "perplexity-comparison"
    if args.test:
        project += "-test"
    wandb.init(project=project, name=args.name)
    seeds = list(range(args.num_seeds))
    fn_dict = create_acl_half_fns(crosscoder, seeds, args.crosscoder)
    result = evaluate_interventions(
        base_model,
        chat_model,
        dataset[args.dataset_col],
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
