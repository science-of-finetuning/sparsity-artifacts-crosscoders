from dictionary_learning.dictionary import CrossCoder
from collections import defaultdict
import einops
import torch as th
from tools.split_gemma import split_gemma
from argparse import ArgumentParser
from tqdm.auto import tqdm
import pandas as pd
import json
from abc import ABC, abstractmethod
from torch.nn.functional import relu
import wandb
from dlabutils import model_path
from pathlib import Path
from torch.nn.functional import kl_div
from torchmetrics.aggregation import MeanMetric

CHAT_TEMPLATE = open("templates/gemma_chat_template.jinja").read()


class HalfStepPreprocessFn(ABC):
    @abstractmethod
    def preprocess(
        self, base_activations, instruct_activations
    ) -> tuple[th.Tensor, None] | tuple[None, th.Tensor]:
        """
        Preprocess the activations before the last half forward.
        Returns activations, None if the base model should finish the forward pass.
        Returns None, activations if the instruct model should finish the forward pass.
        """
        raise NotImplementedError

    def __call__(self, base_activations, instruct_activations):
        return self.preprocess(base_activations, instruct_activations)


class IdentityPreprocessFn(HalfStepPreprocessFn):
    def __init__(self, continue_with_base: bool = False):
        self.continue_with_base = continue_with_base

    def continue_with_model(self, result):
        return (result, None) if self.continue_with_base else (None, result)

    def preprocess(self, base_activations, instruct_activations):
        return (
            (base_activations, None)
            if self.continue_with_base
            else (None, instruct_activations)
        )


class SwitchPreprocessFn(HalfStepPreprocessFn):
    def __init__(self, continue_with_base: bool = False):
        self.continue_with_base = continue_with_base

    def preprocess(self, base_activations, instruct_activations):
        return (
            (instruct_activations, None)
            if self.continue_with_base
            else (None, base_activations)
        )


class CrossCoderReconstruction(IdentityPreprocessFn):
    def __init__(
        self,
        crosscoder: CrossCoder,
        reconstruct_with_base: bool = False,
        continue_with_base: bool = False,
    ):
        super().__init__(continue_with_base)
        self.crosscoder = crosscoder
        self.reconstruct_with_base = reconstruct_with_base

    def preprocess(self, base_activations, instruct_activations):
        cc_input = th.stack(
            [base_activations, instruct_activations], dim=2
        ).float()  # b, seq, 2, d
        cc_input = einops.rearrange(cc_input, "b s m d -> (b s) m d")
        f = self.crosscoder.encode(cc_input)  # (b s) D
        reconstruction = th.einsum(
            "bD, Dd->bd",
            f,
            self.crosscoder.decoder.weight[0 if self.reconstruct_with_base else 1],
        )  # (b s) d
        reconstruction = einops.rearrange(
            reconstruction, "(b s) d -> b s d", b=base_activations.shape[0]
        )
        return self.continue_with_model(reconstruction.bfloat16())


class CrossCoderSteeringFeature(IdentityPreprocessFn):
    def __init__(
        self,
        crosscoder: CrossCoder,
        steer_base_activations: bool,
        steer_with_base_features: bool,
        features_to_steer: list[int] | None,
        continue_with_base: bool,
    ):
        super().__init__(continue_with_base)
        if features_to_steer is None:
            features_to_steer = list(range(crosscoder.decoder.weight.shape[1]))
        self.encoder_weight = crosscoder.encoder.weight[:, :, features_to_steer]  # ldf
        self.decoder_weight = crosscoder.decoder.weight[:, features_to_steer]  # lfd
        self.steer_with_base_features = steer_with_base_features
        self.steer_base_activations = steer_base_activations

    def preprocess(self, base_activations, instruct_activations):
        cc_input = th.stack(
            [base_activations, instruct_activations], dim=2
        ).float()  # b, seq, 2, d
        f = relu(
            th.einsum("bsld, ldf -> bsf", cc_input, self.encoder_weight)
        )  # b, seq, f
        decoded = th.einsum("bsf, lfd -> bsld", f, self.decoder_weight)
        steering_feature = decoded[:, :, 1, :] - decoded[:, :, 0, :]
        if self.steer_with_base_features:
            steering_feature = -steering_feature
        act = base_activations if self.steer_base_activations else instruct_activations
        res = act + steering_feature
        return self.continue_with_model(res.bfloat16())


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
    save_path: Path | None = None,
):
    MeanMetricToDevice = lambda: MeanMetric().to(device)
    nlls = defaultdict(MeanMetricToDevice)
    perplexity = defaultdict(MeanMetricToDevice)
    instruct_kl = defaultdict(MeanMetricToDevice)
    base_kl = defaultdict(MeanMetricToDevice)
    nlls_wrt_instruct_pred = defaultdict(MeanMetricToDevice)
    perplexity_wrt_instruct_pred = defaultdict(MeanMetricToDevice)
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
            base_activations_edited, instruct_activations_edited = (
                preprocess_before_last_half_fn(base_activations, instruct_activations)
            )
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
                    with open(save_path / f"{wandb.run.name}_result.json", "w") as f:
                        json.dump(compute_result(), f)

    return compute_result()


def create_half_fn_dict_main(
    crosscoder: CrossCoder,
    it_only_features: list[int],
    base_only_features: list[int],
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}

    half_fns["1-instruct"] = IdentityPreprocessFn(continue_with_base=False)

    # steer the base activations with all features from IT decoder and continue with instruct   ===   replace it error with base error
    half_fns["2-steer_all"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=True,
        steer_with_base_features=False,
        continue_with_base=False,
        features_to_steer=None,
    )

    half_fns["3-base_to_instruct"] = SwitchPreprocessFn(continue_with_base=False)

    # steer the base activations with the IT only features from instruct decoder and continue with instruct
    half_fns["4-steer_it_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=True,
        steer_with_base_features=False,
        continue_with_base=False,
        features_to_steer=it_only_features,
    )
    # 5/ RANDOM

    # instruct reconstruction
    half_fns["6-instruct_reconstruct"] = CrossCoderReconstruction(
        crosscoder, reconstruct_with_base=False, continue_with_base=False
    )
    # Take the instruct activations and remove the IT only features
    half_fns["7-remove_it_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=False,
        steer_with_base_features=True,
        continue_with_base=False,
        features_to_steer=it_only_features,
    )

    # steer the base activations with the IT only & base only features from instruct decoder and continue with instruct
    half_fns["?-steer_it_and_base_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=True,
        steer_with_base_features=False,
        continue_with_base=False,
        features_to_steer=it_only_features + base_only_features,
    )

    # reconstruct instruct and finish with instruct

    return half_fns


def create_half_fn_dict_secondary(
    crosscoder: CrossCoder, it_only_features: list[int], base_only_features: list[int]
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    half_fns["base_reconstruct"] = CrossCoderReconstruction(
        crosscoder, reconstruct_with_base=True, continue_with_base=True
    )
    # steer the base activations with the IT only features from instruct decoder and continue with base
    half_fns["steer_it_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=True,
        steer_with_base_features=False,
        continue_with_base=True,
        features_to_steer=it_only_features,
    )
    # steer the base activations with the IT only & base only features from instruct decoder and continue with base
    half_fns["steer_it_and_base_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=True,
        steer_with_base_features=False,
        continue_with_base=True,
        features_to_steer=it_only_features + base_only_features,
    )
    # steer the base activations with the base only features from instruct decoder and continue with base
    half_fns["steer_base_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=True,
        steer_with_base_features=False,
        continue_with_base=True,
        features_to_steer=base_only_features,
    )
    # steer the base activations with all features from IT decoder and continue with base   ===   replace it error with base error
    half_fns["steer_all_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=True,
        steer_with_base_features=False,
        continue_with_base=True,
        features_to_steer=None,
    )
    # steer the base activations with the base only features from instruct decoder and continue with instruct
    half_fns["steer_base_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_base_activations=True,
        steer_with_base_features=False,
        continue_with_base=False,
        features_to_steer=base_only_features,
    )

    return half_fns


def create_half_fn_dict_seeds(
    crosscoder: CrossCoder, seeds: list[int], num_features: int
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    for seed in seeds:
        th.manual_seed(seed)
        features_to_steer = th.randperm(crosscoder.decoder.weight.shape[1])[
            :num_features
        ]
        # half_fns[f"5-random_s{seed}"] = CrossCoderSteeringFeature(
        #     crosscoder,
        #     steer_base_activations=True,
        #     steer_with_base_features=False,
        #     continue_with_base=False,
        #     features_to_steer=features_to_steer,
        # )
        half_fns[f"?-remove_random_s{seed}"] = CrossCoderSteeringFeature(
            crosscoder,
            steer_base_activations=False,
            steer_with_base_features=True,
            continue_with_base=False,
            features_to_steer=features_to_steer,
        )
    return half_fns


def create_half_fn_dict_no_cross() -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    half_fns["base"] = IdentityPreprocessFn(continue_with_base=True)
    half_fns["instruct - debugging"] = IdentityPreprocessFn(continue_with_base=False)
    half_fns["instruct_to_base"] = SwitchPreprocessFn(continue_with_base=True)
    return half_fns


def create_it_only_ft_fn_dict(
    crosscoder: CrossCoder, it_only_features: list[int], base_only_features: list[int]
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {
        "remove_it_only_custom": CrossCoderSteeringFeature(
            crosscoder,
            steer_base_activations=False,
            steer_with_base_features=True,
            continue_with_base=False,
            features_to_steer=it_only_features,
        ),
        "remove_it_and_base_only_custom": CrossCoderSteeringFeature(
            crosscoder,
            steer_base_activations=False,
            steer_with_base_features=True,
            continue_with_base=False,
            features_to_steer=it_only_features + base_only_features,
        ),
        "steer_it_only_custom_to_base": CrossCoderSteeringFeature(
            crosscoder,
            steer_base_activations=True,
            steer_with_base_features=False,
            continue_with_base=True,
            features_to_steer=it_only_features,
        ),
        "steer_it_only_custom": CrossCoderSteeringFeature(
            crosscoder,
            steer_base_activations=True,
            steer_with_base_features=False,
            continue_with_base=False,
            features_to_steer=it_only_features,
        ),
        "steer_it_and_base_only_custom": CrossCoderSteeringFeature(
            crosscoder,
            steer_base_activations=True,
            steer_with_base_features=False,
            continue_with_base=False,
            features_to_steer=it_only_features + base_only_features,
        ),
    }
    return half_fns


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
        dataset = dataset.select(range(100))
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
    fn_dict.update(create_it_only_ft_fn_dict(crosscoder, it_only_features, base_only_features))
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
