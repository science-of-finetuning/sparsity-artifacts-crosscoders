import time
import sys
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from datasets import load_dataset
import torch as th

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils.edit_eval_results import add_random_means
from tools.compute_utils import RunningMeanStd, chunked_kl
from tools.setup_to_eval import HalfStepPreprocessFn
from tools.split_gemma import split_gemma
from tools.tokenization_utils import tokenize_with_ctrl_mask, patch_tokenizer
from tools.utils import mask_k_first_ones_vec
from tools.setup_to_eval import (
    create_acl_half_fns,
    create_acl_vanilla_half_fns,
    create_acl_patching_half_fns,
    create_acl_crosscoder_half_fns,
    arxiv_paper_half_fns,
    sae_steering_half_fns,
)
from dictionary_learning.dictionary import CrossCoder
from tools.cc_utils import load_dictionary_model, load_latent_df
from coolname import generate_slug


def compute_metrics_for_subset(
    logits, labels, base_logits, instruct_logits, subset_mask, chat_model
):
    """Compute all metrics for a subset of tokens.
    Args:
        logits: Logits of the model
        labels: Labels of the model
        base_logits: Logits of the base model
        instruct_logits: Logits of the instruct model
        subset_mask: Mask of the subset of tokens at which the model is making predictions
        chat_model: Chat model
    """
    # Verify shapes match
    assert (
        base_logits.shape == instruct_logits.shape
    ), f"Base and instruct logits must have same shape, got {base_logits.shape} and {instruct_logits.shape}"
    assert (
        labels.shape == subset_mask.shape
    ), f"Labels and subset mask must have same shape, got {labels.shape} and {subset_mask.shape}"
    assert (
        logits.shape[:2] == labels.shape
    ), f"Logits and labels batch/sequence dims must match, got {logits.shape} and {labels.shape}"
    assert (
        base_logits.shape == logits.shape
    ), f"Base and logits must have same shape, got {base_logits.shape} and {logits.shape}"
    base_logits = base_logits[:, :-1][subset_mask[:, :-1]]
    instruct_logits = instruct_logits[:, :-1][subset_mask[:, :-1]]
    logits = logits[:, :-1][subset_mask[:, :-1]].float()
    labels = labels[:, 1:][subset_mask[:, :-1]]
    instruct_preds = th.argmax(instruct_logits, dim=-1)
    num_samples = logits.shape[0]

    loss = chat_model.compute_loss(logits, labels, already_shifted=True)
    loss_wrt_instruct = chat_model.compute_loss(
        logits, instruct_preds, already_shifted=True
    )
    assert (
        logits.shape == instruct_logits.shape
    ), "Log probs and instruct logits must have same shape"
    assert (
        logits.shape == base_logits.shape
    ), "Log probs and base logits must have same shape"
    kl_instruct = chunked_kl(logits, instruct_logits)
    kl_base = chunked_kl(logits, base_logits)
    th.cuda.empty_cache()
    return {
        "loss": loss,
        "kl_instruct": kl_instruct,
        "kl_base": kl_base,
        "loss_wrt_instruct": loss_wrt_instruct,
        "num_samples": num_samples,
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
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    # Initialize metric trackers for each subset
    metrics_all = create_metrics_dict()
    metrics_kfirst = create_metrics_dict()
    metrics_post_k_first = create_metrics_dict()

    base_model = split_gemma(base_model)
    chat_model = split_gemma(chat_model)
    patch_tokenizer(
        chat_model.tokenizer,
        "google/gemma-2-2b-it",
    )

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
        assistant_mask = batch_tokens["assistant_masks"]
        ctrl_mask = batch_tokens["ctrl_mask"].to(device)
        # shift the assistant mask to the left by 1 to have the token at which you make the prediction rather than the token you need to predict
        assistant_pred_mask = th.zeros_like(assistant_mask)
        assistant_pred_mask[:, :-1] = assistant_mask[:, 1:]
        assistant_pred_mask = assistant_pred_mask.to(device)
        k_first_pred_toks_mask = (
            mask_k_first_ones_vec(assistant_pred_mask, k_first).to(device) & ~ctrl_mask
        )
        next_ass_toks_mask = assistant_pred_mask & ~k_first_pred_toks_mask
        if not assistant_pred_mask.any():
            print("Got an empty batch, skipping...")
            continue

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
                assistant_pred_mask=assistant_pred_mask,
                next_ass_toks_mask=next_ass_toks_mask,
                k_first_pred_toks_mask=k_first_pred_toks_mask,
            )

            if mask is not None:
                base_logits = base_logits_raw[mask]
                instruct_logits = instruct_logits_raw[mask]
                base_causal_mask = base_causal_mask_raw[mask]
                instruct_causal_mask = instruct_causal_mask_raw[mask]
                effective_assistant_mask = assistant_pred_mask[mask]
                effective_k_first = k_first_pred_toks_mask[mask]
                effective_post_k_first = next_ass_toks_mask[mask]
                labels = input_ids[mask]
            else:
                base_logits = base_logits_raw
                instruct_logits = instruct_logits_raw
                base_causal_mask = base_causal_mask_raw
                instruct_causal_mask = instruct_causal_mask_raw
                effective_assistant_mask = assistant_pred_mask
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
                        # cleanup
                        for tensor in metrics.values():
                            del tensor
                        del metrics
                        th.cuda.empty_cache()
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
                    result = compute_result()
                    result = add_random_means(result)
                    json.dump(result, f)

    return compute_result()


# python scripts/evaluate_interventions_effects.py --dataset science-of-finetuning/lmsys-chat-1m-chat-formatted --dataset-col conversation --split validation --name lmsys-chat-1m-validation-beta-cols --columns beta_ratio_reconstruction beta_ratio_error

# python scripts/evaluate_interventions_effects.py --dataset science-of-finetuning/lmsys-chat-1m-chat-formatted --dataset-col conversation --split validation --name lmsys-chat-1m-validation-others-cols --columns rank_sum "base uselessness score"


# python scripts/evaluate_interventions_effects.py --name ultrachat-gemma-all-columns


# python scripts/evaluate_interventions_effects.py --name ultrachat-gemma-all-new-columns --skip-target-patch --skip-vanilla --skip-patching --columns "dec_norm_diff" "lmsys_freq" "lmsys_ctrl_%" "lmsys_ctrl_freq" "lmsys_avg_act" "beta_activation_ratio" "beta_activation_chat" "beta_activation_base" "beta_error_chat" "beta_error_base"
# python scripts/evaluate_interventions_effects.py --name ultrachat-gemma-concat-sae --df-path results/eval_crosscoder/gemma-2-2b-L13-mu5.2e-02-lr1e-04-local-shuffling-SAEloss_model_final.pt/data/feature_df.csv --chat-only-indices /workspace/data/latent_indices/gemma-2-2b-L13-mu5.2e-02-lr1e-04-local-shuffling-SAEloss/low_norm_diff_indices_2839.pt --crosscoder /workspace/julian/repositories/representation-structure-comparison/checkpoints/gemma-2-2b-L13-mu5.2e-02-lr1e-04-local-shuffling-SAEloss/model_final.pt

# python scripts/evaluate_interventions_effects.py --name ultrachat-gemma-minicc --df-path results/eval_crosscoder/gemma-2-2b-L13-mu4.1e-02-lr1e-04-local-shuffling-CCloss_model_final.pt/data/feature_df.csv --crosscoder /workspace/julian/repositories/representation-structure-comparison/checkpoints/gemma-2-2b-L13-mu4.1e-02-lr1e-04-local-shuffling-CCloss/model_final.pt

# python scripts/evaluate_interventions_effects.py --name ultrachat-gemma-sae --crosscoder /workspace/julian/repositories/representation-structure-comparison/checkpoints/SAE-chat-gemma-2-2b-L13-k100-lr1e-04-local-shuffling/model_final.pt --df-path /workspace/julian/repositories/representation-structure-comparison/checkpoints/SAE-chat-gemma-2-2b-L13-k100-lr1e-04-local-shuffling/SAE-chat-gemma-2-2b-L13-k100-lr1e-04-local-shuffling.csv

# python scripts/evaluate_interventions_effects.py --name ultrachat-gemma-concat-sae-200M --df-path /workspace/julian/repositories/representation-structure-comparison/results/eval_crosscoder/gemma-2-2b-L13-mu5.2e-02-lr1e-04-2x100M-local-shuffling-SAELoss_model_final.pt/data/feature_df.csv --chat-only-indices /workspace/data/latent_indices/gemma-2-2b-L13-mu5.2e-02-lr1e-04-2x100M-local-shuffling-SAELoss/low_norm_diff_indices_3176.pt --crosscoder /workspace/julian/repositories/representation-structure-comparison/checkpoints/gemma-2-2b-L13-mu5.2e-02-lr1e-04-2x100M-local-shuffling-SAELoss/model_final.pt

# python scripts/evaluate_interventions_effects.py --name ultrachat-gemma-batchtopk-CC --df-path /workspace/julian/repositories/representation-structure-comparison/results/eval_crosscoder/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss_model_final.pt/data/feature_df.csv --chat-only-indices /workspace/data/latent_indices/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss/low_norm_diff_indices_3176.pt --crosscoder /workspace/julian/repositories/representation-structure-comparison/checkpoints/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss/model_final.pt
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
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-path", type=Path, default=Path("results/interv_effects")
    )
    parser.add_argument("--k-first", type=int, default=10)
    parser.add_argument("--checkpoint", type=int, default=10)
    parser.add_argument(
        "--percentage", type=int, nargs="+", default=[5, 10, 30, 50, 100]
    )
    parser.add_argument("--is-sae", action="store_true")
    # parser.add_argument(
    #     "--columns",
    #     nargs="+",
    #     default=[
    #         "rank_sum",
    #         "beta_ratio_reconstruction",
    #         "beta_ratio_error",
    #         "base uselessness score",
    #         "dec_norm_diff",
    #     ],
    # )
    # parser.add_argument("--skip-target-patch", action="store_true")
    # parser.add_argument("--skip-vanilla", action="store_true")
    # parser.add_argument("--skip-patching", action="store_true")
    parser.add_argument("--add-base-only-latents", action="store_true")
    parser.add_argument("--df-path", type=Path, default=None)
    parser.add_argument("--chat-only-indices", type=Path, default=None)
    args = parser.parse_args()
    print(f"using args: {args}")
    dataset = load_dataset(args.dataset, split=args.split)
    device = (
        args.device
        if args.device != "auto"
        else "cuda" if th.cuda.is_available() else "cpu"
    )
    seeds = list(range(args.num_seeds))
    dictionary = load_dictionary_model(args.dictionary, is_sae=args.is_sae).to(device)
    percentages = args.percentage
    # fn_dict, infos = create_acl_half_fns(
    #     dictionary,
    #     seeds,
    #     args.dictionary,
    #     percentages,
    #     args.columns,
    #     skip_target_patch=args.skip_target_patch,
    #     skip_vanilla=args.skip_vanilla,
    #     skip_patching=args.skip_patching,
    #     add_base_only_latents=args.add_base_only_latents,
    # )
    if args.df_path is not None:
        print(f"Loading df from {args.df_path}")
        df = load_latent_df(args.df_path)
    else:
        print(f"Loading df from dictionary {args.dictionary}")
        df = load_latent_df(args.dictionary)
    if args.chat_only_indices is not None:
        chat_only_indices = th.load(args.chat_only_indices).tolist()
        df = df.iloc[chat_only_indices]
        df["tag"] = "Chat only"

    print(f"len df: {len(df)}")
    if not isinstance(dictionary, CrossCoder):
        fn_dict, infos = sae_steering_half_fns(
            dictionary,
            seeds,
            df,
        )
    else:
        fn_dict, infos = arxiv_paper_half_fns(
            dictionary,
            df,
            add_base_only_latents=args.add_base_only_latents,
        )
    if args.test:
        fn_dict = create_acl_vanilla_half_fns()
        fn_dict.update(create_acl_patching_half_fns())
        dataset = dataset.select(range(120))
    else:
        dataset = dataset.select(range(min(30_000, len(dataset))))
    print(f"running {len(fn_dict)} interventions")
    if args.name is None and args.test:
        args.name = "test"
        print(f"Testing using {args.name}")
    name = "_" + args.name if args.name is not None else ""
    args.name = str(int(time.time())) + name + "_" + generate_slug(2)
    project = "perplexity-comparison"
    if args.test:
        project += "-test"
    wandb.init(project=project, name=args.name)
    wdb_name = wandb.run.name
    (args.save_path / wdb_name).mkdir(parents=True, exist_ok=True)
    with open(args.save_path / wdb_name / "metadata.json", "w") as f:
        json.dump(
            {
                "infos": infos,
                "args": {
                    "named_args": {
                        k: str(v)
                        for k, v in args.__dict__.items()
                        if not k.startswith("_")
                    }
                },
            },
            f,
        )
    # input("breakpoint")
    chat_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=th.bfloat16,
        device_map=args.chat_device,
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    chat_model.tokenizer = tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map=args.base_device,
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
    )
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
        save_path=args.save_path / wandb.run.name,
        checkpoint_every=args.checkpoint,
    )

    # Add mean across random seeds
    result = add_random_means(result)

    args.save_path.mkdir(parents=True, exist_ok=True)
    with open(args.save_path / f"{wdb_name}_result.json", "w") as f:
        json.dump(result, f)
