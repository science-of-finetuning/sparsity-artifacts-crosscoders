import time
import sys
import json
from argparse import ArgumentParser
from collections import defaultdict
from typing import Literal
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from datasets import load_dataset
import torch as th
import pandas as pd
from coolname import generate_slug
from loguru import logger
from dictionary_learning.dictionary import Dictionary

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils.edit_eval_results import add_random_means
from tools.compute_utils import RunningMeanStd, chunked_kl
from tools.setup_to_eval import HalfStepPreprocessFn
from tools.split_model import split_model
from tools.tokenization_utils import tokenize_with_ctrl_mask, patch_tokenizer
from tools.utils import mask_k_first_ones_vec, load_hf_model
from tools.setup_to_eval import (
    # create_acl_half_fns,
    # create_acl_vanilla_half_fns,
    # create_acl_patching_half_fns,
    # create_acl_crosscoder_half_fns,
    arxiv_paper_half_fns,
    sae_steering_half_fns,
    baselines_half_fns,
)
from tools.configs import MODEL_CONFIGS
from dictionary_learning.dictionary import CrossCoder
from tools.cc_utils import load_dictionary_model, load_latent_df


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
    tokenizer_name,
    dataset,
    preprocess_before_last_half_fns: dict[str, HalfStepPreprocessFn],
    layer_to_stop,
    run_name: str,
    batch_size=8,
    device="cuda",
    max_seq_len=1024,
    log_every=100,
    checkpoint_every=2000,
    k_first: int = 10,
    save_path: Path | None = None,
    token_level_replacement: str | None = None,
    max_num_tokens: int | None = None,
    ignore_first_n_tokens: int | None = None,
):
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    # Initialize metric trackers for each subset
    metrics_all = create_metrics_dict()
    metrics_kfirst = create_metrics_dict()
    metrics_post_k_first = create_metrics_dict()

    tokenizer = patch_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name, padding_side="right"),
        tokenizer_name,
    )
    base_model = split_model(base_model, tokenizer)
    chat_model = split_model(chat_model, tokenizer)

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

    total_num_tokens = 0
    pbar = (
        tqdm(range(0, len(dataset), batch_size))
        if max_num_tokens is None
        else tqdm(range(0, len(dataset), batch_size), total=max_num_tokens)
    )
    for i in pbar:
        batch = dataset[i : i + batch_size]
        try:
            batch_tokens = tokenize_with_ctrl_mask(
                batch,
                tokenizer=tokenizer,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
        except Exception as e:
            print(f"Error tokenizing batch {i}: {e}")
            print(batch[0])
            print()
            print(batch)
            raise
        input_ids = batch_tokens["input_ids"].to(device)
        if token_level_replacement is not None:
            base_input_ids = input_ids.clone()
            for old_token_id, new_token_id in token_level_replacement.items():
                base_input_ids[base_input_ids == old_token_id] = new_token_id
        else:
            base_input_ids = input_ids
        attn_mask = batch_tokens["attention_mask"].to(device)
        assistant_mask = batch_tokens["assistant_masks"]
        new_tokens = assistant_mask.sum().item()
        total_num_tokens += new_tokens
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
            continue
        (
            base_outputs,
            base_activations,
            base_first_half_args,
        ) = base_model.full_forward_collect_hidden_states(
            input_ids=base_input_ids,
            attention_mask=attn_mask,
            layer_idx=layer_to_stop,
        )

        (
            instruct_outputs,
            instruct_activations,
            instruct_first_half_args,
        ) = chat_model.full_forward_collect_hidden_states(
            input_ids=input_ids,
            attention_mask=attn_mask,
            layer_idx=layer_to_stop,
        )
        base_logits_raw = base_outputs.logits
        instruct_logits_raw = instruct_outputs.logits

        for fn_name, preprocess_fn in preprocess_before_last_half_fns.items():
            base_activations_edited, instruct_activations_edited, mask = preprocess_fn(
                base_activations,
                instruct_activations,
                ctrl_mask=ctrl_mask,
                assistant_pred_mask=assistant_pred_mask,
                next_ass_toks_mask=next_ass_toks_mask,
                k_first_pred_toks_mask=k_first_pred_toks_mask,
            )

            if mask is not None:  # todo: add a apply_mask method in split_model
                raise NotImplementedError(
                    "Mask are deprecated temporarily, see todo above"
                )
            #     base_logits = base_logits_raw[mask]
            #     instruct_logits = instruct_logits_raw[mask]
            #     base_causal_mask = base_causal_mask_raw[mask]
            #     instruct_causal_mask = instruct_causal_mask_raw[mask]
            #     effective_assistant_mask = assistant_pred_mask[mask]
            #     effective_k_first = k_first_pred_toks_mask[mask]
            #     effective_post_k_first = next_ass_toks_mask[mask]
            #     labels = input_ids[mask]
            else:
                base_logits = base_logits_raw
                instruct_logits = instruct_logits_raw
                # base_causal_mask = base_causal_mask_raw
                # instruct_causal_mask = instruct_causal_mask_raw
                effective_assistant_mask = assistant_pred_mask
                effective_k_first = k_first_pred_toks_mask
                effective_post_k_first = next_ass_toks_mask
                labels = None
                if instruct_activations_edited is not None:
                    labels = input_ids
                elif base_activations_edited is not None:
                    labels = base_input_ids

            final_logits = None
            if base_activations_edited is not None:
                assert (
                    base_activations_edited.shape == base_activations.shape
                ), f"Base activations edited shape {base_activations_edited.shape} does not match base activations shape {base_activations.shape} for fn {fn_name}"
                if ignore_first_n_tokens is not None:
                    base_activations_edited[:, :ignore_first_n_tokens] = (
                        base_activations[:, :ignore_first_n_tokens]
                    )
                final_logits = base_model.second_half_forward(
                    base_activations_edited,
                    base_first_half_args,
                    return_dict=True,
                    layer_idx=layer_to_stop,
                ).logits
            elif instruct_activations_edited is not None:
                assert (
                    instruct_activations_edited.shape == instruct_activations.shape
                ), f"Instruct activations edited shape {instruct_activations_edited.shape} does not match instruct activations shape {instruct_activations.shape} for fn {fn_name}"
                if ignore_first_n_tokens is not None:
                    instruct_activations_edited[:, :ignore_first_n_tokens] = (
                        instruct_activations[:, :ignore_first_n_tokens]
                    )
                final_logits = chat_model.second_half_forward(
                    instruct_activations_edited,
                    instruct_first_half_args,
                    return_dict=True,
                    layer_idx=layer_to_stop,
                ).logits

            if final_logits is not None:
                # Compute metrics for all tokens
                for metrics_dict, prefix, mask in [
                    (metrics_all, "", effective_assistant_mask),
                    (metrics_kfirst, "k_first_", effective_k_first),
                    (metrics_post_k_first, "post_k_first_", effective_post_k_first),
                ]:
                    if mask.any():
                        metrics = compute_metrics_for_subset(
                            final_logits,
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
                with open(save_path / f"{run_name}_{i}_result.json", "w") as f:
                    result = compute_result()
                    result = add_random_means(result)
                    json.dump(result, f)
        if i % log_every == 0:
            wandb.log(
                {
                    "total_num_tokens": total_num_tokens,
                },
                step=i,
            )
        if max_num_tokens is None:
            pbar.set_postfix(total_num_tokens=total_num_tokens)
        else:
            pbar.update(new_tokens)
            if total_num_tokens >= max_num_tokens:
                break
    logger.info(f"Total number of tokens: {total_num_tokens}")
    return compute_result(), total_num_tokens


def kl_experiment(
    # Model parameters
    dictionary: Dictionary | None,
    base_model: AutoModelForCausalLM,
    chat_model: AutoModelForCausalLM,
    tokenizer_name: str,
    # Dataset parameters
    dataset_name: str,
    split: str,
    dataset_col: str,
    layer_to_stop,
    # Dictionary parameters
    latent_df: pd.DataFrame | None,
    chat_only_indices: list[int] | None = None,
    add_base_only_latents: bool = False,
    is_sae: bool = False,
    is_difference_sae: bool = False,
    # Evaluation parameters
    batch_size: int = 6,
    device: str = "cuda",
    max_seq_len: int = 1024,
    log_every: int = 10,
    k_first: int = 10,
    checkpoint_every: int = 10,
    num_seeds: int = 5,
    # Output parameters
    save_path: Path = Path("results/interv_effects"),
    name: str | None = None,
    dictionary_name: str | None = None,
    model_name: str | None = None,
    test: bool = False,
    token_level_replacement: str | None = None,
    ignore_first_n_tokens: int | None = None,
    max_num_tokens: int | None = None,
    add_coolname: bool = True,
    num_sae_latents: int | None = None,
    sae_model: Literal["base", "chat"] | None = None,
) -> dict:
    """
    Main function to evaluate interventions effects.
    """
    if dictionary is None:
        name = (name or "") + "-baselines"
    if is_sae or is_difference_sae:
        if num_sae_latents is None:
            raise ValueError(
                "num_sae_latents must be provided if {} is True".format(
                    "is_sae" if is_sae else "is_difference_sae"
                )
            )
        if sae_model is None:
            raise ValueError(
                "sae_model must be provided if {} is True".format(
                    "is_sae" if is_sae else "is_difference_sae"
                )
            )

    dataset = load_dataset(dataset_name, split=split)[dataset_col]
    if test:
        name = name or "test"
        print(f"Testing using {name}")
        dataset = dataset[:120]  # Limit dataset size for testing
    else:
        dataset = dataset[: min(30_000, len(dataset))]
    if name is None:
        name = dictionary_name.split("/")[-1] if dictionary_name else "unnamed"
        if model_name:
            name += "-" + model_name.split("/")[-1]
        name += f"-{split}"
        if dataset_col:
            name += f"-{dataset_col}"

    # Generate unique run name
    run_name = (
        name
        + (("_" + generate_slug(2)) if add_coolname else "")
        + ("_" + str(int(time.time())))
    )

    # Setup save directory
    run_save_path = save_path / run_name
    run_save_path.mkdir(parents=True, exist_ok=True)

    # Process chat_only_indices if provided
    if chat_only_indices is not None and latent_df is not None:
        latent_df = latent_df.iloc[chat_only_indices]
        latent_df["tag"] = "Chat only"
    if latent_df is not None:
        print(f"len df: {len(latent_df)}")

    # Generate function dictionary based on model type
    seeds = list(range(num_seeds))
    if dictionary is None:
        fn_dict = baselines_half_fns()
        infos = {}
    elif not isinstance(dictionary, CrossCoder):
        fn_dict, infos = sae_steering_half_fns(
            dictionary,
            seeds,
            latent_df,
            is_difference_sae=is_difference_sae,
            num_latents=num_sae_latents,
            sae_model=sae_model,
        )
    else:
        fn_dict, infos = arxiv_paper_half_fns(
            dictionary,
            latent_df,
            add_base_only_latents=add_base_only_latents,
        )

    # Save metadata
    metadata = {
        "infos": infos,
        "parameters": {
            "layer_to_stop": layer_to_stop,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "device": str(device),
            "num_seeds": num_seeds,
            "is_sae": is_sae,
            "test": test,
            "name": name,
            "dictionary_name": dictionary_name,
            "model_name": model_name,
            "max_num_tokens": max_num_tokens,
            "run_name": run_name,
            "token_level_replacement": token_level_replacement,
            "add_base_only_latents": add_base_only_latents,
            "is_difference_sae": is_difference_sae,
            "sae_model": sae_model,
            "num_sae_latents": num_sae_latents,
            "ignore_first_n_tokens": ignore_first_n_tokens,
            "dataset_name": dataset_name,
            "split": split,
            "dataset_col": dataset_col,
        },
    }
    with open(run_save_path / "metadata.json", "w") as f:
        json.dump(
            metadata,
            f,
        )

    project = "perplexity-comparison" + ("-test" if test else "")
    wandb.init(project=project, name=run_name, config=metadata["parameters"])

    # Run evaluation
    result, total_num_tokens = evaluate_interventions(
        base_model,
        chat_model,
        tokenizer_name,
        dataset,
        fn_dict,
        layer_to_stop=layer_to_stop,
        batch_size=batch_size,
        device=device,
        max_seq_len=max_seq_len,
        log_every=log_every,
        save_path=run_save_path,
        checkpoint_every=checkpoint_every,
        token_level_replacement=token_level_replacement,
        ignore_first_n_tokens=ignore_first_n_tokens,
        max_num_tokens=max_num_tokens,
        run_name=run_name,
    )
    metadata["total_num_tokens"] = total_num_tokens
    with open(run_save_path / "metadata.json", "w") as f:
        json.dump(
            metadata,
            f,
        )

    # Add mean across random seeds and save results
    result = add_random_means(result)
    with open(save_path / f"{run_name}_result.json", "w") as f:
        json.dump(result, f)
    wandb.finish()

    return result


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
    parser.add_argument(
        "--dictionary",
        type=str,
        default=None,
        help="Dictionary to use for interventions. If None, will only run baselines.",
    )
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
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--skip-token-level-replacement", action="store_true")
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
    device = (
        args.device
        if args.device != "auto"
        else "cuda" if th.cuda.is_available() else "cpu"
    )
    seeds = list(range(args.num_seeds))
    if args.dictionary is not None:
        dictionary = load_dictionary_model(args.dictionary, is_sae=args.is_sae).to(
            device
        )
    else:
        dictionary = None
    if dictionary is not None:
        chat_only_indices = None
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
    else:
        df = None
        chat_only_indices = None
    if (args.base_model in MODEL_CONFIGS) != (args.chat_model in MODEL_CONFIGS):
        not_in_model_configs = (
            args.base_model if args.chat_model in MODEL_CONFIGS else args.chat_model
        )
        raise ValueError(
            f"Weird, one of the models is in MODEL_CONFIGS and the other is not. Ensure that both models are in MODEL_CONFIGS. {not_in_model_configs} is not in MODEL_CONFIGS."
        )
    if args.base_model in MODEL_CONFIGS:
        ignore_first_n_tokens = MODEL_CONFIGS[args.base_model][
            "ignore_first_n_tokens_per_sample"
        ]
        if (
            ignore_first_n_tokens
            != MODEL_CONFIGS[args.chat_model]["ignore_first_n_tokens_per_sample"]
        ):
            raise ValueError(
                f"Weird, ignore_first_n_tokens_per_sample for {args.base_model} and {args.chat_model} are different. If it's expected, you need to adapt the code to handle this."
            )
    else:
        ignore_first_n_tokens = None
    if args.base_model in MODEL_CONFIGS and not args.skip_token_level_replacement:
        token_level_replacement = MODEL_CONFIGS[args.base_model][
            "token_level_replacement"
        ]
        logger.info(f"Using token level replacement: {token_level_replacement}")
    else:
        if args.base_model in MODEL_CONFIGS:
            logger.info(
                f"Skipping token level replacement for {args.base_model} as --skip-token-level-replacement flag is set"
            )
        else:
            logger.info(
                f"Skipping token level replacement & ignore_first_n_tokens for {args.base_model} as it is not in MODEL_CONFIGS"
            )
        token_level_replacement = None
    result = kl_experiment(
        dictionary=dictionary,
        base_model=load_hf_model(args.base_model),
        chat_model=load_hf_model(args.chat_model),
        tokenizer_name=args.chat_model,
        dataset_name=args.dataset,
        split=args.split,
        dataset_col=args.dataset_col,
        latent_df=df,
        chat_only_indices=chat_only_indices,
        add_base_only_latents=args.add_base_only_latents,
        is_sae=args.is_sae,
        layer_to_stop=args.layer_to_stop,
        batch_size=args.batch_size,
        device=device,
        max_seq_len=args.max_seq_len,
        log_every=args.log_every,
        k_first=args.k_first,
        checkpoint_every=args.checkpoint,
        num_seeds=args.num_seeds,
        save_path=args.save_path,
        name=args.name,
        dictionary_name=args.dictionary,
        model_name=args.chat_model,
        test=args.test,
        max_num_tokens=args.max_num_tokens,
        token_level_replacement=token_level_replacement,
        ignore_first_n_tokens=ignore_first_n_tokens,
    )
