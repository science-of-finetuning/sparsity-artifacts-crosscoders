import sys
from pathlib import Path
from collections import defaultdict
import json


import torch as th
from tqdm.auto import tqdm
from torch.nn.functional import kl_div
import wandb


sys.path.append(str(Path(__file__).parent.parent))
from tools.split_gemma import split_gemma
from tools.setup_to_eval import HalfStepPreprocessFn
from tools.tokenization_utils import tokenize_with_ctrl_mask
from tools.utils import mask_k_first_ones_vec
from tools.compute_utils import RunningMeanStd


def compute_metrics_for_subset(
    logits, labels, base_logits, instruct_logits, subset_mask, instruct_model
):
    """Compute all metrics for a subset of tokens."""
    base_log_probs = th.log_softmax(base_logits[subset_mask].float(), dim=-1)
    instruct_preds = th.argmax(instruct_logits, dim=-1)[subset_mask]
    instruct_log_probs = th.log_softmax(instruct_logits[subset_mask].float(), dim=-1)
    logits = logits[:, :-1][subset_mask[:, 1:]].float()
    labels = labels[:, 1:][subset_mask[:, 1:]]
    log_probs = th.log_softmax(logits, dim=-1)

    loss = instruct_model.compute_loss(logits, labels, already_shifted=True)
    loss_wrt_instruct = instruct_model.compute_loss(
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
    instruct_model,
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
    instruct_model = split_gemma(instruct_model)

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
            tokenizer=instruct_model.tokenizer,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = batch_tokens["input_ids"].to(device)
        attn_mask = batch_tokens["attention_mask"].to(device)
        assistant_mask = th.tensor(batch_tokens["assistant_masks"]).bool().to(device)
        ctrl_mask = batch_tokens["ctrl_mask"].to(device)
        k_first_ass_toks_mask = mask_k_first_ones_vec(assistant_mask, k_first).to(
            device
        )
        next_ass_toks_mask = assistant_mask & ~k_first_ass_toks_mask

        base_activations, base_causal_mask_raw, base_position_ids = (
            base_model.first_half_forward(
                input_ids=input_ids,
                attention_mask=attn_mask,
                layer_idx=layer_to_stop,
            )
        )
        instruct_activations, instruct_causal_mask_raw, instruct_position_ids = (
            instruct_model.first_half_forward(
                input_ids=input_ids,
                attention_mask=attn_mask,
                layer_idx=layer_to_stop,
            )
        )

        base_logits_raw = base_model.second_half_forward(
            base_activations,
            base_causal_mask_raw,
            base_position_ids,
            layer_idx=layer_to_stop,
            return_dict=True,
        ).logits
        instruct_logits_raw = instruct_model.second_half_forward(
            instruct_activations,
            instruct_causal_mask_raw,
            instruct_position_ids,
            layer_idx=layer_to_stop,
            return_dict=True,
        ).logits

        for fn_name, preprocess_fn in preprocess_before_last_half_fns.items():
            base_activations_edited, instruct_activations_edited, mask = preprocess_fn(
                base_activations,
                instruct_activations,
                ctrl_mask=ctrl_mask,
                assistant_mask=assistant_mask,
                k_first_ass_toks_mask=k_first_ass_toks_mask,
                next_ass_toks_mask=next_ass_toks_mask,
            )

            if mask is not None:
                base_logits = base_logits_raw[mask]
                instruct_logits = instruct_logits_raw[mask]
                base_causal_mask = base_causal_mask_raw[mask]
                instruct_causal_mask = instruct_causal_mask_raw[mask]
                effective_assistant_mask = assistant_mask[mask]
                effective_k_first = k_first_ass_toks_mask[mask]
                effective_post_k_first = next_ass_toks_mask[mask]
                labels = input_ids[mask]
            else:
                base_logits = base_logits_raw
                instruct_logits = instruct_logits_raw
                base_causal_mask = base_causal_mask_raw
                instruct_causal_mask = instruct_causal_mask_raw
                effective_assistant_mask = assistant_mask
                effective_k_first = k_first_ass_toks_mask
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
                )
            elif instruct_activations_edited is not None:
                final_out = instruct_model.second_half_forward(
                    instruct_activations_edited,
                    instruct_causal_mask,
                    instruct_position_ids,
                    layer_idx=layer_to_stop,
                    return_dict=True,
                )

            if final_out is not None:
                # Compute metrics for all tokens
                for metrics_dict, prefix, mask in [
                    (metrics_all, "", effective_assistant_mask),
                    (metrics_kfirst, "k_first_", effective_k_first),
                    (metrics_post_k_first, "post_k_first_", effective_post_k_first),
                ]:
                    if prefix == "" or mask.any():
                        metrics = compute_metrics_for_subset(
                            final_out.logits,
                            labels,
                            base_logits,
                            instruct_logits,
                            mask,
                            instruct_model,
                        )
                        update_metrics(metrics_dict, metrics, fn_name, prefix)
                        log_metrics(metrics, fn_name, i, prefix)

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
                                metrics_dict["nlls"][fn_name].compute()[0].item()
                            ),
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
                                metrics_dict["nlls_wrt_instruct"][fn_name]
                                .compute()[0]
                                .item()
                            ),
                            f"{prefix}num_samples/{fn_name}": metrics_dict[
                                "num_samples"
                            ][fn_name],
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
