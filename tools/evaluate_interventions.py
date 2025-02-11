import sys
from pathlib import Path
from collections import defaultdict
import json


import torch as th
from tqdm.auto import tqdm
from torch.nn.functional import kl_div
from torchmetrics.aggregation import MeanMetric
import wandb


sys.path.append(str(Path(__file__).parent.parent))
from tools.split_gemma import split_gemma
from tools.setup_to_eval import *  # noqa
from tools.tokenization_utils import tokenize_with_ctrl_mask
from tools.utils import mask_k_first_ones_vec


def MeanMetricToDevice(device):
    return lambda: MeanMetric().to(device)


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
    """
    Evaluate different intervention strategies on language models.

    Args:
        base_model: Base language model to evaluate
        instruct_model: Instruction-tuned model to evaluate
        dataset: Dataset of conversations to evaluate on
        preprocess_before_last_half_fns: Dict mapping strategy names to preprocessing functions
        layer_to_stop: Layer index to split model at (default: 13)
        batch_size: Batch size for evaluation (default: 8)
        device: Device to run on (default: "cuda")
        max_seq_len: Maximum sequence length (default: 1024)
        log_every: How often to log metrics (default: 100)
        checkpoint_every: How often to save checkpoints (default: 2000)
        k_first: Number of first tokens to consider (default: 10)
        save_path: Path to save results (default: None)

    Returns:
        Dict containing evaluation metrics:
        - loss: Negative log likelihood
        - perplexity: Model perplexity
        - kl-instruct: KL divergence vs instruction model
        - kl-base: KL divergence vs base model
        - loss_wrt_instruct_pred: Loss relative to instruction model predictions
        - perplexity_wrt_instruct_pred: Perplexity relative to instruction model predictions
    """
    nlls = defaultdict(MeanMetricToDevice(device))
    perplexity = defaultdict(MeanMetricToDevice(device))
    instruct_kl = defaultdict(MeanMetricToDevice(device))
    base_kl = defaultdict(MeanMetricToDevice(device))
    nlls_wrt_instruct_pred = defaultdict(MeanMetricToDevice(device))
    perplexity_wrt_instruct_pred = defaultdict(MeanMetricToDevice(device))
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

        for (
            fn_name,
            preprocess_before_last_half_fn,
        ) in preprocess_before_last_half_fns.items():
            base_activations_edited, instruct_activations_edited, mask = (
                preprocess_before_last_half_fn(
                    base_activations,
                    instruct_activations,
                    ctrl_mask=ctrl_mask,
                    assistant_mask=assistant_mask,
                    k_first_ass_toks_mask=k_first_ass_toks_mask,
                    next_ass_toks_mask=next_ass_toks_mask,
                )
            )
            if mask is not None:
                base_logits = base_logits_raw[mask]
                instruct_logits = instruct_logits_raw[mask]
                base_causal_mask = base_causal_mask_raw[mask]
                instruct_causal_mask = instruct_causal_mask_raw[mask]
                assistant_mask = assistant_mask[mask]
                labels = input_ids[mask][assistant_mask]
            else:
                base_logits = base_logits_raw
                instruct_logits = instruct_logits_raw
                base_causal_mask = base_causal_mask_raw
                instruct_causal_mask = instruct_causal_mask_raw
                assistant_mask = assistant_mask
                labels = input_ids[assistant_mask]
            final_out = None
            if base_activations_edited is not None:
                final_out = base_model.second_half_forward(
                    base_activations_edited,
                    base_causal_mask,
                    base_position_ids,
                    layer_idx=layer_to_stop,
                    return_dict=True,
                )
                n_pred = base_activations_edited.shape[0]
            elif instruct_activations_edited is not None:
                final_out = instruct_model.second_half_forward(
                    instruct_activations_edited,
                    instruct_causal_mask,
                    instruct_position_ids,
                    layer_idx=layer_to_stop,
                    return_dict=True,
                )
                n_pred = instruct_activations_edited.shape[0]
            if final_out is not None:
                base_log_probs = th.log_softmax(
                    base_logits[assistant_mask].float(), dim=-1
                )
                instruct_preds = th.argmax(instruct_logits, dim=-1)[assistant_mask]
                instruct_log_probs = th.log_softmax(
                    instruct_logits[assistant_mask].float(), dim=-1
                )
                logits = final_out.logits[assistant_mask].float()
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
                loss = instruct_model.compute_loss(logits, labels)
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
                assert log_probs.dim() == 2
                wandb.log({f"kl-instruct/{fn_name}": (it_kl.mean()).item()}, step=i)
                wandb.log({f"kl-base/{fn_name}": (b_kl.mean()).item()}, step=i)
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
