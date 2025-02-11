import sys
from pathlib import Path
from collections import defaultdict
import json

sys.path.append(str(Path(__file__).parent.parent))

import torch as th
from tqdm.auto import tqdm
import wandb
from torch.nn.functional import kl_div
from torchmetrics.aggregation import MeanMetric
from dictionary_learning.dictionary import CrossCoder
from tools.split_gemma import split_gemma
from tools.setup_to_eval import *

ROOT_PATH = Path(__file__).parent
CHAT_TEMPLATE = open(ROOT_PATH / "templates/gemma_chat_template.jinja").read()


def MeanMetricToDevice(device):
    return lambda: MeanMetric().to(device)


@th.inference_mode()
def evaluate_interpretation(
    base_model,
    instruct_model,
    crosscoder: CrossCoder,
    dataset,
    feature_index,
    get_predicted_mask,
    batch_size=8,
    device="cuda",
    # max_seq_len=1024,
    layer_to_stop=13,
):
    """
    Evaluate the interpretation of a feature. Return the activation of a feature over the predicted to be active tokens
    and the activation of the feature over the other tokens.

    Args:
        base_model: The base model to evaluate.
        instruct_model: The instruct model to evaluate.
        dataset: The dataset to evaluate on.
        feature_index: The index of the feature to evaluate.
        get_predicted_mask: A function that takes an input_ids tensor and returns a predicted mask.
        The mask should be a boolean tensor of shape (batch_size, seq_len) where True indicates the token is predicted to be active.
        batch_size: The batch size to use for evaluation.
        device: The device to use for evaluation.
        layer_to_stop: The layer to take the activations from.
    """
    base_model = split_gemma(base_model)
    instruct_model = split_gemma(instruct_model)
    true_activations = []
    false_activations = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        batch_tokens = instruct_model.tokenizer(
            batch,
            # return_assistant_tokens_mask=True,
            # chat_template=CHAT_TEMPLATE,
            # return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        input_ids = batch_tokens["input_ids"].to(device)
        attention_mask = batch_tokens["attention_mask"].to(device)
        # assistant_mask = th.tensor(batch_tokens["assistant_masks"]).bool().to(device)
        predicted_mask = get_predicted_mask(input_ids)
        base_activations, *_ = base_model.first_half_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_idx=layer_to_stop,
        )
        instruct_activations, *_ = instruct_model.first_half_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_idx=layer_to_stop,
        )
        cc_input = th.stack([base_activations, instruct_activations], dim=2).float()
        cc_input = einops.rearrange(cc_input, "b s l d -> (b s) l d")
        cc_activations = crosscoder.encoder(cc_input, select_features=[feature_index])
        cc_activations = einops.rearrange(
            cc_activations, "(b s) f -> b s f", b=len(batch)
        ).squeeze(2)
        attn_mask = attention_mask.bool()
        pred_act = cc_activations[predicted_mask & attn_mask]
        false_act = cc_activations[(~predicted_mask) & attn_mask]
        true_activations.append(pred_act.cpu())
        false_activations.append(false_act.cpu())
    return th.cat(true_activations), th.cat(false_activations)


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
        base_activations, base_causal_mask_raw, base_position_ids = (
            base_model.first_half_forward(
                input_ids=batch_tokens["input_ids"],
                attention_mask=batch_tokens["attention_mask"],
                layer_idx=layer_to_stop,
            )
        )
        instruct_activations, instruct_causal_mask_raw, instruct_position_ids = (
            instruct_model.first_half_forward(
                input_ids=batch_tokens["input_ids"],
                attention_mask=batch_tokens["attention_mask"],
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
                preprocess_before_last_half_fn(base_activations, instruct_activations)
            )
            if mask is not None:
                base_logits = base_logits_raw[mask]
                instruct_logits = instruct_logits_raw[mask]
                base_causal_mask = base_causal_mask_raw[mask]
                instruct_causal_mask = instruct_causal_mask_raw[mask]
                assistant_mask = batch_tokens["assistant_masks"][mask]
                labels = batch_tokens["input_ids"][mask][assistant_mask]
            else:
                base_logits = base_logits_raw
                instruct_logits = instruct_logits_raw
                base_causal_mask = base_causal_mask_raw
                instruct_causal_mask = instruct_causal_mask_raw
                assistant_mask = batch_tokens["assistant_masks"]
                labels = batch_tokens["input_ids"][assistant_mask]
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
