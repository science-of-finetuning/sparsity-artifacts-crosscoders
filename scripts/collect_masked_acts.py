import sys
import logging
from tqdm import tqdm, trange
from argparse import ArgumentParser
import torch as th
from collections import defaultdict
import psutil

sys.path.append(".")
from tools.utils import (
    tokenize_with_ctrl_mask,
    mask_k_first_ones_vec,
    load_lmsys_formatted,
    save_json,
    patch_tokenizer,
)
from nnterp import load_model
from nnterp.nnsight_utils import get_layer_output, get_layer

# Add near the top of the file, after imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@th.no_grad
def compute_acts_on_masks(
    dataset,
    base_model,
    chat_model,
    layer,
    *,
    batch_size,
    max_seq_len=1024,
    k_first=10,
    num_acts=1_000_000,
    max_num_acts_times=10,
):
    max_num_acts = num_acts * max_num_acts_times
    base_acts_dict = {
        "ctrl": [],
        "assistant_pred": [],
        "all": [],
        "k_first_pred": [],
        "human": [],
    }
    chat_acts_dict = {
        "ctrl": [],
        "assistant_pred": [],
        "all": [],
        "k_first_pred": [],
        "human": [],
    }
    mask_names = ["all", "human", "k_first_pred", "assistant_pred", "ctrl"]
    num_acts_dict = defaultdict(int)
    assert set(base_acts_dict.keys()) == set(chat_acts_dict.keys()) == set(mask_names)
    for i in trange(0, len(dataset), batch_size):
        print(f"Processing batch {i} of {len(dataset)}")
        batch = dataset[i : min(i + batch_size, len(dataset))]
        batch_tokens = tokenize_with_ctrl_mask(
            batch,
            tokenizer=chat_model.tokenizer,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        attn_mask = batch_tokens["attention_mask"].bool()
        batch_size, seq_len = attn_mask.shape
        ctrl_mask = batch_tokens["ctrl_mask"]
        assistant_mask = batch_tokens["assistant_masks"]
        human_mask = attn_mask & ~ctrl_mask & ~assistant_mask
        human_mask[:, 0] = (
            chat_model.tokenizer.bos_token is not None
        )  # remove the bos token if it exists
        # shift the assistant mask to the left by 1 to have the token at which you make the prediction rather than the token you need to predict
        assistant_pred_mask = th.zeros_like(assistant_mask)
        assistant_pred_mask[:, :-1] = assistant_mask[:, 1:]
        k_first_pred_toks_mask = (
            mask_k_first_ones_vec(assistant_pred_mask, k_first) & ~ctrl_mask
        )
        assert (
            attn_mask.dtype
            == ctrl_mask.dtype
            == assistant_pred_mask.dtype
            == k_first_pred_toks_mask.dtype
            == th.bool
        )
        if (
            num_acts_dict["human"] >= num_acts
            and num_acts_dict["assistant_pred"] >= num_acts
        ):
            logger.info("Reached num_acts for easy masks")
            if (
                num_acts_dict["first_k_pred"] >= num_acts
                and num_acts_dict["ctrl"] >= num_acts
            ):
                logger.info("Reached num_acts for all masks")
                break
            # remove useless tokens to focus on ctrl tokens and k_first_pred tokens
            arange = th.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            assert (
                arange.shape == attn_mask.shape
            ), f"{arange.shape} != {attn_mask.shape}"
            assert (
                arange[0] == th.arange(seq_len)
            ).all(), f"{arange[0]} != {th.arange(seq_len)}"
            if ctrl_mask.any():
                max_idx_ctrl = arange[ctrl_mask].max()
            else:
                max_idx_ctrl = 0
            if k_first_pred_toks_mask.any():
                max_idx_k_first = arange[k_first_pred_toks_mask].max()
            else:
                max_idx_k_first = 0
            max_idx = max(max_idx_ctrl, max_idx_k_first)
            if max_idx == 0:
                print("No ctrl or k_first_pred tokens found in batch")
                continue
            attn_mask = attn_mask[:, :max_idx]
            ctrl_mask = ctrl_mask[:, :max_idx]
            assistant_pred_mask = assistant_pred_mask[:, :max_idx]
            k_first_pred_toks_mask = k_first_pred_toks_mask[:, :max_idx]
            human_mask = human_mask[:, :max_idx]
            batch_tokens["attention_mask"] = batch_tokens["attention_mask"][:, :max_idx]
            batch_tokens["input_ids"] = batch_tokens["input_ids"][:, :max_idx]
        with base_model.trace(batch_tokens):
            base_acts = (
                get_layer_output(base_model, layer).to("cpu", non_blocking=True).save()
            )
            get_layer(base_model, layer).output.stop()
        with chat_model.trace(batch_tokens):
            chat_acts = (
                get_layer_output(chat_model, layer).to("cpu", non_blocking=True).save()
            )
            get_layer(chat_model, layer).output.stop()
        try:
            for act_dict, acts in zip(
                [base_acts_dict, chat_acts_dict], [base_acts, chat_acts]
            ):
                for mask_name, mask in zip(
                    mask_names,
                    [
                        attn_mask,
                        human_mask,
                        k_first_pred_toks_mask,
                        assistant_pred_mask,
                        ctrl_mask,
                    ],
                ):
                    if num_acts_dict[mask_name] >= max_num_acts:
                        logger.info(f"Reached max_num_acts for mask {mask_name}")
                        continue
                    mask_acts = acts[mask]
                    num_acts_dict[mask_name] += mask_acts.shape[0]
                    act_dict[mask_name].append(mask_acts)
                    # raise if ram is > 100GB
                    if (
                        psutil.virtual_memory().used > 100 * 1024 * 1024 * 1024
                    ):  # 100GB in bytes
                        raise MemoryError("Total system RAM usage exceeded 100GB")
        except MemoryError:
            logger.error("MemoryError: Total system RAM usage exceeded 100GB")
            break

    return base_acts_dict, chat_acts_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k-first", type=int, default=10)
    parser.add_argument("--acts-save-path", type=str, default="results/")
    parser.add_argument("--num-acts", "-n", type=int, default=1_000_000)
    args = parser.parse_args()
    dataset = load_lmsys_formatted(args.split)["conversation"]
    base_model = load_model(args.base_model, device_map=args.device)
    chat_model = load_model(args.chat_model, device_map=args.device)
    patch_tokenizer(chat_model.tokenizer, args.chat_model)
    base_acts_dict, chat_acts_dict = compute_acts_on_masks(
        dataset,
        base_model,
        chat_model,
        layer=args.layer,
        batch_size=args.batch_size,
        max_seq_len=1024,
        k_first=args.k_first,
        num_acts=args.num_acts,
    )
    act_save_path = args.acts_save_path / "activations/per_mask_acts"
    base_save_path = (
        base_acts_dict,
        act_save_path
        / args.base_model.split("/")[-1]
        / "lmsys"
        / f"layer_{args.layer}_acts.pt",
    )
    base_save_path.parent.mkdir(parents=True, exist_ok=True)
    th.save(
        base_acts_dict,
        base_save_path,
    )
    chat_save_path = (
        chat_acts_dict,
        act_save_path
        / args.chat_model.split("/")[-1]
        / "lmsys"
        / f"layer_{args.layer}_acts.pt",
    )
    chat_save_path.parent.mkdir(parents=True, exist_ok=True)
    th.save(
        chat_acts_dict,
        chat_save_path,
    )
