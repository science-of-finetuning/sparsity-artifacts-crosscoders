import sys

sys.path.append(".")
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.cache import ActivationCache
from dictionary_learning.dictionary import CrossCoder
from datasets import load_from_disk, load_dataset
from loguru import logger
import torch as th
from pathlib import Path
import os
import json
from tools.deprecated.latent_analysis import latent_statistics
from tools.utils import load_connor_crosscoder

from tools.configs import HF_NAME

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crosscoder-path",
        type=str,
        default="Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
    )
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--instruct-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset-size", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--connor-crosscoder", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default="stats",
    )
    parser.add_argument(
        "--base-dataset",
        type=str,
        default=f"{HF_NAME}/fineweb-100m-sample-test-set",
        help="Dataset to compute statistics for base model",
    )
    parser.add_argument(
        "--chat-dataset",
        type=str,
        default=f"{HF_NAME}/lmsys-chat-1m-gemma-2-it-formatted",
        help="Dataset to compute statistics for instruction model",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="validation",
        help="Split of the dataset to use",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = th.device(args.device)

    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=th.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    instruction_model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model,
        torch_dtype=th.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model)

    # Load crosscoder
    if args.connor_crosscoder:
        args.crosscoder_path = "ckkissane_crosscoder-gemma-2-2b-model-diff"
        cc = load_connor_crosscoder()
        text_column_chat = "text_base_format"
    else:
        cc = CrossCoder.from_pretrained(args.crosscoder_path, from_hub=True)
        text_column_chat = "text"
    cc = cc.to(device)

    # Load datasets
    logger.info(f"Loading base dataset from {args.base_dataset}")
    validation_set_base = load_dataset(args.base_dataset, split=args.dataset_split)
    validation_set_base = validation_set_base.select(
        range(min(args.subset_size, len(validation_set_base)))
    )

    logger.info(f"Loading chat dataset from {args.chat_dataset}")
    validation_set_chat = load_dataset(args.chat_dataset, split=args.dataset_split)
    validation_set_chat = validation_set_chat.select(
        range(min(args.subset_size, len(validation_set_chat)))
    )

    logger.info("Computing statistics for base dataset...")
    stats_fineweb = latent_statistics(
        validation_set_base.select_columns("text"),
        tokenizer,
        base_model,
        instruction_model,
        cc,
        args.layer,
        batch_size=args.batch_size,
    )

    logger.info("Computing statistics for chat dataset...")
    stats_lmsys = latent_statistics(
        validation_set_chat.select_columns(text_column_chat),
        tokenizer,
        base_model,
        instruction_model,
        cc,
        args.layer,
        batch_size=args.batch_size,
        text_column=text_column_chat,
    )

    # Save results
    results_dir = args.results_dir / args.crosscoder_path.replace("/", "_")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(results_dir / "args.json", "w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f
        )

    th.save(stats_fineweb, results_dir / "fineweb.pt")
    th.save(stats_lmsys, results_dir / "lmsys.pt")


if __name__ == "__main__":
    main()
