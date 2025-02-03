import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.cache import ActivationCache
from datasets import load_from_disk, load_dataset
from loguru import logger
import torch as th
from nnsight import LanguageModel
from pathlib import Path
import os
import time

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activation-store-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Layer indices to collect activations from",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to collect activations from. Examples are 'science-of-finetuning/lmsys-chat-1m-gemma-2-it-formatted' and 'science-of-finetuning/fineweb-100m-sample-test-set'",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Split of the dataset to collect activations from. Examples are 'train' and 'test'",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10**6,
        help="Maximum number of samples to collect activations from",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10**8,
        help="Maximum number of tokens to collect activations from",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column in the dataset to collect activations from",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing activations"
    )
    parser.add_argument(
        "--store-tokens", action="store_true", help="Store tokens in the activation cache"
    )
    args = parser.parse_args()

    if len(args.layers) == 0:
        raise ValueError("Must provide at least one layer")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = LanguageModel(model, tokenizer=tokenizer)
    num_layers = int(len(model.model.layers))
    layers = args.layers
    logger.info(f"Collecting activations from layers: {layers}")

    submodules = [model.model.layers[layer] for layer in layers]
    submodule_names = ["layer_{}".format(layer) for layer in layers]

    d_model = model.config.hidden_size
    logger.info(f"d_model={d_model}")

    store_dir = Path(args.activation_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset.split("/")[-1]
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if args.text_column != "text":
        args.dataset_split = f"{args.dataset_split}-col{args.text_column}"
    out_dir = store_dir / args.model.split("/")[-1] / dataset_name / args.dataset_split
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Collecting activations to {out_dir}")
    time.sleep(10)
    ActivationCache.collect(
        dataset[args.text_column],
        submodules,
        submodule_names,
        model,
        out_dir,
        shuffle_shards=False,
        io="out",
        shard_size=10**6,
        batch_size=args.batch_size,
        context_len=1024,
        d_model=d_model,
        last_submodule=submodules[-1],
        max_total_tokens=args.max_tokens,
        store_tokens=args.store_tokens,
    )
