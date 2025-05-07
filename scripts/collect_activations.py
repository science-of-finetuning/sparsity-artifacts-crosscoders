import sys

sys.path.append(".")
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

os.environ["WANDB__SERVICE_WAIT"] = "300"

from tools.configs import MODEL_CONFIGS

th.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="jkminder")
    parser.add_argument("--wandb-project", default="activation_collection")
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
        default=None,
        help="Overwrite the text column in the dataset to collect activations from",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing activations"
    )
    parser.add_argument(
        "--store-tokens",
        action="store_true",
        help="Store tokens in the activation cache",
    )
    parser.add_argument(
        "--disable-multiprocessing", action="store_true", help="Disable multiprocessing"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Data type to use for activations"
    )
    args = parser.parse_args()

    if args.dtype == "bfloat16":
        dtype = th.bfloat16
    elif args.dtype == "float16":
        dtype = th.float16
    elif args.dtype == "float32":
        dtype = th.float32
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")

    if len(args.layers) == 0:
        raise ValueError("Must provide at least one layer")

    if args.wandb:
        import wandb

        wandb.init(
            name=args.model.split("/")[-1]
            + "_"
            + args.dataset.split("/")[-1]
            + "_"
            + args.dataset_split,
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=args,
        )

    CFG = MODEL_CONFIGS[args.model]
    print("MODEL_CONFIGS=", CFG)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=MODEL_CONFIGS[args.model]["attn_implementation"],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    nnmodel = LanguageModel(model, tokenizer=tokenizer)
    print("dtype=", nnmodel.dtype)
    num_layers = int(len(nnmodel.model.layers))
    layers = args.layers
    logger.info(f"Collecting activations from layers: {layers}")

    submodules = [nnmodel.model.layers[layer] for layer in layers]
    submodule_names = ["layer_{}".format(layer) for layer in layers]

    d_model = nnmodel._model.config.hidden_size
    logger.info(f"d_model={d_model}")

    store_dir = Path(args.activation_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset.split("/")[-1]
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    text_column = MODEL_CONFIGS[args.model]["text_column"]

    if args.text_column is not None:
        text_column = args.text_column

    if text_column != "text":
        args.dataset_split = f"{args.dataset_split}-col{text_column}"

    print("Text column=", text_column)
    out_dir = store_dir / args.model.split("/")[-1] / dataset_name / args.dataset_split
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Collecting activations to {out_dir}")
    time.sleep(10)
    ActivationCache.collect(
        dataset[text_column],
        submodules,
        submodule_names,
        nnmodel,
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
        multiprocessing=not args.disable_multiprocessing,
        ignore_first_n_tokens_per_sample=CFG["ignore_first_n_tokens_per_sample"],
        overwrite=args.overwrite,
        token_level_replacement=CFG["token_level_replacement"],
    )
