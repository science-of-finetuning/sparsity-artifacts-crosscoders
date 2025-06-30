import sys

sys.path.append(".")
import torch as th
import argparse
from tqdm import trange, tqdm
from pathlib import Path
from nnsight import LanguageModel
from dictionary_learning.cache import PairedActivationCache
from dictionary_learning import ActivationBuffer, CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.dictionary import CodeNormalization

from dictionary_learning.trainers import BatchTopKTrainer, BatchTopKSAE
import os

from tools.cache_utils import DifferenceCache

import wandb

wandb.require("legacy-service")

th.set_float32_matmul_precision("high")

from tools.utils import load_activation_dataset


def get_local_shuffled_indices(
    num_samples_per_dataset, shard_size, single_dataset=False
):
    num_shards_per_dataset = num_samples_per_dataset // shard_size + (
        1 if num_samples_per_dataset % shard_size != 0 else 0
    )
    print(f"Number of shards per dataset: {num_shards_per_dataset}", flush=True)

    shuffled_indices = []
    for i in trange(num_shards_per_dataset):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, num_samples_per_dataset)
        shard_size_curr = end_idx - start_idx

        if single_dataset:
            shard_indices = th.randperm(shard_size_curr) + start_idx
        else:
            fineweb_indices = th.randperm(shard_size_curr) + start_idx
            lmsys_indices = (
                th.randperm(shard_size_curr) + num_samples_per_dataset + start_idx
            )

            shard_indices = th.zeros(2 * shard_size_curr, dtype=th.long)
            shard_indices[0::2] = fineweb_indices
            shard_indices[1::2] = lmsys_indices
        shuffled_indices.append(shard_indices)

    shuffled_indices = th.cat(shuffled_indices)
    return shuffled_indices


def setup_cache(args, fineweb_cache, lmsys_cache):
    if args.target == "base":
        fineweb_cache = fineweb_cache.activation_cache_1
        lmsys_cache = lmsys_cache.activation_cache_1
    elif args.target == "chat":
        fineweb_cache = fineweb_cache.activation_cache_2
        lmsys_cache = lmsys_cache.activation_cache_2
    elif args.target == "difference_bc":
        fineweb_cache = DifferenceCache(
            fineweb_cache.activation_cache_1, fineweb_cache.activation_cache_2
        )
        lmsys_cache = DifferenceCache(
            lmsys_cache.activation_cache_1, lmsys_cache.activation_cache_2
        )
    elif args.target == "difference_cb":
        fineweb_cache = DifferenceCache(
            fineweb_cache.activation_cache_2, fineweb_cache.activation_cache_1
        )
        lmsys_cache = DifferenceCache(
            lmsys_cache.activation_cache_2, lmsys_cache.activation_cache_1
        )
    return fineweb_cache, lmsys_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation-store-dir",
        type=str,
        default="activations",
        help="Directory containing stored activations",
    )
    parser.add_argument(
        "--base-model", type=str, default="google/gemma-2-2b", help="Base model hf name"
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="google/gemma-2-2b-it",
        help="Chat model hf name",
    )
    parser.add_argument("--layer", type=int, default=13, help="Layer to train SAE on")
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="jkminder",
        help="Weights & Biases entity name",
    )
    parser.add_argument(
        "--disable-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--expansion-factor", type=int, default=32, help="SAE expansion factor"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Training batch size"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of data loader workers"
    )
    parser.add_argument("--k", type=int, default=50, help="Top-k sparsity parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--validate-every-n-steps",
        type=int,
        default=10000,
        help="Validation frequency in steps",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Custom run name for logging"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100_000_000,
        help="Total number of training samples",
    )
    parser.add_argument(
        "--num-validation-samples",
        type=int,
        default=2_000_000,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column name for lmsys dataset",
    )
    parser.add_argument(
        "--no-train-shuffle",
        action="store_true",
        help="Disable training data shuffling",
    )
    parser.add_argument(
        "--local-shuffling",
        action="store_true",
        help="Use local shuffling (shuffle within each shard rather than over the entire dataset) for faster cache loading",
    )
    parser.add_argument(
        "--target",
        default="chat",
        choices=["chat", "base", "difference_bc", "difference_cb"],
        required=True,
        help="Target to train the SAE on. 'chat': train on chat model activations, 'base': train on base model activations, 'difference_bc': train on (base - chat) activation differences, 'difference_cb': train on (chat - base) activation differences",
    )
    parser.add_argument(
        "--pretrained-ae",
        type=str,
        default=None,
        help="Path to pretrained AE model",
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Load pretrained AE model from hub",
    )
    args = parser.parse_args()

    print(f"Training args: {args}")
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    if args.text_column == "text":
        lmsys_split_suffix = ""
        fineweb_split_suffix = ""
    else:
        lmsys_split_suffix = f"-col{args.text_column}"
        fineweb_split_suffix = ""

    activation_store_dir = Path(args.activation_store_dir)

    submodule_name = f"layer_{args.layer}_out"

    # Setup paths
    # Load validation dataset
    activation_store_dir = Path(args.activation_store_dir)

    base_model_stub = args.base_model.split("/")[-1]
    chat_model_stub = args.chat_model.split("/")[-1]
    fineweb_cache, lmsys_cache = load_activation_dataset(
        activation_store_dir,
        base_model=base_model_stub,
        instruct_model=chat_model_stub,
        layer=args.layer,
        lmsys_split="train" + lmsys_split_suffix,
        fineweb_split="train" + fineweb_split_suffix,
    )

    fineweb_cache, lmsys_cache = setup_cache(args, fineweb_cache, lmsys_cache)

    if args.target == "base":
        num_samples_per_dataset = min(args.num_samples, len(fineweb_cache))
        train_dataset = th.utils.data.Subset(
            fineweb_cache, th.arange(0, num_samples_per_dataset)
        )
        single_dataset = True
    elif args.target == "chat" or "difference" in args.target:
        num_samples_per_dataset = min(args.num_samples, len(lmsys_cache))
        train_dataset = th.utils.data.Subset(
            lmsys_cache, th.arange(0, num_samples_per_dataset)
        )
        single_dataset = True
    if args.local_shuffling:
        print(
            "Using local shuffling to optimize for cache locality while allowing randomization",
            flush=True,
        )
        # Create interleaved dataset of fineweb and lmsys samples

        # Shuffle within 1M sample shards while maintaining interleaving
        if isinstance(lmsys_cache, PairedActivationCache):
            shard_size = lmsys_cache.activation_cache_1.config["shard_size"]
        else:
            shard_size = lmsys_cache.config["shard_size"]
        num_shards_per_dataset = num_samples_per_dataset // shard_size + (
            1 if num_samples_per_dataset % shard_size != 0 else 0
        )
        print(f"Number of shards per dataset: {num_shards_per_dataset}", flush=True)

        shuffled_indices = []
        if args.epochs > 1:
            print(f"Using {args.epochs} epochs of local shuffling.", flush=True)
            for i in range(args.epochs):
                shuffled_indices.append(
                    get_local_shuffled_indices(
                        num_samples_per_dataset, shard_size, single_dataset
                    )
                )
            shuffled_indices = th.cat(shuffled_indices)
        else:
            shuffled_indices = get_local_shuffled_indices(
                num_samples_per_dataset, shard_size, single_dataset
            )
        print(f"Shuffled indices: {shuffled_indices.shape}", flush=True)
        train_dataset = th.utils.data.Subset(train_dataset, shuffled_indices)
        print(f"Shuffled train dataset with {len(train_dataset)} samples.", flush=True)
        args.no_train_shuffle = True
    else:
        assert (
            args.epochs == 1
        ), "Only one epoch of shuffling is supported if local shuffling is disabled."
        train_dataset = th.utils.data.ConcatDataset(
            [
                th.utils.data.Subset(
                    fineweb_cache, th.arange(0, num_samples_per_dataset)
                ),
                th.utils.data.Subset(
                    lmsys_cache, th.arange(0, num_samples_per_dataset)
                ),
            ]
        )

    activation_dim = train_dataset[0].shape[0]
    dictionary_size = args.expansion_factor * activation_dim

    fineweb_cache_val, lmsys_cache_val = load_activation_dataset(
        activation_store_dir,
        base_model=base_model_stub,
        instruct_model=chat_model_stub,
        layer=args.layer,
        lmsys_split="validation" + lmsys_split_suffix,
        fineweb_split="validation" + fineweb_split_suffix,
    )
    fineweb_cache_val, lmsys_cache_val = setup_cache(
        args, fineweb_cache_val, lmsys_cache_val
    )
    if "difference" in args.target:
        validation_dataset = th.utils.data.Subset(
            lmsys_cache_val, th.arange(0, args.num_validation_samples)
        )
    elif args.target == "base":
        validation_dataset = th.utils.data.Subset(
            fineweb_cache_val, th.arange(0, args.num_validation_samples)
        )
    elif args.target == "chat":
        validation_dataset = th.utils.data.Subset(
            lmsys_cache_val, th.arange(0, args.num_validation_samples)
        )

    name = (
        f"SAE-{args.target}-{args.base_model.split('/')[-1]}-L{args.layer}-k{args.k}-x{args.expansion_factor}-lr{args.lr:.0e}"
        + (f"-{args.run_name}" if args.run_name is not None else "")
        + ("-local-shuffling" if args.local_shuffling else "")
        + (f"-ft-{args.target}" if args.pretrained_ae is not None else "")
    )

    device = "cuda" if th.cuda.is_available() else "cpu"
    if args.max_steps is None:
        args.max_steps = len(train_dataset) // args.batch_size
    print(f"Training on device={device}.")
    trainer_cfg = {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": args.lr,
        "device": device,
        "warmup_steps": 1000,
        "layer": args.layer,
        "lm_name": f"{args.chat_model}-{args.base_model}",
        "wandb_name": name,
        "k": args.k,
        "steps": args.max_steps,
        "pretrained_ae": (
            BatchTopKSAE.from_pretrained(args.pretrained_ae, from_hub=args.from_hub)
            if args.pretrained_ae is not None
            else None
        ),
    }

    print(f"Training on {len(train_dataset)} token activations.")
    dataloader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # Nora said shuffling doesn't matter
        shuffle=not args.no_train_shuffle,
        num_workers=args.workers,
        pin_memory=True,
    )
    validation_dataloader = th.utils.data.DataLoader(
        validation_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # train the sparse autoencoder (SAE)
    ae = trainSAE(
        data=dataloader,
        trainer_config=trainer_cfg,
        validate_every_n_steps=args.validate_every_n_steps,
        validation_data=validation_dataloader,
        use_wandb=not args.disable_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project="crosscoder",
        log_steps=50,
        save_dir=f"checkpoints/{name}",
        steps=args.max_steps,
        save_steps=args.validate_every_n_steps,
    )
