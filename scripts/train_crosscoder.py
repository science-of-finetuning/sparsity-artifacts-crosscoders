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
from dictionary_learning.dictionary import LossType
import os

import wandb
wandb.require("legacy-service")

th.set_float32_matmul_precision("high")

from tools.utils import load_activation_dataset

def get_local_shuffled_indices(num_samples_per_dataset, shard_size):
    num_shards_per_dataset = num_samples_per_dataset // shard_size + (1 if num_samples_per_dataset % shard_size != 0 else 0)
    print(f"Number of shards per dataset: {num_shards_per_dataset}", flush=True)
    
    shuffled_indices = []
    for i in trange(num_shards_per_dataset):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, num_samples_per_dataset)
        shard_size_curr = end_idx - start_idx
        
        fineweb_indices = th.randperm(shard_size_curr) + start_idx
        lmsys_indices = th.randperm(shard_size_curr) + num_samples_per_dataset + start_idx
        
        shard_indices = th.zeros(2 * shard_size_curr, dtype=th.long)
        shard_indices[0::2] = fineweb_indices
        shard_indices[1::2] = lmsys_indices
        shuffled_indices.append(shard_indices)
        
    shuffled_indices = th.cat(shuffled_indices)
    return shuffled_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--wandb-entity", type=str, default="jkminder")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mu", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validate-every-n-steps", type=int, default=10000)
    parser.add_argument("--same-init-for-all-layers", action="store_true")
    parser.add_argument("--norm-init-scale", type=float, default=0.005)
    parser.add_argument("--init-with-transpose", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resample-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None, nargs="+")
    parser.add_argument("--num-samples", type=int, default=200_000_000)
    parser.add_argument("--num-validation-samples", type=int, default=2_000_000)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--no-train-shuffle", action="store_true")
    parser.add_argument("--local-shuffling", action="store_true")
    parser.add_argument("--loss-type", type=str, default="crosscoder", choices=["crosscoder", "sae"])

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
    num_samples_per_dataset = args.num_samples // 2
    num_samples_per_dataset = min(num_samples_per_dataset, len(fineweb_cache))
    num_samples_per_dataset = min(num_samples_per_dataset, len(lmsys_cache))
    train_dataset = th.utils.data.ConcatDataset([
        th.utils.data.Subset(fineweb_cache, th.arange(0, num_samples_per_dataset)),
        th.utils.data.Subset(lmsys_cache, th.arange(0, num_samples_per_dataset)),
    ])

    if args.local_shuffling:
        print("Using local shuffling to optimize for cache locality while allowing randomization", flush=True)
        # Create interleaved dataset of fineweb and lmsys samples

        # Shuffle within 1M sample shards while maintaining interleaving
        shard_size = lmsys_cache.activation_cache_1.config["shard_size"] 
        num_shards_per_dataset = num_samples_per_dataset // shard_size + (1 if num_samples_per_dataset % shard_size != 0 else 0)
        print(f"Number of shards per dataset: {num_shards_per_dataset}", flush=True)

        shuffled_indices = []
        if args.epochs > 1:
            print(f"Using {args.epochs} epochs of local shuffling.", flush=True)
            for i in range(args.epochs):
                shuffled_indices.append(get_local_shuffled_indices(num_samples_per_dataset, shard_size))
            shuffled_indices = th.cat(shuffled_indices)
        else:
            shuffled_indices = get_local_shuffled_indices(num_samples_per_dataset, shard_size)
        print(f"Shuffled indices: {shuffled_indices.shape}", flush=True)
        train_dataset = th.utils.data.Subset(train_dataset, shuffled_indices)
        print(f"Shuffled train dataset with {len(train_dataset)} samples.", flush=True)
        args.no_train_shuffle = True
    else:
        assert args.epochs == 1, "Only one epoch of shuffling is supported if local shuffling is disabled."
        train_dataset = th.utils.data.ConcatDataset(
            [
                th.utils.data.Subset(fineweb_cache, th.arange(0, num_samples_per_dataset)),
                th.utils.data.Subset(lmsys_cache, th.arange(0, num_samples_per_dataset)),
            ]
        )

    activation_dim = train_dataset[0].shape[1]
    dictionary_size = args.expansion_factor * activation_dim


    fineweb_cache_val, lmsys_cache_val = load_activation_dataset(
        activation_store_dir,
        base_model=base_model_stub,
        instruct_model=chat_model_stub,
        layer=args.layer,
        lmsys_split="validation" + lmsys_split_suffix,
        fineweb_split="validation" + fineweb_split_suffix,
    )
    num_validation_samples = args.num_validation_samples // 2
    validation_dataset = th.utils.data.ConcatDataset([
        th.utils.data.Subset(fineweb_cache_val, th.arange(0, num_validation_samples)),
        th.utils.data.Subset(lmsys_cache_val, th.arange(0, num_validation_samples)),
    ])
    
    name = f"{args.base_model.split('/')[-1]}-L{args.layer}-mu{args.mu:.1e}-lr{args.lr:.0e}" + \
        (f"-{args.run_name}" if args.run_name is not None else "") + \
        (f"-local-shuffling" if args.local_shuffling else "") + \
        (f"-CCloss" if args.loss_type == "crosscoder" else "-SAEloss")
    if args.pretrained is not None:
        name += f"-pt"
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device={device}.")
    loss_type = LossType.from_string(args.loss_type)
    print(f"Loss type: {loss_type}")
    trainer_cfg = {
        "trainer": CrossCoderTrainer,
        "dict_class": CrossCoder,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": args.lr,
        "resample_steps": args.resample_steps,
        "device": device,
        "warmup_steps": 1000,
        "layer": args.layer,
        "lm_name": f"{args.chat_model}-{args.base_model}",
        "compile": True,
        "wandb_name": name,
        "l1_penalty": args.mu,
        "dict_class_kwargs": {
            "same_init_for_all_layers": args.same_init_for_all_layers,
            "norm_init_scale": args.norm_init_scale,
            "init_with_transpose": args.init_with_transpose,
            "encoder_layers": args.encoder_layers,
            "loss_type": loss_type,
        },
        "pretrained_ae": (
            CrossCoder.from_pretrained(args.pretrained)
            if args.pretrained is not None
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
