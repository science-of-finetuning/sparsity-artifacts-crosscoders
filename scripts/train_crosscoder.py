import torch as th
import argparse
from pathlib import Path
from nnsight import LanguageModel
from dictionary_learning.cache import PairedActivationCache
from dictionary_learning import ActivationBuffer, CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer
from dictionary_learning.training import trainSAE
import os

from tools.utils import load_activation_dataset

th.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--base-model", type=str, default="gemma-2-2b")
    parser.add_argument("--instruct-model", type=str, default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--wandb-entity", type=str, default="jkminder")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--mu", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validate-every-n-steps", type=int, default=10000)
    parser.add_argument("--same-init-for-all-layers", action="store_true")
    parser.add_argument("--norm-init-scale", type=float, default=0.005)
    parser.add_argument("--init-with-transpose", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resample-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None, nargs="+")
    parser.add_argument("--num-samples", type=int, default=200_000_000)
    parser.add_argument("--num-validation-samples", type=int, default=2_000_000)
    parser.add_argument("--text-column", type=str, default="text")
    args = parser.parse_args()

    print(f"Training args: {args}")
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    if args.text_column == "text":
        lmsys_split_suffix = ""
        fineweb_split_suffix = ""
    else:
        lmsys_split_suffix = f"-col{args.text_column}"
        fineweb_split_suffix = f"-col{args.text_column}"

    activation_store_dir = Path(args.activation_store_dir)

    base_model_dir = activation_store_dir / args.base_model
    instruct_model_dir = activation_store_dir / args.instruct_model

    base_model_fineweb = base_model_dir / "fineweb"
    base_model_lmsys_chat = base_model_dir / "lmsys_chat"
    instruct_model_fineweb = instruct_model_dir / "fineweb"
    instruct_model_lmsys_chat = instruct_model_dir / "lmsys_chat"

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
    
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device={device}.")
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
        "lm_name": f"{args.instruct_model}-{args.base_model}",
        "compile": True,
        "wandb_name": f"L{args.layer}-mu{args.mu:.1e}-lr{args.lr:.0e}"
        + (f"-{args.run_name}" if args.run_name is not None else ""),
        "l1_penalty": args.mu,
        "dict_class_kwargs": {
            "same_init_for_all_layers": args.same_init_for_all_layers,
            "norm_init_scale": args.norm_init_scale,
            "init_with_transpose": args.init_with_transpose,
            "encoder_layers": args.encoder_layers,
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
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    validation_dataloader = th.utils.data.DataLoader(
        validation_dataset,
        batch_size=8192,
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
        save_dir="checkpoints",
        steps=args.max_steps,
        save_steps=args.validate_every_n_steps,
    )
