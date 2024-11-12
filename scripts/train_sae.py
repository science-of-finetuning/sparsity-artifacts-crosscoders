import torch as th
import argparse
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache, ActivationCache
from dictionary_learning.dictionary import AutoEncoderNew
from torch.utils.data import default_collate
from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer, StandardTrainer
from dictionary_learning.training import trainSAE
import os

th.set_float32_matmul_precision("high")


def collate_fn(batch):
    batch = default_collate(batch)
    b, l, d = batch.shape
    # we want b, l*d
    return batch.view(b, l * d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--model", type=str, default="gemma-2-2b")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--wandb-name", type=str, default="jkminder")
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--mu", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-every-n-steps", type=int, default=10000)
    args = parser.parse_args()

    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    activation_store_dir = Path(args.activation_store_dir)

    submodule_name = f"layer_{args.layer}_out"
    model_dir = activation_store_dir / args.model

    fineweb_cache = ActivationCache(model_dir / "fineweb" / submodule_name)
    lmsys_chat_cache = ActivationCache(model_dir / "lmsys_chat" / submodule_name)

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    activation_dim = dataset[0].shape[1] * 2
    dictionary_size = args.expansion_factor * activation_dim

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device={device}.")
    trainer_cfg = {
        "trainer": StandardTrainer,
        "dict_class": AutoEncoderNew,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": 1e-3,
        "device": device,
        "warmup_steps": 1000,
        "layer": args.layer,
        "lm_name": f"{args.instruct_model}-{args.base_model}",
        "compile": True,
        "wandb_name": "CrossCoderTest",
        "l1_penalty": args.mu,
    }

    validation_size = 10**6
    train_dataset, validation_dataset = th.utils.data.random_split(
        dataset, [len(dataset) - validation_size, validation_size]
    )
    print(f"Training on {len(train_dataset)} token activations.")
    dataloader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    validation_dataloader = th.utils.data.DataLoader(
        validation_dataset,
        batch_size=8192,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # train the sparse autoencoder (SAE)
    ae = trainSAE(
        data=dataloader,
        trainer_configs=[trainer_cfg],
        validate_every_n_steps=None,  # args.validate_every_n_steps,
        validation_data=validation_dataloader,
        use_wandb=args.wandb_name is not None,
        wandb_entity=args.wandb_name,
        wandb_project="crosscoder",
        log_steps=10,
        save_steps=10000,
        save_dir="checkpoints",
    )
