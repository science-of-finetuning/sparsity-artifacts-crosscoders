import torch as th
import argparse
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import json
from loguru import logger

from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer, FeatureScalerTrainer
from dictionary_learning.dictionary import FeatureScaler
from dictionary_learning.training import trainSAE
import os
from tqdm import tqdm
from typing import Optional
th.set_float32_matmul_precision("high")


def train_feature_scaler(
    target_layer: int,
    source_layer: int,
    activation_store_dir: str | Path,
    output_dir: str | Path,
    base_model: str,
    instruct_model: str,
    layer: int,
    wandb_entity: str,
    disable_wandb: bool,
    batch_size: int,
    workers: int,
    mu: float,
    seed: int,
    max_steps: int,
    validate_every_n_steps: int,
    run_name: str,
    lr: float,
    cc_path: str | Path,
    zero_init_scaler: bool,
    random_source: bool,
    random_indices: bool,
    feature_indices: Optional[th.Tensor] = None,
    dataset_split: str = "train",
) -> None:
    logger.info(f"Training with seed={seed}, lr={lr}, mu={mu}")
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    activation_store_dir = Path(activation_store_dir)
    base_model_dir = activation_store_dir / base_model
    instruct_model_dir = activation_store_dir / instruct_model

    base_model_fineweb = base_model_dir / "fineweb-1m-sample" / dataset_split
    base_model_lmsys_chat = base_model_dir / "lmsys-chat-1m-gemma-formatted" / dataset_split
    instruct_model_fineweb = instruct_model_dir / "fineweb-1m-sample" / dataset_split
    instruct_model_lmsys_chat = instruct_model_dir / "lmsys-chat-1m-gemma-formatted" / dataset_split

    submodule_name = f"layer_{layer}_out"

    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_chat_cache = PairedActivationCache(
        base_model_lmsys_chat / submodule_name,
        instruct_model_lmsys_chat / submodule_name,
    )

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    # load the cross-coder and modify the decoder layers
    cc = CrossCoder.from_pretrained(cc_path)
    target_decoder_layers = [target_layer]
    # load the feature indices
    if feature_indices is None:
        indices = th.tensor([])
    else:
        indices = th.load(feature_indices)
    print(f"Got {len(indices)} feature indices to modify.")
    if random_indices:
        indices = th.randperm(cc.dict_size)[: len(indices)]
        print(f"Using {len(indices)} random indices.")

    decoder_weight = th.clone(cc.decoder.weight.data)
    if random_source:
        # randomly select same number of indices as source layer has non-zero features
        source_norms = decoder_weight[source_layer, indices].norm(dim=-1, keepdim=True)
        random_vectors = th.randn_like(decoder_weight[source_layer, indices])
        random_vectors = (
            random_vectors / random_vectors.norm(dim=-1, keepdim=True) * source_norms
        )
        decoder_weight[target_layer, indices, :] = random_vectors
        logger.info(f"Using {len(indices)} random source vectors with matched norms.")
    elif len(indices):
        decoder_weight[target_layer, indices, :] = decoder_weight[
            source_layer, indices, :
        ]
        logger.info(f"Using vectors from source layer {source_layer}.")
    else:
        logger.info("No indices to modify.")

    cc.decoder.weight = th.nn.Parameter(decoder_weight)

    # only allow gradients on the scaler parameters for the indices
    mask = th.ones(cc.dict_size, device="cuda")
    if len(indices):
        mask[indices] = 0.0
    else:
        mask = th.zeros(cc.dict_size, device="cuda")
    logger.info(
        f"Masking {int(mask.sum().item())} out of {cc.dict_size} feature scaler parameters."
    )
    feature_scaler = FeatureScaler(
        cc.dict_size, fixed_mask=mask.bool(), zero_init=zero_init_scaler
    )

    activation_dim = cc.activation_dim
    dictionary_size = cc.dict_size
    run_name_str = (
        f"L{layer}-mu{mu:.1e}-lr{lr:.0e}"
        + f"Ind" + str(feature_indices[0]) if len(feature_indices) == 1 else ""
        + (f"-{run_name}" if run_name is not None else "")
        + ("-ZeroInit" if zero_init_scaler else "")
    )
    if random_indices:
        run_name_str = "RandomIndices" + run_name_str
    if random_source:
        run_name_str = "RandomSource" + run_name_str
    device = "cuda" if th.cuda.is_available() else "cpu"
    logger.info(f"Training on device={device}.")
    trainer_cfg = {
        "trainer": FeatureScalerTrainer,
        "dict_class": CrossCoder,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": lr,
        "device": device,
        "warmup_steps": 1000,
        "compile": True,
        "wandb_name": "FS-" + run_name_str,
        "l1_penalty": mu,
        "cross_coder": cc,
        "feature_scaler": feature_scaler,
        "target_decoder_layers": target_decoder_layers,
    }

    validation_size = 10**6
    train_dataset, validation_dataset = th.utils.data.random_split(
        dataset, [len(dataset) - validation_size, validation_size]
    )
    logger.info(f"Training on {len(train_dataset)} token activations.")
    dataloader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    validation_dataloader = th.utils.data.DataLoader(
        validation_dataset,
        batch_size=8192,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    ae = trainSAE(
        data=dataloader,
        trainer_config=trainer_cfg,
        validate_every_n_steps=validate_every_n_steps,
        validation_data=validation_dataloader,
        use_wandb=not disable_wandb,
        wandb_entity=wandb_entity,
        wandb_project="cross_coder_feature_scaler",
        log_steps=10,
        steps=max_steps,
        save_last_eval=True,
    )

    # save the feature scaler
    out_dir = Path(output_dir) / "feature_scaler" / run_name_str
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(
            {
                "feature_indices": feature_indices.tolist() if feature_indices is not None else [],
                "target_layer": target_layer,
                "source_layer": source_layer,
                "activation_store_dir": str(activation_store_dir),
                "base_model": base_model,
                "instruct_model": instruct_model,
                "layer": layer,
                "wandb_entity": wandb_entity,
                "disable_wandb": disable_wandb,
                "batch_size": batch_size,
                "workers": workers,
                "mu": mu,
                "seed": seed,
                "max_steps": max_steps,
                "validate_every_n_steps": validate_every_n_steps,
                "run_name": run_name,
                "lr": lr,
                "cc_path": cc_path,
                "zero_init_scaler": zero_init_scaler,
                "random_source": random_source,
                "random_indices": random_indices,
                "dataset_split": dataset_split,
            },
            f,
        )
    th.save(
        feature_scaler.state_dict(),
        out_dir / f"scaler_{target_layer}_{source_layer}.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-indices-file", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=0)
    parser.add_argument("--source-layer", type=int, default=1)
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--base-model", type=str, default="gemma-2-2b")
    parser.add_argument("--instruct-model", type=str, default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--wandb-entity", type=str, default="jkminder")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--mu", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validate-every-n-steps", type=int, default=10000)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cc", type=str, required=True)
    parser.add_argument("--zero-init-scaler", action="store_true")
    parser.add_argument("--random-source", action="store_true")
    parser.add_argument("--random-indices", action="store_true")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--individual-indices", action="store_true")

    args = parser.parse_args()
    assert not (
        args.random_indices and args.random_source
    ), "Cannot specify both random-indices and random-source."

    if args.feature_indices_file is not None:
        feature_indices = th.load(args.feature_indices_file)
    else:
        feature_indices = None

    if args.individual_indices:
        assert (
            feature_indices is not None
        ), "Must provide feature indices when using individual indices"
        for i, index in enumerate(feature_indices):
            logger.info(
                f"Training with index {index} ({i / len(feature_indices) * 100:.2f}% of {len(feature_indices)} indices)"
            )
            train_feature_scaler(
                target_layer=args.target_layer,
                source_layer=args.source_layer,
                activation_store_dir=args.activation_store_dir,
                base_model=args.base_model,
                instruct_model=args.instruct_model,
                layer=args.layer,
                wandb_entity=args.wandb_entity,
                disable_wandb=args.disable_wandb,
                batch_size=args.batch_size,
                workers=args.workers,
                mu=args.mu,
                seed=args.seed,
                max_steps=args.max_steps,
                validate_every_n_steps=args.validate_every_n_steps,
                run_name=args.run_name,
                lr=args.lr,
                cc_path=args.cc,
                zero_init_scaler=args.zero_init_scaler,
                random_source=args.random_source,
                random_indices=args.random_indices,
                feature_indices=th.tensor([index]),
                dataset_split=args.dataset_split,
                output_dir=args.output_dir,
            )
    else:
        train_feature_scaler(
            target_layer=args.target_layer,
            source_layer=args.source_layer,
            activation_store_dir=args.activation_store_dir,
            base_model=args.base_model,
            instruct_model=args.instruct_model,
            layer=args.layer,
            wandb_entity=args.wandb_entity,
            disable_wandb=args.disable_wandb,
            batch_size=args.batch_size,
            workers=args.workers,
            mu=args.mu,
            seed=args.seed,
            max_steps=args.max_steps,
            validate_every_n_steps=args.validate_every_n_steps,
            run_name=args.run_name,
            lr=args.lr,
            cc_path=args.cc,
            zero_init_scaler=args.zero_init_scaler,
            random_source=args.random_source,
            random_indices=args.random_indices,
            feature_indices=feature_indices,
            dataset_split=args.dataset_split,
            output_dir=args.output_dir,
        )
