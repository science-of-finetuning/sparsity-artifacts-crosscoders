import torch as th
import argparse
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import json

from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer, FeatureScalerTrainer
from dictionary_learning.dictionary import FeatureScaler
from dictionary_learning.training import trainSAE
import os

th.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-indices-file", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=0)
    parser.add_argument("--source-layer", type=int, default=1)
    parser.add_argument("--activation-store-dir", type=str, default="activations")
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
    args = parser.parse_args()
    assert not (
        args.random_indices and args.random_source
    ), "Cannot specify both random-indices and random-source."

    print(f"Training args: {args}")
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    activation_store_dir = Path(args.activation_store_dir)

    base_model_dir = activation_store_dir / args.base_model
    instruct_model_dir = activation_store_dir / args.instruct_model

    base_model_fineweb = base_model_dir / "fineweb"
    base_model_lmsys_chat = base_model_dir / "lmsys_chat"
    instruct_model_fineweb = instruct_model_dir / "fineweb"
    instruct_model_lmsys_chat = instruct_model_dir / "lmsys_chat"

    submodule_name = f"layer_{args.layer}_out"

    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_chat_cache = PairedActivationCache(
        base_model_lmsys_chat / submodule_name,
        instruct_model_lmsys_chat / submodule_name,
    )

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    # load the cross-coder and modify the decoder layers
    cc = CrossCoder.from_pretrained(args.cc)
    target_decoder_layers = [args.target_layer]
    # load the feature indices
    if args.feature_indices_file is None:
        indices = th.tensor([])
    else:
        indices = th.load(args.feature_indices_file)
    print(
        f"Loaded {len(indices)} feature indices to modify from {args.feature_indices_file}."
    )
    if args.random_indices:
        indices = th.randperm(cc.dict_size)[: len(indices)]
        print(f"Using {len(indices)} random indices.")

    decoder_weight = th.clone(cc.decoder.weight.data)
    if args.random_source:
        # randomly select same number of indices as source layer has non-zero features
        source_norms = decoder_weight[args.source_layer, indices].norm(
            dim=-1, keepdim=True
        )
        random_vectors = th.randn_like(decoder_weight[args.source_layer, indices])
        random_vectors = (
            random_vectors / random_vectors.norm(dim=-1, keepdim=True) * source_norms
        )
        decoder_weight[args.target_layer, indices, :] = random_vectors
        print(f"Using {len(indices)} random source vectors with matched norms.")
    elif len(indices):
        decoder_weight[args.target_layer, indices, :] = decoder_weight[
            args.source_layer, indices, :
        ]
        print(f"Using vectors from source layer {args.source_layer}.")
    else:
        print("No indices to modify.")

    cc.decoder.weight = th.nn.Parameter(decoder_weight)

    # only allow gradients on the scaler parameters for the indices
    mask = th.ones(cc.dict_size, device="cuda")
    if len(indices):
        mask[indices] = 0.0
    else:
        mask = th.zeros(cc.dict_size, device="cuda")
    print(
        f"Masking {int(mask.sum().item())} out of {cc.dict_size} feature scaler parameters."
    )
    feature_scaler = FeatureScaler(
        cc.dict_size, fixed_mask=mask.bool(), zero_init=args.zero_init_scaler
    )

    activation_dim = cc.activation_dim
    dictionary_size = cc.dict_size
    run_name = (
        f"L{args.layer}-mu{args.mu:.1e}-lr{args.lr:.0e}"
        + (f"-{args.run_name}" if args.run_name is not None else "")
        + ("-ZeroInit" if args.zero_init_scaler else "")
    )
    if args.random_indices:
        run_name = "RandomIndices" + run_name
    if args.random_source:
        run_name = "RandomSource" + run_name
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device={device}.")
    trainer_cfg = {
        "trainer": FeatureScalerTrainer,
        "dict_class": CrossCoder,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": args.lr,
        "device": device,
        "warmup_steps": 1000,
        "compile": True,
        "wandb_name": "FS-" + run_name,
        "l1_penalty": args.mu,
        "cross_coder": cc,
        "feature_scaler": feature_scaler,
        "target_decoder_layers": target_decoder_layers,
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
    )
    validation_dataloader = th.utils.data.DataLoader(
        validation_dataset,
        batch_size=8192,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    ae = trainSAE(
        data=dataloader,
        trainer_config=trainer_cfg,
        validate_every_n_steps=args.validate_every_n_steps,
        validation_data=validation_dataloader,
        use_wandb=not args.disable_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project="cross_coder_feature_scaler",
        log_steps=10,
        steps=args.max_steps,
    )

    # save the feature scaler
    out_dir = Path("checkpoints") / "feature_scaler" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f)
    th.save(
        feature_scaler.state_dict(),
        out_dir / f"scaler_{args.target_layer}_{args.source_layer}.pt",
    )
