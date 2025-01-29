import sys

sys.path.append(".")
import torch as th
from typing import Callable, Union
from dictionary_learning import CrossCoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import numpy as np
from loguru import logger
from functools import partial
import argparse
import os

th.set_grad_enabled(False)
th.set_float32_matmul_precision("highest")

from tools.latent_scaler.closed_form import remove_latents
from tools.utils import load_connor_crosscoder

 # Define processing functions (reusing from compute_scalers.py)
from compute_scalers import load_base_reconstruction, load_base_error, load_base_activation
from compute_scalers import load_chat_reconstruction, load_chat_error, load_chat_activation

def compute_variance_explained(
    betas: th.Tensor,
    latent_vectors: th.Tensor,
    latent_indices: th.Tensor,
    dataloader: DataLoader,
    crosscoder: CrossCoder,
    activation_postprocessing_fn: Callable[[th.Tensor], th.Tensor],
    target_fn: Callable[[th.Tensor], th.Tensor],
    latent_activation_postprocessing_fn: Callable[[th.Tensor], th.Tensor] = None,
    device: th.device = th.device("cuda"),
    dtype: Union[th.dtype, None] = th.float32,
) -> dict:
    """
    Compute fraction of explained variance for each scaler.
    """
    assert betas.shape == (len(latent_indices),)
    total_variance = th.zeros(len(latent_indices), device=device, dtype=dtype)
    explained_variance = th.zeros(len(latent_indices), device=device, dtype=dtype)
    before_explained_variance = th.zeros(len(latent_indices), device=device, dtype=dtype)
    count = th.zeros(len(latent_indices), device=device, dtype=dtype)

    for batch in tqdm(dataloader, desc="Computing variance explained"):
        batch_size = batch.shape[0]
        batch = batch.to(dtype).to(device)

        # Get latent activations
        latent_activations = crosscoder.encode(batch)
        if latent_activation_postprocessing_fn is not None:
            latent_activations = latent_activation_postprocessing_fn(latent_activations)

        # Get target activations
        target = target_fn(
            batch,
            crosscoder=crosscoder,
            latent_activations=latent_activations,
            latent_indices=latent_indices,
            latent_vectors=latent_vectors,
        )
        activation = activation_postprocessing_fn(
            batch,
            crosscoder=crosscoder,
            latent_activations=latent_activations,
            latent_indices=latent_indices,
            latent_vectors=latent_vectors,
        )
        print(activation.shape, target.shape)
        # Get relevant latent activations
        latent_activations = latent_activations[:, latent_indices]
        assert latent_activations.shape == (batch_size, len(latent_indices))

        # Scale latent activations by betas
        scaled_latents = latent_activations * betas.unsqueeze(0)
        assert scaled_latents.shape == (batch_size, len(latent_indices))

        # Compute reconstructions using scaled latents
        for i in range(len(latent_indices)):
            if activation.dim() == 2:
                assert scaled_latents[:, i].shape == (batch_size,)
                assert latent_vectors[i].shape == (activation.shape[-1],)
                reconstructions = activation + scaled_latents[:, i].unsqueeze(1) * latent_vectors[i].unsqueeze(0)
                activation_before = activation
            else:
                assert activation.shape == (len(latent_indices),batch_size, activation.shape[-1])
                reconstructions = activation[i] + scaled_latents[:, i].unsqueeze(1) * latent_vectors[i].unsqueeze(0)
                activation_before = activation[i]
            # Compute variances
            assert target.shape == reconstructions.shape
            target_var = th.var(target, dim=0).sum()  # Total variance across all dimensions
            residual_var = th.var(target - reconstructions, dim=0).sum()
            before_residual_var = th.var(target - activation_before, dim=0).sum()
            assert target_var.shape == ()
            assert residual_var.shape == ()
            explained_variance[i] += 1 - residual_var / target_var
            count[i] += 1
            before_explained_variance[i] += 1 - before_residual_var / target_var

        # Clean up GPU memory
        del latent_activations, target, reconstructions
        if device == "cuda":
            th.cuda.empty_cache()

    # Compute final metrics
    frac_explained = explained_variance / count
    frac_explained_before = before_explained_variance / count
    return {
        "frac_variance_explained": frac_explained.cpu(),
        "frac_variance_explained_no_scaler": frac_explained_before.cpu(),
        "count": count.cpu(),
    }

def load_betas(args, computation, results_dir):
    name = computation
    if args.threshold_active_latents is not None:
        name += f"_jumprelu{args.threshold_active_latents}"
    if args.name:
        name += f"_{args.name}"
        
    betas_path = results_dir / f"betas_{name}.pt"
    logger.info(f"Processing betas from: {betas_path}")
    
    if not betas_path.exists():
        raise FileNotFoundError(f"Betas file not found: {betas_path}")

    # Load betas and continue with existing logic
    betas = th.load(betas_path)
    return betas, name

def main():
    # Reuse argument parsing from compute_scalers.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--crosscoder-path", type=str, required=True)
    parser.add_argument("--activation-store-dir", type=Path, required=True)
    parser.add_argument("--chat-only-indices-path", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="gemma-2-2b")
    parser.add_argument("--instruct-model", type=str, default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float64", "bfloat16"])
    # Add compute_scalers.py specific arguments for betas path construction
    parser.add_argument("--results-dir", type=Path, default="/workspace/data/results/closed_form_scalars")
    parser.add_argument("--threshold-active-latents", type=float, default=None)
    parser.add_argument("--chat-error", action="store_true")
    parser.add_argument("--chat-reconstruction", action="store_true")
    parser.add_argument("--base-error", action="store_true")
    parser.add_argument("--base-reconstruction", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1000000)
    args = parser.parse_args()

    # Construct betas path following compute_scalers.py logic
    results_dir = args.results_dir / args.crosscoder_path.replace("/", "_")
    

    # Setup device and dtype
    if args.device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = th.device(args.device)

    dtype_map = {"float32": th.float32, "float64": th.float64, "bfloat16": th.bfloat16}
    dtype = dtype_map[args.dtype]
    logger.info(f"Using dtype: {dtype}")

    # Load crosscoder
    cc = CrossCoder.from_pretrained(args.crosscoder_path, from_hub=True)
    cc = cc.to(device).to(dtype)

    # Load validation dataset
    activation_store_dir = Path(args.activation_store_dir)
    base_model_dir = activation_store_dir / args.base_model
    instruct_model_dir = activation_store_dir / args.instruct_model

    submodule_name = f"layer_{args.layer}_out"

    # Load validation caches
    base_model_fineweb = base_model_dir / "fineweb-1m-sample/validation"
    base_model_lmsys = base_model_dir / "lmsys-chat-1m-gemma-formatted/validation"
    instruct_model_fineweb = instruct_model_dir / "fineweb-1m-sample/validation"
    instruct_model_lmsys = (
        instruct_model_dir / "lmsys-chat-1m-gemma-formatted/validation"
    )

    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_cache = PairedActivationCache(
        base_model_lmsys / submodule_name, instruct_model_lmsys / submodule_name
    )

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_cache])
    dataset = th.utils.data.Subset(dataset, th.arange(args.num_samples))
    logger.info(f"Number of activations: {len(dataset)}")
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Load betas and chat indices
    chat_only_indices = th.load(args.chat_only_indices_path)

    # Get decoder weights
    it_decoder = cc.decoder.weight[1, :, :].clone().to(dtype)
    base_decoder = cc.decoder.weight[0, :, :].clone().to(dtype)
    latent_vectors = it_decoder[chat_only_indices].clone()

    # Determine which computation to verify
    computations = []
    if args.base_reconstruction:
        computations.append(("base_reconstruction", load_base_reconstruction, load_base_activation))
    if args.base_error:
        computations.append(("base_error", partial(load_base_error, base_decoder=base_decoder), load_base_activation))
    if args.chat_reconstruction:
        computations.append(("it_reconstruction", load_chat_reconstruction, load_chat_activation))
    if args.chat_error:
        computations.append(("it_error", load_chat_error, load_chat_activation))
    
    if len(computations) == 0:
        logger.info("No computations selected, running all")
        computations = [
            ("base_reconstruction", load_base_reconstruction, load_base_activation),
            ("base_error", partial(load_base_error, base_decoder=base_decoder), load_base_activation),
            ("it_reconstruction", load_chat_reconstruction, load_chat_activation),
            ("it_error", load_chat_error, load_chat_activation),
        ]
    # Compute variance explained for each computation type
    for name, loader_fn, target_fn in computations:
        logger.info(f"Computing variance explained for {name}")
        betas, betas_name = load_betas(args, name, results_dir)
        metrics = compute_variance_explained(
            betas=betas,
            latent_vectors=latent_vectors,
            latent_indices=chat_only_indices,
            dataloader=dataloader,
            crosscoder=cc,
            activation_postprocessing_fn=loader_fn,
            target_fn=target_fn,
            device=device,
            dtype=dtype,
        )
        # Save results
        output_path = (
            results_dir / f"fve_{betas_name}.pt"
        )
        th.save(metrics, output_path)
        logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
