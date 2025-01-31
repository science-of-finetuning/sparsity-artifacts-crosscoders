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
from torchmetrics import MeanSquaredError
import os

th.set_grad_enabled(False)
th.set_float32_matmul_precision("highest")

from tools.latent_scaler.closed_form import remove_latents
from tools.utils import load_connor_crosscoder

# Define processing functions (reusing from compute_scalers.py)
from compute_scalers import (
    load_base_reconstruction,
    load_base_error,
    load_base_activation,
)
from compute_scalers import (
    load_chat_reconstruction,
    load_chat_error,
    load_chat_activation,
)

def get_bucket_edges(max_activations, n_buckets, add_noise_threshold=0.05):
    step_size = 1 / n_buckets
    steps = th.arange(n_buckets+1) * step_size
    if add_noise_threshold is not None:
        steps = th.cat([th.tensor([0, add_noise_threshold]), steps[1:]]).to(max_activations.device)
    return th.stack([steps*max_act for max_act in max_activations])

def compute_stats(
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
    max_activations: th.Tensor = None,
    compute_fve: bool = False,
    n_buckets: int = 3,
    add_noise_threshold: float = None,
) -> dict:
    """
    Compute statistics for each scaler.
    """
    dim_model = crosscoder.activation_dim
    assert betas.shape == (len(latent_indices),)
    explained_variance = th.zeros(len(latent_indices), device=device, dtype=dtype)
    before_explained_variance = th.zeros(
        len(latent_indices), device=device, dtype=dtype
    )
    mse = th.zeros(len(latent_indices), device=device, dtype=th.float64)
    mse_before = th.zeros(len(latent_indices), device=device, dtype=th.float64)
    count = th.zeros(len(latent_indices), device=device, dtype=dtype)

    if max_activations is not None:
        if add_noise_threshold is not None:
            n_buckets += 1
        assert max_activations.shape == (crosscoder.dict_size,)
        max_activations = max_activations[latent_indices].to(device)
        assert len(max_activations) == len(latent_indices)
        bucket_edges = get_bucket_edges(max_activations, n_buckets, add_noise_threshold)
        mse_buckets = th.zeros(n_buckets, len(latent_indices), device=device, dtype=th.float64)
        mse_before_buckets = th.zeros(n_buckets, len(latent_indices), device=device, dtype=th.float64)
        mse_count = th.zeros(n_buckets, len(latent_indices), device=device, dtype=dtype)

    for batch in tqdm(dataloader, desc="Computing scaler stats"):
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

        # Get relevant latent activations
        latent_activations = latent_activations[:, latent_indices]
        assert latent_activations.shape == (batch_size, len(latent_indices))

        # Scale latent activations by betas
        scaled_latents = latent_activations * betas.unsqueeze(0)
        assert scaled_latents.shape == (batch_size, len(latent_indices))

        # Compute reconstructions using scaled latents (vectorized)
        if activation.dim() == 2:
            # [batch_size, hidden_dim] + [batch_size, n_latents, 1] * [n_latents, hidden_dim]
            reconstructions = activation.unsqueeze(1) + scaled_latents.unsqueeze(
                2
            ) * latent_vectors.unsqueeze(0)
            reconstructions = reconstructions.permute(1, 0, 2)
            activation_before = activation.unsqueeze(0).expand_as(reconstructions)

            assert reconstructions.shape == (len(latent_indices), batch_size, dim_model)
            assert activation_before.shape == (
                len(latent_indices),
                batch_size,
                dim_model,
            )
        else:
            assert activation.shape == (len(latent_indices), batch_size, dim_model)
            assert scaled_latents.shape == (batch_size, len(latent_indices))
            assert latent_vectors.shape == (len(latent_indices), dim_model)

            # activation is already [n_latents, batch_size, hidden_dim]
            # [n_latents, batch_size, hidden_dim] + [batch_size, n_latents, 1] * [n_latents, hidden_dim]
            reconstructions = activation + scaled_latents.transpose(0, 1).unsqueeze(
                2
            ) * latent_vectors.unsqueeze(1)
            activation_before = activation

            assert reconstructions.shape == (len(latent_indices), batch_size, dim_model)
            assert activation_before.shape == (
                len(latent_indices),
                batch_size,
                dim_model,
            )

        # Compute variances
        if target.dim() == 2:
            assert target.shape == (batch_size, dim_model)
            target = target.unsqueeze(0).expand_as(reconstructions)
            assert target.shape == reconstructions.shape

        residual = target - reconstructions
        residual_before = target - activation_before
        assert residual.shape == (len(latent_indices), batch_size, dim_model)
        if compute_fve:
            # [n_latents]
            target_var = th.var(target, dim=1).sum(dim=-1)
            residual_var = th.var(residual, dim=1).sum(dim=-1)
            before_residual_var = th.var(residual_before, dim=1).sum(dim=-1)

            assert target_var.shape == (len(latent_indices),)
            assert residual_var.shape == (len(latent_indices),)
            assert before_residual_var.shape == (len(latent_indices),)

            explained_variance += 1 - residual_var / target_var
            before_explained_variance += 1 - before_residual_var / target_var

        # residual is [n_latents, batch_size, dim_model]
        mse += residual.pow(2).sum(dim=1).mean(dim=-1)  # [n_latents]
        mse_before += residual_before.pow(2).sum(dim=1).mean(dim=-1)  # [n_latents]
        count += th.ones_like(mse) * batch_size

        if max_activations is not None:
            for i in range(1, n_buckets+1):
                mask = ((latent_activations > bucket_edges[:, i-1]) & (latent_activations <= bucket_edges[:, i])).T.unsqueeze(-1)
                assert mask.shape == (len(latent_indices), batch_size, 1)
                residual_masked = residual*mask
                residual_before_masked = residual_before*mask
                mse_bucket_result = residual_masked.pow(2).sum(dim=1).mean(dim=-1)
                assert mse_bucket_result.shape == (len(latent_indices),)
                mse_buckets[i-1] += mse_bucket_result
                mse_before_buckets[i-1] += residual_before_masked.pow(2).sum(dim=1).mean(dim=-1)
                mse_count[i-1] += mask.sum(dim=1).squeeze(-1)

        # Clean up GPU memory
        del latent_activations, target, reconstructions, residual, residual_before, activation_before
        if device == "cuda":
            th.cuda.empty_cache()

    output = {}
    # Compute final metrics
    if compute_fve:
        frac_explained = explained_variance / count
        frac_explained_before = before_explained_variance / count
        output["frac_variance_explained"] = frac_explained.cpu().numpy()
        output["frac_variance_explained_no_scaler"] = frac_explained_before.cpu().numpy()
        output["count"] = count.cpu().numpy()

    if max_activations is not None:
        output["mse_buckets"] = mse_buckets.cpu().numpy() / mse_count.cpu().numpy()
        output["mse_before_buckets"] = mse_before_buckets.cpu().numpy() / mse_count.cpu().numpy()
        output["mse_count"] = mse_count.cpu().numpy()
        output["bucket_edges"] = bucket_edges.cpu().numpy()
    output["mse"] = mse.cpu().numpy() / count.cpu().numpy()
    output["mse_before"] = mse_before.cpu().numpy() / count.cpu().numpy()

    return output


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

def load_zero_vector(
    batch,
    **kwargs,
):
    return th.zeros_like(batch[:, 0, :])

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
    parser.add_argument("--max-activations", type=Path, default=None)
    parser.add_argument("--n-buckets", type=int, default=3)
    parser.add_argument("--add-noise-threshold", type=float, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64", "bfloat16"],
    )
    # Add compute_scalers.py specific arguments for betas path construction
    parser.add_argument(
        "--results-dir",
        type=Path,
        default="/workspace/data/results/closed_form_scalars",
    )
    parser.add_argument("--threshold-active-latents", type=float, default=None)
    parser.add_argument("--chat-error", action="store_true")
    parser.add_argument("--chat-reconstruction", action="store_true")
    parser.add_argument("--base-error", action="store_true")
    parser.add_argument("--base-reconstruction", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1000000)
    parser.add_argument("--dataset-split", type=str, default="validation")
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
    if args.base_error:
        computations.append(
            (
                "base_error",
                load_base_reconstruction,
                load_base_activation,
            )
        )
    if args.base_reconstruction:
        computations.append(
            ("base_reconstruction", load_zero_vector, load_base_reconstruction)
        )
    if args.chat_reconstruction:
        computations.append(
            ("it_reconstruction", load_zero_vector, load_chat_reconstruction)
        )
    if args.chat_error:
        computations.append(("it_error", load_chat_reconstruction, load_chat_activation))

    if len(computations) == 0:
        logger.info("No computations selected, running all")
        computations = [
            ("base_reconstruction", load_zero_vector, load_base_reconstruction),
            (
                "base_error",
                load_base_reconstruction,
                load_base_activation,
            ),
            ("it_reconstruction", load_zero_vector, load_chat_reconstruction),
            ("it_error", load_chat_reconstruction, load_chat_activation),
        ]

    if args.max_activations is not None:
        max_activations = th.load(args.max_activations)
    else:
        max_activations = None

    # Compute stats for each computation type
    for name, loader_fn, target_fn in computations:
        logger.info(f"Computing stats for {name}")
        betas, betas_name = load_betas(args, name, results_dir)
        metrics = compute_stats(
            betas=betas,
            latent_vectors=latent_vectors,
            latent_indices=chat_only_indices,
            dataloader=dataloader,
            crosscoder=cc,
            activation_postprocessing_fn=loader_fn,
            target_fn=target_fn,
            device=device,
            dtype=dtype,
            max_activations=max_activations,
            n_buckets=args.n_buckets,
            add_noise_threshold=args.add_noise_threshold,
        )
        # Save results
        output_path = results_dir / f"stats_{betas_name}_.pt"
        th.save(metrics, output_path)
        logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
