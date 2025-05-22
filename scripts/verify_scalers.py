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
from tools.utils import load_activation_dataset, load_crosscoder
from tools.compute_utils import BucketedStats, RunningMeanStd
from tools.configs import DATA_ROOT

th.set_grad_enabled(False)
th.set_float32_matmul_precision("highest")

from tools.latent_scaler.closed_form import remove_latents
from tools.utils import load_connor_crosscoder

# Define processing functions (reusing from compute_scalers.py)
if __name__ == "__main__":
    from compute_scalers import (
        load_base_reconstruction,
        load_base_error,
        load_base_activation,
        load_chat_reconstruction,
        load_chat_error,
        load_chat_activation,
    )
else:
    # Deal with relative imports from other scripts
    from .compute_scalers import (
        load_base_reconstruction,
        load_base_error,
        load_base_activation,
        load_chat_reconstruction,
        load_chat_error,
        load_chat_activation,
    )


def get_bucket_edges(n_buckets, add_noise_threshold=0.05):
    step_size = 1 / n_buckets
    steps = th.arange(n_buckets + 1) * step_size
    if add_noise_threshold is not None:
        steps = th.cat([th.tensor([0, add_noise_threshold]), steps[1:]])
    return steps


def squared_error(residual):
    # residual: [n_latents, batch_size, dim_model]
    # return: [n_latents, batch_size]
    n_latents, batch_size, dim_model = residual.shape
    out = residual.pow(2).sum(dim=2)
    assert out.shape == (n_latents, batch_size)
    return out


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
    n_buckets: int = 3,
    add_noise_threshold: float = None,
    compute_individual_errors: bool = False,
    num_samples: int = None,
    compute_latent_activations: bool = False,
) -> dict:
    """
    Compute statistics for each scaler.
    """
    dim_model = crosscoder.activation_dim
    betas = betas.to(device)
    assert betas.shape == (len(latent_indices),)
    mse = th.zeros(len(latent_indices), device=device, dtype=th.float64)
    mse_std = RunningMeanStd()
    mse_before = th.zeros(len(latent_indices), device=device, dtype=th.float64)
    mse_before_std = RunningMeanStd()
    count = th.zeros(len(latent_indices), device=device, dtype=dtype)

    if max_activations is not None:
        if add_noise_threshold is not None:
            n_buckets += 1
        assert max_activations.shape == (crosscoder.dict_size,)
        max_activations = max_activations[latent_indices].to(device)
        assert len(max_activations) == len(latent_indices)
        bucket_edges = get_bucket_edges(n_buckets, add_noise_threshold)

        mse_buckets = BucketedStats(
            num_latents=len(latent_indices),
            max_activations=max_activations,
            device=device,
            bucket_edges=bucket_edges,
        )
        mse_before_buckets = BucketedStats(
            num_latents=len(latent_indices),
            max_activations=max_activations,
            device=device,
            bucket_edges=bucket_edges,
        )

    if compute_individual_errors:
        assert (
            num_samples is not None
        ), "num_samples must be provided if compute_individual_errors is True"
        mse_individual = th.zeros(
            len(latent_indices), num_samples, device=device, dtype=th.float32
        )
        mse_before_individual = th.zeros(
            len(latent_indices), num_samples, device=device, dtype=th.float32
        )

    if compute_latent_activations:
        latent_activations_individual = th.zeros(
            len(latent_indices), num_samples, device=device, dtype=th.float32
        )

    start_idx = 0
    for batch in tqdm(dataloader, desc="Computing scaler stats"):
        batch_size = batch.shape[0]
        end_idx = start_idx + batch_size
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

        if compute_latent_activations:
            latent_activations_individual[:, start_idx:end_idx] = (
                latent_activations.permute(1, 0)
            )

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

        # residual is [n_latents, batch_size, dim_model]
        # squared_error_residual is [n_latents, batch_size]
        squared_error_residual = squared_error(residual)
        squared_error_residual_before = squared_error(residual_before)
        assert squared_error_residual.shape == (len(latent_indices), batch_size)
        assert squared_error_residual_before.shape == (len(latent_indices), batch_size)

        mse += squared_error_residual.sum(dim=-1).double()  # [n_latents]
        mse_before += squared_error_residual_before.sum(dim=-1).double()  # [n_latents]
        mse_std.update(squared_error_residual.permute(1, 0))
        mse_before_std.update(squared_error_residual_before.permute(1, 0))
        count += th.ones_like(mse) * batch_size

        if max_activations is not None:
            mse_buckets.update(latent_activations, squared_error_residual.permute(1, 0))
            mse_before_buckets.update(
                latent_activations, squared_error_residual_before.permute(1, 0)
            )

        if compute_individual_errors:
            mse_individual[:, start_idx:end_idx] = squared_error_residual
            mse_before_individual[:, start_idx:end_idx] = squared_error_residual_before

        # Clean up GPU memory
        del (
            latent_activations,
            target,
            reconstructions,
            residual,
            residual_before,
            activation_before,
        )
        if device == "cuda":
            th.cuda.empty_cache()

        start_idx += batch_size

    output = {}

    # Compute final metrics
    if max_activations is not None:
        output["mse_buckets"] = mse_buckets.finish().to_dict()
        output["mse_before_buckets"] = mse_before_buckets.finish().to_dict()
        output["bucket_edges"] = bucket_edges.cpu().numpy()

    if compute_individual_errors:
        output["mse_individual"] = mse_individual.cpu().numpy()
        output["mse_before_individual"] = mse_before_individual.cpu().numpy()

    if compute_latent_activations:
        output["latent_activations_individual"] = (
            latent_activations_individual.cpu().numpy()
        )

    mse_mean, mse_std, mse_count = mse_std.compute()
    output["mse_mean"] = mse_mean.cpu().numpy()
    output["mse_std"] = mse_std.cpu().numpy()
    output["mse_count"] = mse_count

    mse_before_mean, mse_before_std, mse_before_count = mse_before_std.compute()
    output["mse_before_mean"] = mse_before_mean.cpu().numpy()
    output["mse_before_std"] = mse_before_std.cpu().numpy()
    output["mse_before_count"] = mse_before_count
    return output


def load_betas(args, computation, results_dir, train_num_samples, train_n_offset):
    if args.from_sgd:
        betas = th.load(args.sgd_betas_path, weights_only=True)["scaler"]
        name = f"{computation}_sgd_{args.sgd_betas_path.stem}_N{train_num_samples}_n_offset{train_n_offset}"
        return betas, name

    name = computation
    name += f"_N{train_num_samples}_n_offset{train_n_offset}"

    if args.threshold_active_latents is not None:
        name += f"_jumprelu{args.threshold_active_latents}"
    if args.name:
        name += f"_{args.name}"

    betas_path = results_dir / f"betas_{name}.pt"
    logger.info(f"Processing betas from: {betas_path}")

    if not betas_path.exists():
        raise FileNotFoundError(f"Betas file not found: {betas_path}")

    # Load betas and continue with existing logic
    logger.info(f"Loading betas from {betas_path}")
    try:
        betas = th.load(betas_path, weights_only=True)
    except Exception as e:
        # for legacy filenames
        betas = th.load(betas_path.replace("chat", "it"), weights_only=True)
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
    parser.add_argument(
        "--latent-indices-path",
        type=Path,
    )
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-activations", type=Path, default=None)
    parser.add_argument("--n-buckets", type=int, default=3)
    parser.add_argument("--add-noise-threshold", type=float, default=None)
    parser.add_argument("--from-sgd", action="store_true", help="Load from SGD scaler")
    parser.add_argument("--sgd-betas-path", type=Path, default=None)
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
        default=DATA_ROOT / "results/closed_form_scalars",
    )
    parser.add_argument("--threshold-active-latents", type=float, default=None)
    parser.add_argument("--chat-error", action="store_true")
    parser.add_argument("--chat-reconstruction", action="store_true")
    parser.add_argument("--base-error", action="store_true")
    parser.add_argument("--base-reconstruction", action="store_true")
    parser.add_argument("--base-activation", action="store_true")
    parser.add_argument("--chat-activation", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("-N", "--num-samples", type=int, default=5_000_000)
    parser.add_argument("--train-num-samples", type=int, default=50_000_000)
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument(
        "--special-results-dir",
        type=str,
        default="",
        help="Addon to the results directory. Results will be loaded and saved in results_dir/SRD/model_name/",
    )
    parser.add_argument("--n-offset", type=int, default=0)
    parser.add_argument("--train-n-offset", type=int, default=None)
    parser.add_argument("-CIE", "--compute-individual-errors", action="store_true")
    parser.add_argument("-CLA", "--compute-latent-activations", action="store_true")
    parser.add_argument(
        "--betas-subset-path",
        type=Path,
        default=None,
        help="Path to a file containing indices of latents to include in the verification (Indices in the betas file, not latent indices)",
    )
    parser.add_argument(
        "--lmsys-subfolder",
        type=str,
        default=None,
        help="Subfolder for the LMSYS dataset",
    )
    args = parser.parse_args()

    if args.train_num_samples is None:
        train_num_samples = args.num_samples
    else:
        train_num_samples = args.train_num_samples

    if args.train_n_offset is None:
        train_n_offset = args.n_offset
    else:
        train_n_offset = args.train_n_offset

    dtype_map = {"float32": th.float32, "float64": th.float64, "bfloat16": th.bfloat16}
    dtype = dtype_map[args.dtype]
    logger.info(f"Using dtype: {dtype}")

    # Setup device and dtype
    if args.device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = th.device(args.device)

    # Load crosscoder
    cc = load_crosscoder(args.crosscoder_path)
    cc = cc.to(device).to(dtype)

    # If crosscoder is a local path, replace with only the directory name (e.g. /path/to/crosscoder/model_final.pt -> crosscoder)
    if Path(args.crosscoder_path).exists():
        args.crosscoder_path = Path(args.crosscoder_path).parent.name

    print(f"Using crosscoder name: {args.crosscoder_path}")

    # Construct betas path following compute_scalers.py logic
    if args.special_results_dir:
        results_dir = (
            args.results_dir
            / args.special_results_dir
            / args.crosscoder_path.replace("/", "_")
        )
    else:
        results_dir = args.results_dir / args.crosscoder_path.replace("/", "_")
    results_dir = results_dir / Path(args.latent_indices_path).name.split(".")[0]

    # Load validation dataset
    activation_store_dir = Path(args.activation_store_dir)

    base_model_stub = args.base_model.split("/")[-1]
    chat_model_stub = args.chat_model.split("/")[-1]
    fineweb_cache, lmsys_cache = load_activation_dataset(
        activation_store_dir,
        base_model=base_model_stub,
        instruct_model=chat_model_stub,
        layer=args.layer,
        split=args.dataset_split,
        lmsys_subfolder=args.lmsys_subfolder,
    )
    num_samples_per_dataset = args.num_samples // 2
    dataset = th.utils.data.ConcatDataset(
        [
            th.utils.data.Subset(
                fineweb_cache,
                th.arange(
                    args.n_offset * num_samples_per_dataset,
                    (args.n_offset + 1) * num_samples_per_dataset,
                ),
            ),
            th.utils.data.Subset(
                lmsys_cache,
                th.arange(
                    args.n_offset * num_samples_per_dataset,
                    (args.n_offset + 1) * num_samples_per_dataset,
                ),
            ),
        ]
    )

    logger.info(f"Number of activations: {len(dataset)}")
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Load betas and chat indices
    latent_indices = th.load(args.latent_indices_path, weights_only=True)

    # Get decoder weights
    it_decoder = cc.decoder.weight[1, :, :].clone().to(dtype)
    base_decoder = cc.decoder.weight[0, :, :].clone().to(dtype)
    latent_vectors = it_decoder[latent_indices].clone()

    # Determine which computation to verify
    computations = []
    if args.base_error:
        computations.append(
            (
                "base_error",
                load_zero_vector,
                partial(load_base_error, base_decoder=base_decoder),
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
        computations.append(("it_error", load_zero_vector, load_chat_error))
    if args.chat_activation:
        computations.append(("it_activation", load_zero_vector, load_chat_activation))
    if args.base_activation:
        computations.append(("base_activation", load_zero_vector, load_base_activation))

    if len(computations) == 0:
        logger.info("No computations selected, running all")
        computations = [
            ("base_reconstruction", load_zero_vector, load_base_reconstruction),
            (
                "base_error",
                load_base_reconstruction,
                partial(load_base_error, base_decoder=base_decoder),
            ),
            ("it_reconstruction", load_zero_vector, load_chat_reconstruction),
            ("it_error", load_zero_vector, load_chat_error),
            ("it_activation", load_zero_vector, load_chat_activation),
            ("base_activation", load_zero_vector, load_base_activation),
        ]

    if args.max_activations is not None:
        max_activations = th.load(args.max_activations, weights_only=True)
    else:
        max_activations = None

    # Compute stats for each computation type
    for name, loader_fn, target_fn in computations:
        logger.info(f"Computing stats for {name}")
        betas, betas_name = load_betas(
            args, name, results_dir, train_num_samples, train_n_offset
        )
        if args.betas_subset_path is not None:
            betas_subset = th.load(args.betas_subset_path)
            betas = betas[betas_subset]
            latent_vectors = latent_vectors[betas_subset]
            latent_indices = latent_indices[betas_subset]
            betas_name += f"_subset_{args.betas_subset_path.stem}"
        if args.train_num_samples is not None:
            betas_name += f"_EVAL_N{args.num_samples}_n_offset{args.n_offset}"
        metrics = compute_stats(
            betas=betas,
            latent_vectors=latent_vectors,
            latent_indices=latent_indices,
            dataloader=dataloader,
            crosscoder=cc,
            activation_postprocessing_fn=loader_fn,
            target_fn=target_fn,
            device=device,
            dtype=dtype,
            max_activations=max_activations,
            n_buckets=args.n_buckets,
            add_noise_threshold=args.add_noise_threshold,
            compute_individual_errors=args.compute_individual_errors,
            num_samples=len(dataset),
            compute_latent_activations=args.compute_latent_activations,
        )
        # Save results
        output_path = results_dir / f"stats_{betas_name}_{args.dataset_split}.pt"
        th.save(metrics, output_path)
        logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
