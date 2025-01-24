from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import torch as th
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import json
import argparse
from huggingface_hub import hf_hub_download

from dictionary_learning import CrossCoder


def calculate_projection(data, features):
    return data @ features.T / (features @ features.T).diag()


def compute_feature_metrics(
    dataloader,
    features,
    crosscoder,
    n_batch_size=1000,
    d_batch_size=1000,
    device="cpu",
    n_buckets=10,
):
    """
    Compute variance explained and correlation metrics with batching along both N and D dimensions
    """
    results = {}
    N = len(dataloader.dataset)
    D = features.shape[0]
    th.cuda.empty_cache()

    with th.no_grad():
        crosscoder.to(device)

        # Initialize storage for projections
        projections_base = th.zeros((N, D), device="cpu")
        projections_it = th.zeros((N, D), device="cpu")
        scalars = th.zeros((N, D), device="cpu")

        # First pass: compute all projections and store them
        for batch_idx, batch_data in enumerate(
            tqdm(dataloader, desc="Computing feature activations and projections")
        ):
            batch_data = batch_data.to(device)
            batch_base = batch_data[:, 0]  # First column is base model data
            batch_it = batch_data[:, 1]  # Second column is instruction model data

            n_start = batch_idx * dataloader.batch_size
            n_end = n_start + batch_base.size(0)

            batch_features = crosscoder.encode(batch_data)
            scalars[n_start:n_end, :] = batch_features.cpu()

            for d_start in range(0, D, d_batch_size):
                d_end = min(d_start + d_batch_size, D)
                feat_batch = features[d_start:d_end].to(device)

                # Compute and store projections for this batch
                data_projection_base = calculate_projection(batch_base, feat_batch)
                projections_base[n_start:n_end, d_start:d_end] = (
                    data_projection_base.cpu()
                )

                data_projection_it = calculate_projection(batch_it, feat_batch)
                projections_it[n_start:n_end, d_start:d_end] = data_projection_it.cpu()
                del data_projection_it, data_projection_base, feat_batch
                th.cuda.empty_cache()

        # Compute min and max of scalars for each feature
        scalar_min = th.min(scalars, dim=0)[0]  # Shape: (D,)
        scalar_max = th.max(scalars, dim=0)[0]  # Shape: (D,)
        print("Scalar min", scalar_min.shape)
        print("Scalar max", scalar_max.shape)

        # Compute overall correlations for each feature
        correlations = th.stack(
            [
                th.corrcoef(
                    th.stack(
                        [projections_base[:, i].to(device), scalars[:, i].to(device)]
                    )
                )[0, 1]
                for i in trange(
                    D, desc="Computing base correlations", leave=False, position=1
                )
            ]
        )
        correlations_it = th.stack(
            [
                th.corrcoef(
                    th.stack(
                        [projections_it[:, i].to(device), scalars[:, i].to(device)]
                    )
                )[0, 1]
                for i in trange(
                    D, desc="Computing it correlations", leave=False, position=1
                )
            ]
        )
        results["correlations"] = {
            "base": correlations.cpu(),
            "it": correlations_it.cpu(),
        }
        # Compute correlations for each feature in 100 buckets
        bucket_correlations_base = th.zeros((D, n_buckets), device=device)
        bucket_correlations_it = th.zeros((D, n_buckets), device=device)
        bucket_count = th.zeros((D, n_buckets), device=device)

        for i in trange(D, desc="Computing bucket correlations"):
            # Get data for current feature
            feat_proj_base = projections_base[:, i].to(device)
            feat_proj_it = projections_it[:, i].to(device)
            feat_scalar = scalars[:, i].to(device)

            # Create buckets based on scalar values
            bucket_edges = th.linspace(scalar_min[i], scalar_max[i], n_buckets + 1)

            for b in range(n_buckets):
                # Get mask for current bucket
                bucket_mask = (feat_scalar >= bucket_edges[b]) & (
                    feat_scalar < bucket_edges[b + 1]
                )
                bucket_count[i, b] = bucket_mask.sum()
                if bucket_mask.sum() > 1:  # Need at least 2 points for correlation
                    bucket_correlations_base[i, b] = th.corrcoef(
                        th.stack(
                            [feat_proj_base[bucket_mask], feat_scalar[bucket_mask]]
                        )
                    )[0, 1]
                    bucket_correlations_it[i, b] = th.corrcoef(
                        th.stack([feat_proj_it[bucket_mask], feat_scalar[bucket_mask]])
                    )[0, 1]
                else:
                    bucket_correlations_base[i, b] = float("nan")
                    bucket_correlations_it[i, b] = float("nan")

        results["bucket_correlations"] = {
            "base": bucket_correlations_base.cpu(),
            "it": bucket_correlations_it.cpu(),
        }
        results["bucket_count"] = bucket_count.cpu()

        th.save(results, "results/results_500000_bucket_count.pt")
        # Fraction of projection values
        projection_fraction = projections_base / projections_it

        normalized_projection_fraction = (
            (projections_base - projections_it)
            / th.max(projections_base, projections_it)
            + 1
        ) * 0.5

        # Calculate mean fraction of projection values over buckets
        mean_projection_fraction = th.nanmean(projection_fraction, dim=0)
        results["relproj"] = mean_projection_fraction.cpu()
        results["normalized_relproj"] = th.nanmean(
            normalized_projection_fraction, dim=0
        ).cpu()

        bucket_fraction = th.zeros((D, n_buckets), device=device)
        bucket_normalized_fraction = th.zeros((D, n_buckets), device=device)
        for i in trange(D, desc="Computing bucket fractions", leave=False):
            bucket_edges = th.linspace(scalar_min[i], scalar_max[i], n_buckets + 1)
            for b in range(n_buckets):
                bucket_mask = (scalars[:, i] >= bucket_edges[b]) & (
                    scalars[:, i] < bucket_edges[b + 1]
                )
                bucket_fraction[i, b] = th.nanmean(projection_fraction[bucket_mask, i])
                bucket_normalized_fraction[i, b] = th.nanmean(
                    normalized_projection_fraction[bucket_mask, i]
                )
                del bucket_mask
        results["bucket_relproj"] = bucket_fraction.cpu()
        results["bucket_normalized_relproj"] = bucket_normalized_fraction.cpu()

    return results

def normalized_relproj(proj_a, proj_b):
    denom = th.max(proj_a, proj_b) - th.min(proj_a, proj_b)
    return th.where(denom == 0, 
                   th.tensor(float('nan'), device=proj_a.device),
                   ((proj_a - proj_b) / denom + 1) * 0.5)

def compute_feature_metrics_optimized(
    dataloader,
    features,
    crosscoder,
    d_batch_size=1000,
    n_buckets=10,
    device="cpu",
    compute_correlations=True,
):
    """
    Memory-optimized version that computes correlation metrics using online statistics in a single pass
    """
    results = {}
    N = len(dataloader.dataset)
    D = features.shape[0]
    th.cuda.empty_cache()

    if compute_correlations:
        # Initialize online statistics for overall correlation
        count = th.zeros(D, device=device)
        mean_proj_base = th.zeros(D, device=device)
        mean_proj_it = th.zeros(D, device=device)
        mean_scalar = th.zeros(D, device=device)
        M2_proj_base = th.zeros(D, device=device)
        M2_proj_it = th.zeros(D, device=device)
        M2_scalar = th.zeros(D, device=device)
        covar_base = th.zeros(D, device=device)
        covar_it = th.zeros(D, device=device)

    # Running sums and counts for relative projections
    # These are computed over the entire dataset and later used to compute mean relative projections
    sum_rel_proj = th.zeros(D, device=device)
    count_rel_proj = th.zeros(D, device=device)
    sum_normalized_rel_proj = th.zeros(D, device=device)
    count_normalized_rel_proj = th.zeros(D, device=device)
    # Process bucket statistics
    bucket_count = th.zeros((D, n_buckets), device=device)
    bucket_rel_proj = th.zeros((D, n_buckets), device=device)
    bucket_normalized_rel_proj = th.zeros((D, n_buckets), device=device)
    bucket_rel_proj_count = th.zeros((D, n_buckets), device=device)
    bucket_normalized_rel_proj_count = th.zeros((D, n_buckets), device=device)

    # Track min/max for dynamic bucket adjustment
    scalar_min = th.full((D,), float("inf"), device=device)
    scalar_max = th.full((D,), float("-inf"), device=device)
    proj_base_min = th.full((D,), float("inf"), device=device)
    proj_base_max = th.full((D,), float("-inf"), device=device)
    proj_it_min = th.full((D,), float("inf"), device=device)
    proj_it_max = th.full((D,), float("-inf"), device=device)
    with th.no_grad():
        crosscoder.to(device)

        # First pass through the data to compute scalar statistics
        for batch_data in tqdm(dataloader, desc="Computing scalar statistics"):
            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)
            batch_base = batch_data[:, 0]
            batch_it = batch_data[:, 1]
            batch_features = crosscoder.encode(batch_data)

            # Update global min/max for bucket edges
            scalar_min = th.minimum(scalar_min, batch_features.min(dim=0)[0])
            scalar_max = th.maximum(scalar_max, batch_features.max(dim=0)[0])

            proj_base = calculate_projection(batch_base, features)
            proj_it = calculate_projection(batch_it, features)

            proj_base_min = th.minimum(proj_base_min, proj_base.min(dim=0)[0])
            proj_base_max = th.maximum(proj_base_max, proj_base.max(dim=0)[0])
            proj_it_min = th.minimum(proj_it_min, proj_it.min(dim=0)[0])
            proj_it_max = th.maximum(proj_it_max, proj_it.max(dim=0)[0])

        bucket_edges = th.stack([
            th.linspace(scalar_min[i], scalar_max[i], n_buckets + 1) for i in range(D)
        ]).to(device)
        assert bucket_edges.shape == (D, n_buckets + 1)
        print("Bucket edges", bucket_edges.shape)

        # Second pass through the data to compute projection statistics
        for batch_data in tqdm(dataloader, desc="Computing projection statistics"):
            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)
            batch_base = batch_data[:, 0]
            batch_it = batch_data[:, 1]
            batch_features = crosscoder.encode(batch_data)

            # Loop over latents in d_batch_size chunks
            for d_start in range(0, D, d_batch_size):
                d_end = min(d_start + d_batch_size, D)
                d_batch_size_current = d_end - d_start
                feat_batch = features[d_start:d_end].to(device)

                proj_base = calculate_projection(batch_base, feat_batch)
                assert proj_base.shape == (batch_size, d_batch_size_current)
                proj_it = calculate_projection(batch_it, feat_batch)
                assert proj_it.shape == (batch_size, d_batch_size_current)
                scalar_batch = batch_features[:, d_start:d_end]
                assert scalar_batch.shape == (batch_size, d_batch_size_current)

                # Compute relative projections (non-zero denominator)
                rel_proj_valid_mask = proj_it != 0
                rel_proj = th.where(
                    rel_proj_valid_mask,
                    proj_base / proj_it,
                    th.tensor(float("nan"), device=device),
                )
                # Compute normalized relative projections (non-zero denominator)
                normalized_rel_proj = normalized_relproj(proj_base, proj_it)
                
                # Update relative projection statistics
                sum_rel_proj[d_start:d_end] += th.nansum(rel_proj, dim=0)
                count_rel_proj[d_start:d_end] += rel_proj_valid_mask.sum(dim=0)
                sum_normalized_rel_proj[d_start:d_end] += th.nansum(normalized_rel_proj, dim=0)
                count_normalized_rel_proj[d_start:d_end] += (~th.isnan(normalized_rel_proj)).sum(dim=0)

                for b in range(n_buckets):
                    bucket_mask = (
                        batch_features[:, d_start:d_end] >= bucket_edges[d_start:d_end, b]
                    ) & (
                        batch_features[:, d_start:d_end] < bucket_edges[d_start:d_end, b + 1]
                    )
                    bucket_count[d_start:d_end, b] += bucket_mask.sum(dim=0)

                    # Compute bucket relative projections
                    batch_bucket_rel_proj_mask = proj_it[bucket_mask] != 0
                    batch_bucket_rel_proj = th.where(
                        batch_bucket_rel_proj_mask,
                        proj_base[bucket_mask] / proj_it[bucket_mask],
                        th.tensor(float("nan"), device=device),
                    )
                    bucket_rel_proj[d_start:d_end, b] += th.nansum(batch_bucket_rel_proj)
                    bucket_rel_proj_count[
                        d_start:d_end, b
                    ] += batch_bucket_rel_proj_mask.sum(dim=0)

                    # Compute normalized bucket relative projections
                    batch_normalized_rel_proj = normalized_relproj(proj_base[bucket_mask], proj_it[bucket_mask])
                    bucket_normalized_rel_proj[d_start:d_end, b] += th.nansum(
                        batch_normalized_rel_proj
                    )
                    bucket_normalized_rel_proj_count[
                        d_start:d_end, b
                    ] += (~th.isnan(batch_normalized_rel_proj)).sum()

    
                    # Update overall statistics
                    if compute_correlations:
                        new_count = count[d_start:d_end] + proj_base.size(0)
                        delta_proj_base = proj_base - mean_proj_base[d_start:d_end]
                        delta_proj_it = proj_it - mean_proj_it[d_start:d_end]
                        delta_scalar = scalar_batch - mean_scalar[d_start:d_end]

                        mean_proj_base[d_start:d_end] += delta_proj_base.sum(dim=0) / new_count
                        mean_proj_it[d_start:d_end] += delta_proj_it.sum(dim=0) / new_count
                        mean_scalar[d_start:d_end] += delta_scalar.sum(dim=0) / new_count

                        delta2_proj_base = proj_base - mean_proj_base[d_start:d_end]
                        delta2_proj_it = proj_it - mean_proj_it[d_start:d_end]
                        delta2_scalar = scalar_batch - mean_scalar[d_start:d_end]

                        M2_proj_base[d_start:d_end] += (
                            delta_proj_base * delta2_proj_base
                        ).sum(dim=0)
                        M2_proj_it[d_start:d_end] += (delta_proj_it * delta2_proj_it).sum(dim=0)
                        M2_scalar[d_start:d_end] += (delta_scalar * delta2_scalar).sum(dim=0)
                        covar_base[d_start:d_end] += (delta_proj_base * delta2_scalar).sum(dim=0)
                        covar_it[d_start:d_end] += (delta_proj_it * delta2_scalar).sum(dim=0)
                        count[d_start:d_end] = new_count

                del feat_batch, proj_base, proj_it, scalar_batch
                th.cuda.empty_cache()

        if compute_correlations:
            # Compute final correlations
            valid_count = count > 1
            correlations_base = th.full((D,), float("nan"), device=device)
            correlations_it = th.full((D,), float("nan"), device=device)

            var_proj_base = M2_proj_base[valid_count] / (count[valid_count] - 1)
            var_proj_it = M2_proj_it[valid_count] / (count[valid_count] - 1)
            var_scalar = M2_scalar[valid_count] / (count[valid_count] - 1)

            correlations_base[valid_count] = (
                covar_base[valid_count]
                / th.sqrt(var_proj_base * var_scalar)
                / (count[valid_count] - 1)
            )
            correlations_it[valid_count] = (
                covar_it[valid_count]
                / th.sqrt(var_proj_it * var_scalar)
                / (count[valid_count] - 1)
            )

            results["correlations"] = {
                "base": correlations_base.cpu(),
                "it": correlations_it.cpu(),
            }

        # Compute relative projections
        mean_rel_proj = th.where(
            count_rel_proj > 0,
            sum_rel_proj / count_rel_proj,
            th.tensor(float("nan"), device=device),
        )
        results["relproj"] = mean_rel_proj.cpu()
        mean_normalized_rel_proj = th.where(
            count_rel_proj > 0,
            sum_normalized_rel_proj / count_rel_proj,
            th.tensor(float("nan"), device=device),
        )
        results["normalized_relproj"] = mean_normalized_rel_proj.cpu()

        bucket_correlations_base = th.full((D, n_buckets), float("nan"), device=device)
        bucket_correlations_it = th.full((D, n_buckets), float("nan"), device=device)

        for i in range(D):
            for b in range(n_buckets):
                bucket_rel_proj[i, b] /= bucket_rel_proj_count[i, b]
                bucket_normalized_rel_proj[i, b] /= bucket_normalized_rel_proj_count[
                    i, b
                ]

        if compute_correlations:
            results["bucket_correlations"] = {
                "base": bucket_correlations_base.cpu(),
                "it": bucket_correlations_it.cpu(),
            }
        results["bucket_count"] = bucket_count.cpu()
        results["bucket_relproj"] = bucket_rel_proj.cpu()
        results["bucket_normalized_relproj"] = bucket_normalized_rel_proj.cpu()

    return results


def main(
    n,
    batch_size,
    d_batch_size,
    n_buckets,
    activation_store_dir,
    output_dir,
    dataset_split,
    compute_correlations,
):
    print(th.cuda.is_available())
    repo_id = "Butanium/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04"

    df_path = hf_hub_download(
        repo_id=repo_id, filename="feature_df.csv", repo_type="dataset"
    )
    features_df = pd.read_csv(df_path, index_col=0)
    features_df

    crosscoder = CrossCoder.from_pretrained(
        "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04", from_hub=True
    )

    BASE_MODEL = "gemma-2-2b"
    INSTRUCT_MODEL = "gemma-2-2b-it"
    activation_store_dir = Path(activation_store_dir)

    base_model_dir = activation_store_dir / BASE_MODEL
    instruct_model_dir = activation_store_dir / INSTRUCT_MODEL

    base_model_fineweb = base_model_dir / "fineweb-1m-sample" / dataset_split
    base_model_lmsys_chat = (
        base_model_dir / "lmsys-chat-1m-gemma-formatted" / dataset_split
    )
    instruct_model_fineweb = instruct_model_dir / "fineweb-1m-sample" / dataset_split
    instruct_model_lmsys_chat = (
        instruct_model_dir / "lmsys-chat-1m-gemma-formatted" / dataset_split
    )

    submodule_name = f"layer_13_out"

    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_chat_cache = PairedActivationCache(
        base_model_lmsys_chat / submodule_name,
        instruct_model_lmsys_chat / submodule_name,
    )

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    crosscoder.decoder.weight.shape
    it_decoder = crosscoder.decoder.weight[1, :, :].clone()
    base_decoder = crosscoder.decoder.weight[0, :, :].clone()

    n_per_dataset = n // 2
    test_idx = th.cat(
        [th.arange(n_per_dataset), th.arange(n_per_dataset) + len(fineweb_cache)]
    )
    # Get the text indices from the validation dataset
    dataset = th.utils.data.Subset(dataset, test_idx)
    print("Number of activations: ", len(dataset))

    # Create DataLoader for batch processing
    dataloader = th.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10
    )

    # results = compute_feature_metrics(dataloader, it_decoder, crosscoder, n_batch_size=batch_size, d_batch_size=d_batch_size, device="cuda:0")
    results = compute_feature_metrics_optimized(
        dataloader,
        it_decoder,
        crosscoder,
        d_batch_size=d_batch_size,
        n_buckets=n_buckets,
        device="cuda:0",
        compute_correlations=compute_correlations,
    )
    save_path = Path(output_dir)
    # make paths
    save_path.mkdir(parents=True, exist_ok=True)
    file = save_path / f"relproj_results_{n}.pt"
    th.save(results, file)


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(
        description="Compute feature metrics on model activations"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=100_000,
        help="Number of activations to compute metrics on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch size over N dimension",
    )
    parser.add_argument(
        "--d_batch_size",
        type=int,
        default=73728,
        help="Batch size over D dimension",
    )
    parser.add_argument(
        "--n_buckets",
        type=int,
        default=10,
        help="Number of buckets to use for bucketed correlations (default: 10)",
    )
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--compute-correlations", action="store_true")
    
    args = parser.parse_args()

    main(
        args.n,
        args.batch_size,
        args.d_batch_size,
        args.n_buckets,
        args.activation_store_dir,
        args.output_dir,
        args.dataset_split,
        compute_correlations=args.compute_correlations,
    )
