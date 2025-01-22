from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import torch as th
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import json
from dlabutils import model_path
import argparse

from dictionary_learning import CrossCoder


def calculate_projection(data, features):
    return data @ features.T / (features @ features.T).diag()

def compute_feature_metrics(dataloader, features, crosscoder, n_batch_size=1000, d_batch_size=1000, device='cpu', n_buckets=10):
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
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Computing feature activations and projections")):
            batch_data = batch_data.to(device)
            batch_base = batch_data[:, 0]  # First column is base model data
            batch_it = batch_data[:, 1]    # Second column is instruction model data
            

            n_start = batch_idx * dataloader.batch_size
            n_end = n_start + batch_base.size(0)
            
            batch_features = crosscoder.encode(batch_data)
            scalars[n_start:n_end, :] = batch_features.cpu()

            for d_start in range(0, D, d_batch_size):
                d_end = min(d_start + d_batch_size, D)
                feat_batch = features[d_start:d_end].to(device)
                
                # Compute and store projections for this batch
                data_projection_base = calculate_projection(batch_base, feat_batch)
                projections_base[n_start:n_end, d_start:d_end] = data_projection_base.cpu()

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
        correlations = th.stack([
            th.corrcoef(th.stack([projections_base[:, i].to(device), scalars[:, i].to(device)]))[0, 1]
            for i in trange(D, desc="Computing base correlations", leave=False, position=1)
        ])
        correlations_it = th.stack([
            th.corrcoef(th.stack([projections_it[:, i].to(device), scalars[:, i].to(device)]))[0, 1]
            for i in trange(D, desc="Computing it correlations", leave=False, position=1)
        ])
        results["correlations"] = {"base": correlations.cpu(), "it": correlations_it.cpu()}
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
                bucket_mask = (feat_scalar >= bucket_edges[b]) & (feat_scalar < bucket_edges[b+1])
                bucket_count[i, b] = bucket_mask.sum()
                if bucket_mask.sum() > 1:  # Need at least 2 points for correlation
                    bucket_correlations_base[i, b] = th.corrcoef(
                        th.stack([feat_proj_base[bucket_mask], feat_scalar[bucket_mask]])
                    )[0, 1]
                    bucket_correlations_it[i, b] = th.corrcoef(
                        th.stack([feat_proj_it[bucket_mask], feat_scalar[bucket_mask]])
                    )[0, 1]
                else:
                    bucket_correlations_base[i, b] = float('nan')
                    bucket_correlations_it[i, b] = float('nan')

        results["bucket_correlations"] = {"base": bucket_correlations_base.cpu(), "it": bucket_correlations_it.cpu()}
        results["bucket_count"] = bucket_count.cpu()

        th.save(results, "results/results_500000_bucket_count.pt")
        # Fraction of projection values
        projection_fraction = projections_base / projections_it
        
        # Calculate mean fraction of projection values over buckets
        mean_projection_fraction = th.nanmean(projection_fraction, dim=0)
        results["relproj"] = mean_projection_fraction.cpu()

        bucket_fraction = th.zeros((D, n_buckets), device=device)
        for i in trange(D, desc="Computing bucket fractions", leave=False):
            bucket_edges = th.linspace(scalar_min[i], scalar_max[i], n_buckets + 1)
            for b in range(n_buckets):
                bucket_mask = (scalars[:, i] >= bucket_edges[b]) & (scalars[:, i] < bucket_edges[b+1])
                bucket_fraction[i, b] = th.nanmean(projection_fraction[bucket_mask, i])
        results["bucket_relproj"] = bucket_fraction.cpu()

        # Compute residual variances using stored projections
        # residual_sq_sums = th.zeros(D, device=device)
        # residual_sq_sums_nonzero = th.zeros(D, device=device)
        # non_zero_count = th.zeros(D, device=device)
        # residual_sq_sums_zero = th.zeros(D, device=device)
        
        # for n_start in trange(0, N, n_batch_size, desc="Computing residuals"):
        #     n_end = min(n_start + n_batch_size, N)
        #     data_batch = data[n_start:n_end].to(device)
            
        #     for d_start in trange(0, D, d_batch_size, desc="D batches", leave=False):
        #         d_end = min(d_start + d_batch_size, D)
        #         feat_batch = features[d_start:d_end].to(device)
        #         proj_batch = projections[n_start:n_end, d_start:d_end].to(device)
        #         # Use stored projections to compute components and residuals
        #         data_components = proj_batch.unsqueeze(-1) * feat_batch.unsqueeze(0)
        #         data_without_component = data_batch.unsqueeze(1) - data_components
        #         residuals = data_without_component - data_without_component.mean(dim=(0, 2))[:, None]
        #         residual_sq_sums[d_start:d_end] += (residuals**2).sum(dim=(0, 2))

        #         non_zero_mask = scalars[n_start:n_end, d_start:d_end] != 0
        #         non_zero_mask = non_zero_mask.unsqueeze(-1).to(device)
        #         print(non_zero_mask.shape, residuals.shape)
        #         residual_sq_sums_nonzero[d_start:d_end] += (residuals**2 * non_zero_mask).sum(dim=(0, 2))
        #         residual_sq_sums_zero[d_start:d_end] += (residuals**2 * ~non_zero_mask).sum(dim=(0, 2))
        #         non_zero_count[d_start:d_end] += non_zero_mask.sum(dim=(0, 2))

        #         # Clean Up
        #         # Free memory from intermediate tensors
        #         del data_components
        #         del data_without_component 
        #         del residuals
        #         del feat_batch
        #         del proj_batch
        #         th.cuda.empty_cache()

        # residual_variances = residual_sq_sums / ((N * d) - 1)
        # residual_variances_nonzero = residual_sq_sums_nonzero / (non_zero_count - 1)
        # residual_variances_zero = residual_sq_sums_zero / (N - non_zero_count - 1)
        # data_variance = data.var()
        # var_explained = 1 - residual_variances / data_variance
        # var_explained_nonzero = 1 - residual_variances_nonzero / data_variance
        # var_explained_zero = 1 - residual_variances_zero / data_variance
        # var_explained = var_explained.cpu()
        # var_explained_nonzero = var_explained_nonzero.cpu()
        # var_explained_zero = var_explained_zero.cpu()

    return results


def main(n, batch_size, d_batch_size, n_buckets):
    print(th.cuda.is_available())
    features_df = pd.read_csv("results/feature_df.csv")
    features_df.head()

    crosscoder = CrossCoder.from_pretrained("Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04", from_hub=True)

    BASE_MODEL = "gemma-2-2b"
    INSTRUCT_MODEL = "gemma-2-2b-it"
    activation_store_dir = Path("activations") 

    base_model_dir = activation_store_dir / BASE_MODEL
    instruct_model_dir = activation_store_dir / INSTRUCT_MODEL

    base_model_fineweb = base_model_dir / "fineweb"
    base_model_lmsys_chat = base_model_dir / "lmsys_chat"
    instruct_model_fineweb = instruct_model_dir / "fineweb"
    instruct_model_lmsys_chat = instruct_model_dir / "lmsys_chat"

    submodule_name = f"layer_13_out"

    fineweb_cache = PairedActivationCache(base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name)
    lmsys_chat_cache = PairedActivationCache(base_model_lmsys_chat / submodule_name, instruct_model_lmsys_chat / submodule_name)

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    validation_size = 10**6
    train_dataset, validation_dataset = th.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size])

    crosscoder.decoder.weight.shape
    it_decoder = crosscoder.decoder.weight[1, :, :].clone()
    base_decoder = crosscoder.decoder.weight[0, :, :].clone()


    test_idx = th.randint(0, 1000000, (n,))
    # Get the text indices from the validation dataset
    dataset = th.utils.data.Subset(validation_dataset, test_idx)
    # data = th.stack([validation_dataset[i] for i in tqdm(test_idx, desc="Loading activations")])
    # base_activations = data[:, 0]
    # it_activations = data[:, 1]
    print("Number of activations: ", len(dataset))

    
    # Create DataLoader for batch processing
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=10
    )

    # # Process batches using DataLoader
    # for i, batch_data in enumerate(tqdm(data_loader, desc="Encoding features")):
    #     batch_data = batch_data.to('cuda:0')  # Move batch to GPU
    #     # base_activations = batch_data[:, 0]
    #     # it_activations = batch_data[:, 1]
        

    # print("Number of activations: ", len(dataloader.dataset))

    # results = compute_feature_metrics(dataloader, it_decoder, crosscoder, n_batch_size=batch_size, d_batch_size=d_batch_size, device="cuda:0")
    results = compute_feature_metrics_optimized(dataloader, it_decoder, crosscoder, d_batch_size=d_batch_size, n_buckets=n_buckets, device="cuda:0")
    save_path = Path("results")
    # make paths
    save_path.mkdir(parents=True, exist_ok=True)
    file = save_path / f"results_{n}.pt"
    th.save(results, file)


def compute_feature_metrics_optimized(dataloader, features, crosscoder, d_batch_size=1000, n_buckets=10, device='cpu'):
    """
    Memory-optimized version that computes correlation metrics using online statistics in a single pass
    """
    results = {}
    N = len(dataloader.dataset)
    D = features.shape[0]
    th.cuda.empty_cache()

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

    # For relative projections
    sum_rel_proj = th.zeros(D, device=device)
    count_rel_proj = th.zeros(D, device=device)

    # Track min/max for dynamic bucket adjustment
    scalar_min = th.full((D,), float('inf'), device=device)
    scalar_max = th.full((D,), float('-inf'), device=device)

    # Initialize temporary bucket storage
    temp_bucket_data = [{
        'values_base': [],
        'values_it': [],
        'values_scalar': []
    } for _ in range(D)]

    with th.no_grad():
        crosscoder.to(device)

        # Single pass through the data
        for batch_data in tqdm(dataloader, desc="Computing statistics"):
            batch_data = batch_data.to(device)
            batch_base = batch_data[:, 0]
            batch_it = batch_data[:, 1]
            batch_features = crosscoder.encode(batch_data)

            # Update global min/max
            scalar_min = th.minimum(scalar_min, batch_features.min(dim=0)[0])
            scalar_max = th.maximum(scalar_max, batch_features.max(dim=0)[0])

            for d_start in range(0, D, d_batch_size):
                d_end = min(d_start + d_batch_size, D)
                feat_batch = features[d_start:d_end].to(device)
                
                proj_base = calculate_projection(batch_base, feat_batch)
                proj_it = calculate_projection(batch_it, feat_batch)
                scalar_batch = batch_features[:, d_start:d_end]

                # Compute relative projections
                valid_mask = proj_it != 0
                rel_proj = th.where(valid_mask, proj_base / proj_it, th.tensor(float('nan'), device=device))

                # Update overall statistics
                for i in range(d_end - d_start):
                    feat_idx = d_start + i
                    
                    # Store values for later bucket processing
                    temp_bucket_data[feat_idx]['values_base'].append(proj_base[:, i].cpu())
                    temp_bucket_data[feat_idx]['values_it'].append(proj_it[:, i].cpu())
                    temp_bucket_data[feat_idx]['values_scalar'].append(scalar_batch[:, i].cpu())

                    # Update overall statistics
                    new_count = count[feat_idx] + proj_base.size(0)
                    delta_proj_base = proj_base[:, i] - mean_proj_base[feat_idx]
                    delta_proj_it = proj_it[:, i] - mean_proj_it[feat_idx]
                    delta_scalar = scalar_batch[:, i] - mean_scalar[feat_idx]

                    mean_proj_base[feat_idx] += delta_proj_base.sum() / new_count
                    mean_proj_it[feat_idx] += delta_proj_it.sum() / new_count
                    mean_scalar[feat_idx] += delta_scalar.sum() / new_count

                    delta2_proj_base = proj_base[:, i] - mean_proj_base[feat_idx]
                    delta2_proj_it = proj_it[:, i] - mean_proj_it[feat_idx]
                    delta2_scalar = scalar_batch[:, i] - mean_scalar[feat_idx]

                    M2_proj_base[feat_idx] += (delta_proj_base * delta2_proj_base).sum()
                    M2_proj_it[feat_idx] += (delta_proj_it * delta2_proj_it).sum()
                    M2_scalar[feat_idx] += (delta_scalar * delta2_scalar).sum()
                    covar_base[feat_idx] += (delta_proj_base * delta2_scalar).sum()
                    covar_it[feat_idx] += (delta_proj_it * delta2_scalar).sum()
                    count[feat_idx] = new_count

                    # Update relative projection statistics
                    valid_rel_proj = rel_proj[:, i][valid_mask[:, i]]
                    if valid_rel_proj.numel() > 0:
                        sum_rel_proj[feat_idx] += valid_rel_proj.sum()
                        count_rel_proj[feat_idx] += valid_rel_proj.numel()

                del feat_batch, proj_base, proj_it, scalar_batch
                th.cuda.empty_cache()

        # Compute final correlations
        valid_count = count > 1
        correlations_base = th.full((D,), float('nan'), device=device)
        correlations_it = th.full((D,), float('nan'), device=device)

        var_proj_base = M2_proj_base[valid_count] / (count[valid_count] - 1)
        var_proj_it = M2_proj_it[valid_count] / (count[valid_count] - 1)
        var_scalar = M2_scalar[valid_count] / (count[valid_count] - 1)
        
        correlations_base[valid_count] = covar_base[valid_count] / th.sqrt(var_proj_base * var_scalar) / (count[valid_count] - 1)
        correlations_it[valid_count] = covar_it[valid_count] / th.sqrt(var_proj_it * var_scalar) / (count[valid_count] - 1)
        
        results["correlations"] = {
            "base": correlations_base.cpu(),
            "it": correlations_it.cpu()
        }

        # Compute relative projections
        mean_rel_proj = th.where(count_rel_proj > 0, 
                               sum_rel_proj / count_rel_proj,
                               th.tensor(float('nan'), device=device))
        results["relproj"] = mean_rel_proj.cpu()

        # Process bucket statistics
        bucket_correlations_base = th.full((D, n_buckets), float('nan'), device=device)
        bucket_correlations_it = th.full((D, n_buckets), float('nan'), device=device)
        bucket_count = th.zeros((D, n_buckets), device=device)
        bucket_rel_proj = th.full((D, n_buckets), float('nan'), device=device)

        for i in range(D):
            values_base = th.cat(temp_bucket_data[i]['values_base'])
            values_it = th.cat(temp_bucket_data[i]['values_it'])
            values_scalar = th.cat(temp_bucket_data[i]['values_scalar'])
            
            bucket_edges = th.linspace(scalar_min[i], scalar_max[i], n_buckets + 1)
            
            for b in range(n_buckets):
                mask = (values_scalar >= bucket_edges[b]) & (values_scalar < bucket_edges[b+1])
                if mask.sum() > 1:  # Need at least 2 points for correlation
                    bucket_count[i, b] = mask.sum()
                    
                    # Compute bucket correlations
                    bucket_correlations_base[i, b] = th.corrcoef(
                        th.stack([values_base[mask], values_scalar[mask]])
                    )[0, 1]
                    bucket_correlations_it[i, b] = th.corrcoef(
                        th.stack([values_it[mask], values_scalar[mask]])
                    )[0, 1]
                    
                    # Compute bucket relative projections
                    valid_rel = (values_it[mask] != 0)
                    if valid_rel.any():
                        rel_proj = values_base[mask][valid_rel] / values_it[mask][valid_rel]
                        bucket_rel_proj[i, b] = th.nanmean(rel_proj)

        results["bucket_correlations"] = {
            "base": bucket_correlations_base.cpu(),
            "it": bucket_correlations_it.cpu()
        }
        results["bucket_count"] = bucket_count.cpu()
        results["bucket_relproj"] = bucket_rel_proj.cpu()

    return results


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Compute feature metrics on model activations')
    parser.add_argument('-n', type=int, default=100_000,
                    help='Number of activations to load from validation set (default: 100,000)')
    parser.add_argument('-batch_size', type=int, default=4096,
                    help='Batch size over N dimension (default: 1024)')
    parser.add_argument('-d_batch_size', type=int, default=2048,
                    help='Batch size over D dimension (default: 1024)')
    parser.add_argument('-n_buckets', type=int, default=10,
                    help='Number of buckets to use for bucketed correlations (default: 10)')
    args = parser.parse_args()

    main(args.n, args.batch_size, args.d_batch_size, args.n_buckets)