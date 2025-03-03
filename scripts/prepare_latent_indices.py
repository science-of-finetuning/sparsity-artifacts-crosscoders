# %%
import sys
sys.path.append(".")
import torch as th
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from tools.cc_utils import chat_only_latent_indices, base_only_latent_indices, shared_latent_indices, load_latent_df

def main():
    parser = argparse.ArgumentParser(description="Prepare latent indices from dataframe")
    parser.add_argument(
        "--latent-df", 
        type=Path, 
        help="Path or name of the latent dataframe. If not provided, the latent dataframe will be loaded from the crosscoder name.",
        default=None
    )
    parser.add_argument(
        "--results-dir", 
        type=Path, 
        default="/workspace/data/latent_indices",
        help="Directory to save the indices"
    )
    parser.add_argument(
        "--crosscoder", 
        type=str, 
        required=True,
        help="Name or path of the crosscoder model"
    )
    parser.add_argument(
        "--shared-sample-size", 
        type=int, 
        default=1000,
        help="Number of shared indices to sample"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--low-norm-diff-sample-size",
        type=int,
        default=3000,
        help="Number of low norm diff indices to sample"
    )
    args = parser.parse_args()

    if args.latent_df is None:
        args.latent_df = args.crosscoder

    # Set random seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # Create results directory
    results_dir = args.results_dir / args.crosscoder.replace("/", "_")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading latent dataframe from {args.latent_df}")
    
    # Get indices using the utility functions
    logger.info("Extracting chat-only indices")
    chat_only_indices = chat_only_latent_indices(args.latent_df)
    
    logger.info("Extracting base-only indices")
    base_only_indices = base_only_latent_indices(args.latent_df)
    
    logger.info("Extracting shared indices")
    shared_indices = shared_latent_indices(args.latent_df)

    # Extracting the 3000 highest norm difference indices
    df = load_latent_df(args.latent_df)
    df = df.sort_values(by="dec_norm_diff", ascending=True)
    print(df.head(), df.tail())
    low_norm_diff_indices = df.index[:args.low_norm_diff_sample_size]
    
    
    # Sample from shared indices
    logger.info(f"Sampling {args.shared_sample_size} shared indices")
    if len(shared_indices) > args.shared_sample_size:
        shared_sample = th.tensor(np.random.choice(
            shared_indices, 
            size=args.shared_sample_size, 
            replace=False
        ))
    else:
        logger.warning(f"Requested sample size {args.shared_sample_size} is larger than available shared indices ({len(shared_indices)})")
        shared_sample = th.tensor(shared_indices)
    
    # Convert to torch tensors
    chat_only_indices_tensor = th.tensor(chat_only_indices)
    base_only_indices_tensor = th.tensor(base_only_indices)
    shared_indices_tensor = th.tensor(shared_indices)
    low_norm_diff_indices_tensor = th.tensor(low_norm_diff_indices)
    
    # Save tensors
    logger.info(f"Saving indices to {results_dir}")
    th.save(chat_only_indices_tensor, results_dir / "chat_only_indices.pt")
    th.save(base_only_indices_tensor, results_dir / "base_only_indices.pt")
    th.save(shared_indices_tensor, results_dir / "shared_indices.pt")
    th.save(shared_sample, results_dir / "shared_indices_sample.pt")
    th.save(low_norm_diff_indices_tensor, results_dir / f"low_norm_diff_indices_{args.low_norm_diff_sample_size}.pt")
    
    # Log statistics
    logger.info(f"Chat-only indices: {len(chat_only_indices)}")
    logger.info(f"Base-only indices: {len(base_only_indices)}")
    logger.info(f"Shared indices: {len(shared_indices)}")
    logger.info(f"Shared indices sample: {len(shared_sample)}")
    logger.info(f"Low norm diff indices: {len(low_norm_diff_indices)}")

if __name__ == "__main__":
    main()

# %%
