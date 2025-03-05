"""
Script to collect examples at different activation quantiles for a CrossCoder.
"""

from pathlib import Path
import argparse
import gc
import sqlite3
import sys
import random
import numpy as np

import torch as th
from tqdm import tqdm
import wandb
from huggingface_hub import hf_api

sys.path.append(".")

from tools.cache_utils import LatentActivationCache

th.set_grad_enabled(False)


def quantile_examples_to_db(quantile_examples, all_sequences, db_path: Path):
    """Convert quantile examples to a database with binary blob storage for token IDs.

    Args:
        quantile_examples: Dictionary mapping quantile_idx -> feature_idx -> list of (activation_value, sequence_idx)
        all_sequences: List of all sequences used in the examples
        db_path: Path to save the database
    """
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS sequences (
                sequence_idx INTEGER PRIMARY KEY,
                token_ids BLOB
            )"""
        )

        cursor.execute(
            """CREATE TABLE IF NOT EXISTS quantile_examples (
                feature_idx INTEGER,
                quantile_idx INTEGER,
                activation REAL,
                sequence_idx INTEGER,
                max_position INTEGER,
                PRIMARY KEY (feature_idx, sequence_idx),
                FOREIGN KEY (sequence_idx) REFERENCES sequences(sequence_idx)
            )"""
        )

        # First, store all sequences
        for seq_idx, token_ids in tqdm(
            enumerate(all_sequences), desc="Storing sequences"
        ):
            # Convert token IDs to binary blob
            binary_data = np.array(token_ids, dtype=np.int32).tobytes()
            cursor.execute(
                "INSERT INTO sequences VALUES (?, ?)",
                (int(seq_idx), binary_data),
            )

        # Then store the quantile examples with references to sequences
        for q_idx, q_data in tqdm(
            quantile_examples.items(), desc="Storing quantile examples"
        ):
            for feature_idx, examples in q_data.items():
                for activation, sequence_idx in examples:
                    # Get the max position from the original sequence
                    # This assumes we're still tracking max positions somewhere
                    # If not, we'd need to modify the compute_quantile_activating_examples function
                    # to also track positions along with activations

                    # For now, using a placeholder of 0 for max_position
                    max_position = 0

                    cursor.execute(
                        "INSERT INTO quantile_examples VALUES (?, ?, ?, ?, ?)",
                        (
                            int(feature_idx),
                            int(q_idx),
                            float(activation),
                            int(sequence_idx),
                            int(max_position),
                        ),
                    )
        conn.commit()


def sort_quantile_examples(quantile_examples):
    """Sort quantile examples by activation value."""
    for q_idx in quantile_examples:
        for feature_idx in quantile_examples[q_idx]:
            quantile_examples[q_idx][feature_idx] = sorted(
                quantile_examples[q_idx][feature_idx],
                key=lambda x: x[0],
                reverse=True,
            )
    return quantile_examples


@th.no_grad()
def compute_quantile_activating_examples(
    latent_activation_cache,
    quantiles=[0.25, 0.5, 0.75, 0.95],
    min_threshold=1e-4,
    n=100,
    save_path=None,
    name="quantile_examples",
    gc_collect_every=1000,
    test=False,
) -> dict:
    """Compute examples that activate features at different quantile levels.

    Args:
        latent_activation_cache: Pre-computed latent activation cache
        quantiles: List of quantile thresholds (as fractions of max activation)
        min_threshold: Minimum activation threshold to consider
        n: Number of examples to collect per feature per quantile
        save_path: Path to save results
        name: Name for saving results
        gc_collect_every: How often to run garbage collection

    Returns:
        Tuple of (quantile_examples, all_sequences) where:
            - quantile_examples: Dictionary mapping quantile_idx -> feature_idx -> list of (activation_value, sequence_idx, position)
            - all_sequences: List of all token sequences used in the examples
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Move max_activations and quantiles to GPU
    max_activations = latent_activation_cache.max_activations.to(device)
    quantiles_tensor = th.tensor(quantiles, device=device)

    # Calculate quantile thresholds for each feature on GPU
    thresholds = th.einsum("f,q->fq", max_activations, quantiles_tensor)

    if save_path is not None:
        save_path = save_path / name
        save_path.mkdir(parents=True, exist_ok=True)

    # Initialize collections for each quantile
    quantile_examples = {
        q_idx: {feat_idx: [] for feat_idx in range(len(max_activations))}
        for q_idx in range(len(quantiles) + 1)
    }

    # Keep track of how many examples we've seen for each feature and quantile
    example_counts = {
        q_idx: {feat_idx: 0 for feat_idx in range(len(max_activations))}
        for q_idx in range(len(quantiles) + 1)
    }

    # Store all unique sequences
    sequences_set = set()
    all_sequences = []
    current_idx = 0

    next_gb = gc_collect_every
    for tokens, (indices, values) in tqdm(latent_activation_cache):
        next_gb -= 1
        if next_gb <= 0:
            gc.collect()
            next_gb = gc_collect_every
            if test:
                break
        # Move sparse tensors to GPU
        indices = indices.to(device)
        values = values.to(device)

        token_tuple = tuple(tokens.tolist())
        if token_tuple in sequences_set:
            continue
        sequences_set.add(token_tuple)
        all_sequences.append(token_tuple)

        # Group activations by feature (on GPU)
        features = indices[:, 1]  # feature indices
        # Use scatter_reduce to find max activations
        max_acts = th.zeros(max_activations.size(0), device=device)
        max_acts.scatter_reduce_(0, features, values, reduce="amax", include_self=False)

        # Get active features and find quantile indices
        active_features = (max_acts >= min_threshold).nonzero().squeeze(-1)
        active_max_vals = max_acts[active_features]

        # Find quantile indices for all active features at once
        thresholds_dense = thresholds[active_features]  # shape: [n_active, n_quantiles]

        # Find quantile indices using searchsorted
        quantile_indices = th.searchsorted(
            thresholds_dense, active_max_vals.unsqueeze(-1)
        )

        # Move to CPU
        active_features = active_features.cpu()
        active_max_vals = active_max_vals.cpu()
        quantile_indices = quantile_indices.cpu().squeeze(-1)

        # Update quantile examples
        for feat, max_val, q_idx in zip(
            active_features.tolist(),
            active_max_vals.tolist(),
            quantile_indices.tolist(),
        ):
            example_counts[q_idx][feat] += 1
            count = example_counts[q_idx][feat]

            if count <= n:
                quantile_examples[q_idx][feat].append((max_val, current_idx))
            else:
                if random.random() < n / count:
                    replace_idx = random.randint(0, n - 1)
                    quantile_examples[q_idx][feat][replace_idx] = (max_val, current_idx)

        current_idx += 1

    # Sort and finalize results
    print(f"Sorting {len(quantile_examples)} quantiles")
    quantile_examples = sort_quantile_examples(quantile_examples)
    if test:
        name = f"test_{name}"
    # Save to database
    if save_path:
        print(f"Saving to {save_path / f'{name}_final.db'}")
        quantile_examples_to_db(
            quantile_examples, all_sequences, save_path / f"{name}_final.db"
        )
        print(f"Saving to {save_path / f'{name}_final.pt'}")
        # Also save as PyTorch file for compatibility
        th.save((quantile_examples, all_sequences), save_path / f"{name}_final.pt")

    return quantile_examples, all_sequences


# python scripts/collect_max_activating_examples.py gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss --latent-activation-cache-path /workspace/data/latent_activations --test
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("crosscoder", type=str)
    parser.add_argument("--latent-activation-cache-path", type=Path, required=True)
    parser.add_argument("--bos-token-id", type=int, default=2)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--min-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.95, 1.0]
    )
    parser.add_argument(
        "--save-path", type=Path, default=Path("results/quantile_examples")
    )
    parser.add_argument("--only-upload", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    save_path = args.save_path / args.crosscoder

    if not args.only_upload:

        # Initialize wandb
        project = "quantile-activating-examples"
        if args.test:
            project = "test-" + project
        wandb.init(project=project, config=vars(args))

        # Load latent activation cache
        latent_activation_cache = LatentActivationCache(
            args.latent_activation_cache_path / args.crosscoder, expand=False
        )

        # Create save directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate and save quantile examples
        print("Generating quantile examples...")
        quantile_examples, all_sequences = compute_quantile_activating_examples(
            latent_activation_cache=latent_activation_cache,
            quantiles=args.quantiles,
            min_threshold=args.min_threshold,
            n=args.n,
            save_path=save_path,
            name="quantile_examples",
            test=args.test,
        )

        wandb.finish()

    # Upload to HuggingFace Hub
    repo = "science-of-finetuning/diffing-stats-" + args.crosscoder
    if not args.test:
        print(f"Uploading to HuggingFace Hub: {repo}")
        for ftype in ["pt", "db"]:
            file_path = save_path / f"quantile_examples_final.{ftype}"
            if file_path.exists():
                hf_api.upload_file(
                    repo_id=repo,
                    repo_type="dataset",
                    path_or_fileobj=file_path,
                    path_in_repo=f"quantile_examples.{ftype}",
                )


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
