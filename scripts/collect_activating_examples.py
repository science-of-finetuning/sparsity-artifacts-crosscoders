"""
Script to collect examples at different activation quantiles for a CrossCoder.
"""

from pathlib import Path
import argparse
import gc
import sqlite3
import sys
import random
from collections import defaultdict
import numpy as np
import torch as th
from tqdm import tqdm
from huggingface_hub import hf_api, repo_exists, file_exists
import wandb
import time

sys.path.append(".")

from tools.cache_utils import LatentActivationCache
from tools.configs import HF_NAME

th.set_grad_enabled(False)


def quantile_examples_to_db(
    quantile_examples, all_sequences, activation_details, db_path: Path
):
    """Convert quantile examples to a database with binary blob storage for token IDs.

    Args:
        quantile_examples: Dictionary mapping quantile_idx -> feature_idx -> list of (activation_value, sequence_idx)
        all_sequences: List of all sequences used in the examples
        activation_details: Dictionary mapping feature_idx -> sequence_idx -> list of (position, value) pairs
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

                    cursor.execute(
                        "INSERT INTO quantile_examples VALUES (?, ?, ?, ?)",
                        (
                            int(feature_idx),
                            int(q_idx),
                            float(activation),
                            int(sequence_idx),
                        ),
                    )

        # Create a table for storing activation details
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS activation_details (
                feature_idx INTEGER,
                sequence_idx INTEGER,
                positions BLOB,
                activation_values BLOB,
                PRIMARY KEY (feature_idx, sequence_idx),
                FOREIGN KEY (sequence_idx) REFERENCES sequences(sequence_idx),
                FOREIGN KEY (feature_idx, sequence_idx) REFERENCES quantile_examples(feature_idx, sequence_idx)
            )"""
        )

        # After storing all quantile examples
        # Store activation details
        for feature_idx, sequences in tqdm(
            activation_details.items(), desc="Storing activation details"
        ):
            for sequence_idx, pos_val_pairs in sequences.items():
                if len(pos_val_pairs) == 0:
                    continue

                positions_blob = pos_val_pairs[:, 0].tobytes()
                values_blob = pos_val_pairs[:, 1].tobytes()

                cursor.execute(
                    "INSERT INTO activation_details VALUES (?, ?, ?, ?)",
                    (
                        int(feature_idx),
                        int(sequence_idx),
                        positions_blob,
                        values_blob,
                    ),
                )

        conn.commit()


def fix_activations_details(activation_details):
    """Convert activation details from int32 arrays to tuples of (positions, values) with proper types."""
    converted = {}
    for feat_idx, sequences in activation_details.items():
        converted[feat_idx] = {}
        for seq_idx, arr in sequences.items():
            # arr is a Nx2 array where first column is positions (int) and second column is values (float as int32)
            positions = arr[:, 0].astype(np.int32)
            # Convert back the int32 values to float32
            values = arr[:, 1].view(np.float32)
            converted[feat_idx][seq_idx] = (positions, values)
    return converted


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
    gc_collect_every=1000,
    test=False,
    log_time=False,
    use_random_replacement=True,
    file_name: str = "examples",
) -> dict:
    """Compute examples that activate features at different quantile levels.

    Args:
        latent_activation_cache: Pre-computed latent activation cache
        quantiles: List of quantile thresholds (as fractions of max activation)
        min_threshold: Minimum activation threshold to consider
        n: Number of examples to collect per feature per quantile
        save_path: Path to save results
        gc_collect_every: How often to run garbage collection

    Returns:
        Tuple of (quantile_examples, all_sequences) where:
            - quantile_examples: Dictionary mapping quantile_idx -> feature_idx -> list of (activation_value, sequence_idx, position)
            - all_sequences: List of all token sequences used in the examples
    """
    log_time = log_time or test
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Move max_activations and quantiles to GPU
    max_activations = latent_activation_cache.max_activations
    quantiles_tensor = th.tensor(quantiles, device=device)

    # Calculate quantile thresholds for each feature on GPU
    thresholds = th.einsum("f,q->fq", max_activations, quantiles_tensor)

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

    # Dictionary to store feature activation details: {feature_idx: {sequence_idx: [(position, value), ...]}}
    activation_details = defaultdict(dict)

    timings = defaultdict(list)  # Changed to list to store all iterations

    def _log_time(section, start_time, add=False):
        if not log_time:
            return None
        elapsed = time.time() - start_time
        if add:
            timings[section][-1] += elapsed
        else:
            timings[section].append(elapsed)
        return time.time()

    next_gb = gc_collect_every
    current_seq_idx = 0
    for tokens, (indices, values) in tqdm(latent_activation_cache):
        iter_start = time.time() if log_time else None

        # GC and device transfer timing
        next_gb -= 1
        if next_gb <= 0:
            gc.collect()
            next_gb = gc_collect_every
        if test and next_gb <= 800:
            break
        current = _log_time("1. GC and device transfer", iter_start)

        token_tuple = tuple(tokens.tolist())
        if token_tuple in sequences_set:
            continue
        sequences_set.add(token_tuple)
        all_sequences.append(token_tuple)
        current = _log_time("2. Sequence processing", current)

        # Core computation timing
        features, sort_indices = th.sort(indices[:, 1])
        token_indices = indices[:, 0][sort_indices]
        values = values[sort_indices]
        active_features, inverse_indices, counts = features.unique(
            return_inverse=True, return_counts=True
        )
        max_vals = th.zeros_like(active_features, dtype=values.dtype)
        max_vals = th.scatter_reduce(
            max_vals, 0, inverse_indices, values, reduce="amax"
        )
        # sorted_values = values[sorted_feature_indices]
        # quantile_indices = th.searchsorted(all_tresholds, sorted_values)
        active_thresholds = thresholds[active_features]
        # assert sorted_values.dim() == 1, "batching not supported yet"
        # cum_count = th.cumsum(counts, dim=0)
        # max_vals = th.zeros_like(active_features, dtype=values.dtype)
        # _, inverse_indices = th.unique(sorted_features, return_inverse=True)
        # max_vals = th.scatter_reduce(
        #     max_vals, 0, inverse_indices, values, reduce="amax"
        # )
        # cum_max_values = th.cummax(sorted_values, dim=0)[0]
        # max_vals = cum_max_values[cum_count - 1]
        q_idxs = th.searchsorted(active_thresholds, max_vals.unsqueeze(-1)).squeeze()

        current_preloop = _log_time("3. Core computation", current)

        active_features = active_features.tolist()
        counts = counts.tolist()
        max_vals = max_vals.tolist()
        q_idxs = q_idxs.tolist()
        # max_values =
        # Example collection timing
        current_idx = 0
        latent_details = (
            th.stack(
                [token_indices.int(), values.float().view(th.int32)],
                dim=1,
            )
            .cpu()
            .numpy()
        )
        # inverse_indices = inverse_indices.cpu().numpy()
        # print(inverse_indices)
        # input()
        current = _log_time("4. move and convert to numpy", current)
        if log_time:
            for times in [
                "loop1",
                "loop2",
                "loop3",
                "update_details",
            ]:
                timings[times].append(0)
        for feat, count, max_val, q_idx in zip(
            active_features,
            counts,
            # inverse_indices,
            max_vals,
            q_idxs,
        ):
            current = time.time() if log_time else None
            example_counts[q_idx][feat] += 1
            total_count = example_counts[q_idx][feat]

            if total_count <= n:
                quantile_examples[q_idx][feat].append((max_val, current_seq_idx))
                current = _log_time("loop1", current, add=True)
                # Time the activation details collection
                activation_details[feat][current_seq_idx] = latent_details[
                    current_idx : current_idx + count
                ]
                _log_time("update_details", current, add=True)
            elif use_random_replacement:
                if random.random() < n / total_count:
                    replace_idx = random.randint(0, n - 1)
                    replaced_seq_idx = quantile_examples[q_idx][feat][replace_idx][1]
                    quantile_examples[q_idx][feat][replace_idx] = (
                        max_val,
                        current_seq_idx,
                    )
                    current = _log_time("loop2", current, add=True)
                    if (
                        feat in activation_details
                        and replaced_seq_idx in activation_details[feat]
                    ):
                        del activation_details[feat][replaced_seq_idx]
                    current = _log_time("loop3", current, add=True)
                    activation_details[feat][current_seq_idx] = latent_details[
                        current_idx : current_idx + count
                    ]
                    _log_time("update_details", current, add=True)
            current_idx += count

        current = _log_time(
            "5. Example collection and activation details", current_preloop
        )
        if (
            len(timings["5. Example collection and activation details"]) % 10 == 0
            and log_time
        ):  # Print periodically
            print("\nCurrent mean timings per iteration:")
            for section, times in timings.items():
                mean_time = sum(times) / len(times)
                print(f"{section}: {mean_time:.4f}s")
        current_seq_idx += 1

    if log_time:
        print("\nFinal timings:")
        for section, times in timings.items():
            mean_time = sum(times) / len(times)
            print(f"{section}: {mean_time:.4f}s")

    # Sort and finalize results
    print(f"Sorting {len(quantile_examples)} quantiles")
    quantile_examples = sort_quantile_examples(quantile_examples)
    name = ("test_" if test else "") + file_name
    # Save to database
    if save_path is not None:
        print(f"Saving to {save_path / f'{name}.db'}")
        quantile_examples_to_db(
            quantile_examples,
            all_sequences,
            activation_details,
            save_path / f"{name}.db",
        )
        print(f"Saving to {save_path / f'{name}.pt'}")
        # Also save as PyTorch file for compatibility

    activation_details = fix_activations_details(activation_details)
    if save_path is not None:
        th.save(
            (quantile_examples, all_sequences, activation_details),
            save_path / f"{name}.pt",
        )

    return quantile_examples, all_sequences, activation_details


# python scripts/collect_max_activating_examples.py gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss --latent-activation-cache-path $DATASTORE/latent_activations
# python scripts/collect_max_activating_examples.py  gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04 --latent-activation-cache-path $DATASTORE/latent_activations


def collect_activating_examples(
    crosscoder: str,
    latent_activation_cache: LatentActivationCache,
    bos_token_id: int = 2,
    n: int = 100,
    min_threshold: float = 1e-4,
    quantiles: list[float] = [0.25, 0.5, 0.75, 0.95, 1.0],
    save_path: Path = Path("results/quantile_examples"),
    only_upload: bool = False,
    test: bool = False,
    file_name: str = "examples",
) -> None:
    """
    Collect and save examples that activate latent features at different quantiles.

    This function processes latent activations to find examples that activate features
    at specified quantile thresholds. It can optionally save results locally and/or
    upload them to HuggingFace Hub.

    Args:
        crosscoder (str): Name of the crosscoder model to analyze
        latent_activation_cache_path (Path): Path to directory containing latent activation data
        bos_token_id (int, optional): Beginning of sequence token ID. Defaults to 2.
        n (int, optional): Number of examples to collect per quantile. Defaults to 100.
        min_threshold (float, optional): Minimum activation threshold. Defaults to 1e-4.
        quantiles (list[float], optional): Quantile thresholds to analyze.
            Defaults to [0.25, 0.5, 0.75, 0.95, 1.0].
        save_path (Path, optional): Directory to save results.
            Defaults to Path("results/quantile_examples").
        only_upload (bool, optional): If True, only upload existing results to HuggingFace.
            Defaults to False.
        test (bool, optional): If True, run in test mode with smaller dataset.
            Defaults to False.

    Returns:
        None
    """
    save_path = save_path / crosscoder

    if not only_upload:
        # Initialize wandb
        project = "quantile-activating-examples"
        if test:
            project = "test-" + project
        wandb.init(
            project=project,
            config={
                "crosscoder": crosscoder,
                "bos_token_id": bos_token_id,
                "n": n,
                "min_threshold": min_threshold,
                "quantiles": quantiles,
                "save_path": str(save_path),
                "test": test,
            },
        )

        # Create save directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate and save quantile examples
        print("Generating quantile examples...")
        compute_quantile_activating_examples(
            latent_activation_cache=latent_activation_cache,
            quantiles=quantiles,
            min_threshold=min_threshold,
            n=n,
            save_path=save_path,
            file_name=file_name,
            test=test,
        )

        wandb.finish()

    # Upload to HuggingFace Hub
    repo = f"{HF_NAME}/diffing-stats-" + crosscoder
    if not test:
        print(f"Uploading to HuggingFace Hub: {repo}")
        for ftype in ["pt", "db"]:
            name = ("test_" if test else "") + file_name
            file_path = save_path / f"{name}.{ftype}"
            print(f"Uploading {file_path} to {repo}")
            if file_path.exists():
                hf_api.upload_file(
                    repo_id=repo,
                    repo_type="dataset",
                    path_or_fileobj=file_path,
                    path_in_repo=f"{name}.{ftype}",
                )


if __name__ == "__main__":
    import os

    # python scripts/collect_activating_examples.py SAE-base-gemma-2-2b-L13-k100-x32-lr1e-04-local-shuffling --latent-activation-cache-path $DATASTORE/latent_activations --file-name "examples_from_base" --latent-activation-cache-suffix "from_base"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("crosscoder", type=str)
    parser.add_argument(
        "--latent-activation-cache-path", type=Path, default="./data/latent_activations"
    )
    parser.add_argument("--latent-activation-cache-suffix", type=str, default="")
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
    parser.add_argument("--file-name", type=str, default="examples")

    args = parser.parse_args()
    # Load latent activation cache
    if args.only_upload:
        latent_activation_cache = None
    else:
        device = "cuda" if th.cuda.is_available() else "cpu"
        path = args.latent_activation_cache_path / args.crosscoder
        if args.latent_activation_cache_suffix:
            path = path / args.latent_activation_cache_suffix
        latent_activation_cache = LatentActivationCache(
            path,
            expand=False,
        ).to(device)
    collect_activating_examples(
        crosscoder=args.crosscoder,
        latent_activation_cache=latent_activation_cache,
        bos_token_id=args.bos_token_id,
        n=args.n,
        min_threshold=args.min_threshold,
        quantiles=args.quantiles,
        save_path=args.save_path,
        only_upload=args.only_upload,
        test=args.test,
        file_name=args.file_name,
    )
