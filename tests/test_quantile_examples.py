import sys
from pathlib import Path
import torch as th
import pytest
import random
import numpy as np
import bisect

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.collect_max_activating_examples import compute_quantile_activating_examples, quantile_examples_to_db
from tools.utils import QuantileExamplesDB

@th.no_grad()
def compute_quantile_activating_examples_old(
    latent_activation_cache,
    quantiles=[0.25, 0.5, 0.75, 0.95],
    min_threshold=1e-4,
    n=100,
    save_path=None,
    name="quantile_examples",
    gc_collect_every=1000,
    test=False,
) -> dict:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Move max_activations and quantiles to GPU
    max_activations = latent_activation_cache.max_activations.to(device)
    quantiles_tensor = th.tensor(quantiles, device=device)

    # Calculate quantile thresholds for each feature on GPU
    thresholds = th.einsum("f,q->fq", max_activations, quantiles_tensor)
    thresholds = thresholds.cpu().tolist()

    quantile_examples = {
        q_idx: {feat_idx: [] for feat_idx in range(len(max_activations))}
        for q_idx in range(len(quantiles) + 1)
    }

    example_counts = {
        q_idx: {feat_idx: 0 for feat_idx in range(len(max_activations))}
        for q_idx in range(len(quantiles) + 1)
    }

    sequences_set = set()
    all_sequences = []
    current_idx = 0

    for tokens, (indices, values) in latent_activation_cache:
        if test and current_idx > 10:  # Limit iterations for test
            break

        indices = indices.to(device)
        values = values.to(device)

        token_tuple = tuple(tokens.tolist())
        if token_tuple in sequences_set:
            continue
        sequences_set.add(token_tuple)
        all_sequences.append(token_tuple)

        positions = indices[:, 0]
        features = indices[:, 1]
        unique_features = features.unique()

        for feat_idx in unique_features.cpu().tolist():
            feat_mask = features == feat_idx
            feat_acts = values[feat_mask]

            max_act_val = feat_acts.max()
            max_act_val = max_act_val.item()

            if max_act_val < min_threshold:
                continue

            q_thresholds = thresholds[feat_idx]
            q_idx = bisect.bisect_left(q_thresholds, max_act_val)

            example_counts[q_idx][feat_idx] += 1
            count = example_counts[q_idx][feat_idx]

            if count <= n:
                quantile_examples[q_idx][feat_idx].append((max_act_val, current_idx))
            else:
                if random.random() < n / count:
                    replace_idx = random.randint(0, n - 1)
                    quantile_examples[q_idx][feat_idx][replace_idx] = (
                        max_act_val,
                        current_idx,
                    )
        current_idx += 1

    return quantile_examples, all_sequences


class MockLatentActivationCache:
    def __init__(self, num_features=100, num_sequences=5, seq_length=10, sparsity=0.1):
        self.num_features = num_features
        self.num_sequences = num_sequences
        self.seq_length = seq_length

        # Generate random max activations for each feature
        self.max_activations = th.rand(num_features)

        # Generate sparse activations for testing
        self.data = []
        for seq_idx in range(num_sequences):
            # Generate random sparse activations
            num_activations = int(num_features * seq_length * sparsity)
            positions = th.randint(0, seq_length, (num_activations,))
            features = th.randint(0, num_features, (num_activations,))
            values = th.rand(num_activations)

            # Create sequence of tokens
            tokens = th.randint(0, 1000, (seq_length,))

            # Store indices as (position, feature)
            indices = th.stack([positions, features], dim=1)

            self.data.append((tokens, (indices, values)))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def compare_results(result1, result2):
    """Compare two sets of results from the quantile computation functions."""
    quantile_examples1, sequences1 = result1
    quantile_examples2, sequences2 = result2

    # Compare sequences
    assert sequences1 == sequences2, "Sequences don't match"

    # Compare quantile examples structure
    assert (
        quantile_examples1.keys() == quantile_examples2.keys()
    ), "Quantile indices don't match"

    for q_idx in quantile_examples1:
        assert (
            quantile_examples1[q_idx].keys() == quantile_examples2[q_idx].keys()
        ), f"Feature indices don't match for quantile {q_idx}"

        for feat_idx in quantile_examples1[q_idx]:
            examples1 = sorted(quantile_examples1[q_idx][feat_idx])
            examples2 = sorted(quantile_examples2[q_idx][feat_idx])

            # Compare lengths
            assert len(examples1) == len(
                examples2
            ), f"Number of examples doesn't match for quantile {q_idx}, feature {feat_idx}"

            # Compare values with tolerance
            for (val1, idx1), (val2, idx2) in zip(examples1, examples2):
                assert (
                    abs(val1 - val2) < 1e-5
                ), f"Activation values don't match for quantile {q_idx}, feature {feat_idx}"
                assert (
                    idx1 == idx2
                ), f"Sequence indices don't match for quantile {q_idx}, feature {feat_idx}"


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_quantile_examples_implementations(seed):
    """Test that both implementations produce the same results."""
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # Create mock data
    mock_cache = MockLatentActivationCache(
        num_features=100, num_sequences=5, seq_length=10, sparsity=0.1
    )

    # Run both implementations
    result_old = compute_quantile_activating_examples_old(
        mock_cache, quantiles=[0.25, 0.5, 0.75], min_threshold=1e-4, n=5, test=True
    )

    result_new = compute_quantile_activating_examples(
        mock_cache, quantiles=[0.25, 0.5, 0.75], min_threshold=1e-4, n=5, test=True
    )

    # Compare results
    compare_results(result_old, result_new)


def test_edge_cases():
    """Test edge cases like empty sequences or all zeros."""
    # Test with empty sequences
    mock_cache = MockLatentActivationCache(
        num_features=10, num_sequences=0, seq_length=5, sparsity=0.1
    )

    result_old = compute_quantile_activating_examples_old(mock_cache, test=True)
    result_new = compute_quantile_activating_examples(mock_cache, test=True)
    compare_results(result_old, result_new)

    # Test with all zero activations
    mock_cache = MockLatentActivationCache(
        num_features=10, num_sequences=5, seq_length=5, sparsity=0
    )

    result_old = compute_quantile_activating_examples_old(mock_cache, test=True)
    result_new = compute_quantile_activating_examples(mock_cache, test=True)
    compare_results(result_old, result_new)

def test_db_vs_pt_storage(tmp_path):
    """Test that DB and PT storage methods preserve the same information."""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    th.manual_seed(42)

    # Create mock data
    mock_cache = MockLatentActivationCache(
        num_features=10, 
        num_sequences=5, 
        seq_length=10, 
        sparsity=0.1
    )

    # Generate example data
    quantile_examples, all_sequences = compute_quantile_activating_examples(
        mock_cache,
        quantiles=[0.25, 0.5, 0.75],
        min_threshold=1e-4,
        n=5,
        test=True
    )

    # Save both formats
    pt_path = tmp_path / "test_examples.pt"
    db_path = tmp_path / "test_examples.db"
    
    # Save PT format
    th.save((quantile_examples, all_sequences), pt_path)
    
    # Save DB format
    quantile_examples_to_db(quantile_examples, all_sequences, db_path)

    # Load PT data
    pt_data, pt_sequences = th.load(pt_path, weights_only=False)
    
    # Load DB data
    db = QuantileExamplesDB(db_path)

    # Compare basic statistics
    pt_features = set()
    for q_idx in pt_data:
        pt_features.update(pt_data[q_idx].keys())
    diff = set(pt_features) ^ set(db.keys())
    for feat_idx in diff:
        for q_idx in pt_data:
            assert pt_data[q_idx][feat_idx] == [], f"Feature {feat_idx} is not in DB: {pt_data[q_idx][feat_idx]}"

    # assert pt_features == set(db.keys()), f"Different feature indices: {pt_features ^ set(db.keys())}"
    # assert len(pt_features) == len(db), f"Different number of features: {len(pt_features)} != {len(db)}"

    # Compare examples for each feature
    for feat_idx in pt_features:
        # Collect all examples for this feature from PT data
        if feat_idx in diff:
            continue
        pt_examples = []
        for q_idx in pt_data:
            if feat_idx in pt_data[q_idx]:
                for activation, seq_idx in pt_data[q_idx][feat_idx]:
                    pt_examples.append((activation, list(pt_sequences[seq_idx])))
        
        # Sort by activation for comparison
        pt_examples.sort(key=lambda x: x[0], reverse=True)
        
        # Get DB examples
        db_examples = db[feat_idx]
        
        # Compare lengths
        assert len(pt_examples) == len(db_examples), \
            f"Different number of examples for feature {feat_idx}"
        
        # Compare each example
        for (pt_act, pt_tokens), (db_act, db_tokens, _) in zip(pt_examples, db_examples):
            assert np.isclose(pt_act, db_act), \
                f"Different activation values for feature {feat_idx}"
            assert pt_tokens == db_tokens, \
                f"Different token sequences for feature {feat_idx}"
