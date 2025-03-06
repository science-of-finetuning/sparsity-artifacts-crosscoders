import sys
from pathlib import Path
import torch as th
import pytest
import random
import numpy as np
import pandas as pd

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.per_token_stats import ActivationStats, EPSILON


# class MockActivationData:
#     def __init__(
#         self, num_features=100, seq_length=10, sparsity=0.1, num_groups=3, device=None
#     ):
#         if device is None:
#             device = th.device("cuda" if th.cuda.is_available() else "cpu")
#         self.num_features = num_features
#         self.seq_length = seq_length
#         self.sparsity = sparsity

#         # Generate random max activations for each feature
#         self.max_activations = th.rand(num_features, device=device)

#         # Generate random dense activations
#         self.dense_activations = (
#             th.rand(1, seq_length, num_features, device=device)
#             * 0.8
#             * self.max_activations.unsqueeze(0).unsqueeze(0)
#         )

#         # Create random group masks
#         self.group_masks = {}
#         remaining_mask = th.ones(1, seq_length, dtype=th.bool, device=device)
#         for group in list(TOKEN_GROUPS)[:-1]:  # Leave last group for remaining tokens
#             group_mask = th.zeros(1, seq_length, dtype=th.bool, device=device)
#             num_tokens = random.randint(1, max(2, seq_length // num_groups))
#             available_positions = remaining_mask.nonzero()[:, 1]
#             if len(available_positions) > 0:
#                 selected_positions = available_positions[
#                     th.randperm(len(available_positions))[:num_tokens]
#                 ]
#                 group_mask[0, selected_positions] = True
#                 remaining_mask = remaining_mask & ~group_mask
#             self.group_masks[group] = group_mask
#         # Assign remaining tokens to last group
#         self.group_masks[TOKEN_GROUPS[-1]] = remaining_mask

#         # Convert dense to sparse
#         nonzero_mask = self.dense_activations.abs() > 1e-6
#         indices = nonzero_mask.nonzero()
#         values = self.dense_activations[nonzero_mask]

#         # Keep batch dimension for sparse indices
#         self.sparse_indices = th.stack(
#             [indices[:, 0], indices[:, 1]], dim=1
#         )  # Keep batch_idx and seq_idx
#         self.sparse_values = values


def compare_stats(stats1, stats2, rtol=1e-5):
    """Compare two ComputedActivationStats objects."""
    # Compare token counts
    assert stats1.token_counts == stats2.token_counts, "Token counts don't match"

    # Compare stats dataframes
    pd.testing.assert_frame_equal(
        stats1.stats.sort_index(),
        stats2.stats.sort_index(),
        rtol=rtol,
        check_dtype=False,  # Allow for some dtype differences
    )


@pytest.mark.parametrize("seed", [42, 123, 456, None, None])
@pytest.mark.parametrize(
    "num_values,num_tokens,num_latents",
    [(800, 500, 20), (1000, 10, 500), (10, 700, 400)],
)
def test(num_values, num_tokens, num_latents, seed, device="cpu"):
    if seed is None:
        seed = th.randint(0, 10000, ()).to(device)
        # seed = 8308
        print(f"seed:{seed}")
    th.manual_seed(seed)
    # Create test data
    bucket_edges = th.tensor([0.1, 1, 2, 4], device=device)
    values = th.rand(num_values).to(device) * 5
    rnd_tokens = th.randint(0, num_tokens, (num_values * 30,)).to(device)
    rnd_features = th.randint(0, num_latents, (num_values * 30,)).to(device)
    indices = th.stack([rnd_tokens, rnd_features], dim=1).unique(dim=0)[:num_values]
    i = 2
    while len(indices) < num_values:
        print("resampling")
        rnd_tokens = th.randint(0, num_tokens, (num_values * 30 * i,)).to(device)
        rnd_features = th.randint(0, num_latents, (num_values * 30 * i,)).to(device)
        indices = th.stack([rnd_tokens, rnd_features], dim=1).unique(dim=0)[:num_values]
        i += 1
    group_mask = th.randint(0, 2, (num_tokens,), dtype=th.bool).to(device)
    neps = 0
    for i in range(num_values):
        if group_mask[indices[i][0]]:
            values[i] = 1e-9
            neps += 1
        if neps > 4:
            break
    dense = th.zeros(num_tokens, num_latents, device=device)
    dense[indices[:, 0], indices[:, 1]] = values
    expected_means = th.zeros(
        1, num_latents, len(bucket_edges) + 1, dtype=th.float64, device=device
    )  # 1 group, 4 latent, 5 buckets
    expected_counts = th.zeros(
        1, num_latents, len(bucket_edges) + 1, dtype=th.int64, device=device
    )
    for val_, tok_idx, f_idx in zip(values, indices[:, 0], indices[:, 1]):
        if val_ < EPSILON or not group_mask[tok_idx]:
            continue
        elif val_ < 0.1:
            expected_means[0, f_idx, 0] += val_
            expected_counts[0, f_idx, 0] += 1
        elif val_ < 1:
            expected_means[0, f_idx, 1] += val_
            expected_counts[0, f_idx, 1] += 1
        elif val_ < 2:
            expected_means[0, f_idx, 2] += val_
            expected_counts[0, f_idx, 2] += 1
        elif val_ < 4:
            expected_means[0, f_idx, 3] += val_
            expected_counts[0, f_idx, 3] += 1
        else:
            expected_means[0, f_idx, 4] += val_
            expected_counts[0, f_idx, 4] += 1

    # Compute means
    for bucket in range(5):
        expected_means[0, :, bucket] /= expected_counts[0, :, bucket].float()

    stats_sparse = ActivationStats(
        num_latents,
        max_activations=th.ones(num_latents, device=device),
        device=device,
        token_groups=["test"],
        bucket_edges=bucket_edges,
    )
    stats_sparse.update((indices, values), group_mask, "test")

    stats_dense = ActivationStats(
        num_latents,
        max_activations=th.ones(num_latents, device=device),
        device=device,
        token_groups=["test"],
        bucket_edges=bucket_edges,
    )
    stats_dense.update(dense.unsqueeze(0), group_mask.unsqueeze(0), "test")
    mismatch = False
    for stats, name in [(stats_sparse, "sparse"), (stats_dense, "dense")]:
        print(f"Testing {name} stats...")
        if not th.allclose(stats._means, th.nan_to_num(expected_means, 0)):
            print(f"Expected means: {expected_means}")
            print(f"Actual means: {stats._means}")
            mismatch = True
            print(f"Values: {values}")
            print(f"Mask: {group_mask}")
            print(f"Indices: {indices}")
        # Plot histogram of differences between expected and actual means
        means_diff = stats._means - th.nan_to_num(expected_means, 0)
        if means_diff.abs().max() > 1e-6:
            print(f"Max difference in means: {means_diff.max().item():.2e}")
            print(f"Mean difference in means: {means_diff.mean().item():.2e}")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.hist(means_diff.cpu().numpy().flatten(), bins=50)
            plt.yscale("log")
            plt.xlabel("Difference")
            plt.ylabel("Count")
            plt.title(f"Distribution of Differences in Means ({name}) seed:{seed}")
            plt.show()
            for lat in range(num_latents):
                for buck in range(5):
                    if abs(means_diff[0, lat, buck]) > 1e-6:
                        print(f"Latent {lat}, bucket {buck}:")
                        print(f"Expected: {expected_means[0, lat, buck]:.2e}")
                        print(f"Actual: {stats._means[0, lat, buck]:.2e}")
                        print(f"Diff: {means_diff[0, lat, buck]:.2e}")
        if not th.allclose(stats._counts, expected_counts):
            print(f"Expected counts: {expected_counts}")
            print(f"Actual counts: {stats._counts}")


# def test_sparse_dense_consistency(seed):
#     """Test that sparse and dense inputs produce the same results."""
#     # Set random seeds for reproducibility
#     random.seed(seed)
#     np.random.seed(seed)
#     th.manual_seed(seed)

#     # Create mock data
#     mock_data = MockActivationData(num_features=100, seq_length=20, sparsity=0.1)

#     device = th.device("cuda" if th.cuda.is_available() else "cpu")

#     # Initialize stats objects for both dense and sparse
#     dense_stats = ActivationStats(
#         mock_data.num_features, mock_data.max_activations, device=device
#     )

#     sparse_stats = ActivationStats(
#         mock_data.num_features, mock_data.max_activations, device=device
#     )
#     assert (
#         mock_data.dense_activations.dim() == 3
#     ), f"Dense activations has no batch dim? {mock_data.dense_activations.shape}"
#     # Update stats with dense data
#     for group_name, group_mask in mock_data.group_masks.items():
#         dense_stats.update(
#             mock_data.dense_activations.to(device),
#             group_mask.to(device),
#             group_name,
#         )

#     # Update stats with sparse data
#     for group_name, group_mask in mock_data.group_masks.items():
#         sparse_stats.update(
#             (mock_data.sparse_indices.to(device), mock_data.sparse_values.to(device)),
#             group_mask.to(device),
#             group_name,
#         )

#     # Compute final stats
#     dense_computed_stats = dense_stats.finish()
#     sparse_computed_stats = sparse_stats.finish()

#     # Compare results
#     compare_stats(dense_computed_stats, sparse_computed_stats)


# def test_edge_cases():
#     """Test edge cases like empty sequences or all zeros."""
#     device = th.device("cuda" if th.cuda.is_available() else "cpu")

#     # Test with empty sequences
#     mock_data = MockActivationData(num_features=10, seq_length=0)

#     dense_stats = ActivationStats(
#         mock_data.num_features, mock_data.max_activations, device
#     )
#     sparse_stats = ActivationStats(
#         mock_data.num_features, mock_data.max_activations, device
#     )
#     assert (
#         mock_data.dense_activations.dim() == 3
#     ), f"Dense activations has no batch dim? {mock_data.dense_activations.shape}"

#     for group_name, group_mask in mock_data.group_masks.items():
#         dense_stats.update(
#             mock_data.dense_activations.to(device),
#             group_mask.to(device),
#             group_name,
#         )
#         sparse_stats.update(
#             (mock_data.sparse_indices.to(device), mock_data.sparse_values.to(device)),
#             group_mask.to(device),
#             group_name,
#         )

#     compare_stats(dense_stats.finish(), sparse_stats.finish())

#     # Test with all zero activations
#     mock_data = MockActivationData(num_features=10, seq_length=5)
#     mock_data.dense_activations.zero_()
#     mock_data.sparse_values.zero_()

#     dense_stats = ActivationStats(
#         mock_data.num_features, mock_data.max_activations, device
#     )
#     sparse_stats = ActivationStats(
#         mock_data.num_features, mock_data.max_activations, device
#     )
#     assert (
#         mock_data.dense_activations.dim() == 3
#     ), f"Dense activations has no batch dim? {mock_data.dense_activations.shape}"

#     for group_name, group_mask in mock_data.group_masks.items():
#         dense_stats.update(
#             mock_data.dense_activations.to(device),
#             group_mask.to(device),
#             group_name,
#         )
#         sparse_stats.update(
#             (mock_data.sparse_indices.to(device), mock_data.sparse_values.to(device)),
#             group_mask.to(device),
#             group_name,
#         )

#     compare_stats(dense_stats.finish(), sparse_stats.finish())
