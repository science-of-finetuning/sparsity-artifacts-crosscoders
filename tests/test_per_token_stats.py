import sys
from pathlib import Path
import torch as th
import pytest
import random
import numpy as np
import pandas as pd
from collections import defaultdict

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


def create_toy_data(num_values, num_tokens, num_latents, seed=None, device="cpu"):
    if seed is None:
        seed = th.randint(0, 10000, ()).to(device)
        # seed = 8308
        print(f"seed:{seed}")
    th.manual_seed(seed)
    # Create test data
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
    return indices, values


def check_array_equality(actual, expected, name):
    if not np.allclose(actual, expected, equal_nan=True):
        err_str = ""
        for i, (a, b) in enumerate(zip(actual, expected)):
            if a == b or (np.isnan(a) and np.isnan(b)):
                continue
            err_str += f"{i}: {a} != {b} - "
            print(f"{i}: {a} != {b}")
        if not err_str:
            print("No actual differences found")
        assert False, f"{name} mismatch: " + err_str


@pytest.mark.parametrize("seed", [42, 123, 456, None, None])
@pytest.mark.parametrize(
    "num_values,num_tokens,num_latents",
    [(800, 500, 20), (1000, 10, 500), (10, 700, 400)],
)
def test_sparse_and_dense_stats(
    num_values, num_tokens, num_latents, seed, device="cpu"
):
    bucket_edges = th.tensor([0.1, 1, 2, 4], device=device)
    indices, values = create_toy_data(num_values, num_tokens, num_latents, seed, device)
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
    assert stats_dense.token_counts["test"] == group_mask.sum().item()
    assert stats_sparse.token_counts["test"] == group_mask.sum().item()
    for stats, name in [(stats_sparse, "sparse"), (stats_dense, "dense")]:
        print(f"Testing {name} stats...")

        # Plot histogram of differences between expected and actual means
        means_diff = stats._means - th.nan_to_num(expected_means, 0)
        if means_diff.abs().max() > 1e-6:
            print(f"Max difference in means: {means_diff.max().item():.2e}")
            print(f"Mean difference in means: {means_diff.mean().item():.2e}")
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
            assert False, f"Counts mismatch for {name}"
        if not th.allclose(stats._means, th.nan_to_num(expected_means, 0)):
            print(f"Expected means: {expected_means}")
            print(f"Actual means: {stats._means}")
            print(f"Values: {values}")
            print(f"Mask: {group_mask}")
            print(f"Indices: {indices}")
            assert False, f"Means mismatch for {name}"


@pytest.mark.parametrize("seed", [42, 123, 456, None, None])
@pytest.mark.parametrize(
    "num_values,num_tokens,num_latents",
    [(800, 500, 20), (1000, 10, 500), (10, 700, 400)],
)
def test_compute_global_stats(num_values, num_tokens, num_latents, seed, device="cpu"):
    if seed is None:
        seed = th.randint(0, 10000, ()).to(device)
        print(f"seed:{seed}")
    indices, values = create_toy_data(num_values, num_tokens, num_latents, seed, device)
    token_groups = [
        "ctrl_tokens",
        "non_ctrl_tokens",
        "assistant_tokens",
        "user_tokens",
        "bos",
        "all_tokens",
    ]
    group_to_idx = {group: i for i, group in enumerate(token_groups)}
    masks = {}
    th.manual_seed(seed + 1)
    masks["ctrl_tokens"] = th.randint(0, 2, (num_tokens,), dtype=th.bool).to(device)
    masks["non_ctrl_tokens"] = ~masks["ctrl_tokens"]
    masks["assistant_tokens"] = th.randint(0, 2, (num_tokens,), dtype=th.bool).to(
        device
    )
    masks["user_tokens"] = ~masks["assistant_tokens"]
    masks["bos"] = th.zeros(num_tokens, dtype=th.bool).to(device)
    masks["bos"][0] = True
    masks["all_tokens"] = th.ones(num_tokens, dtype=th.bool).to(device)
    total_counts = th.tensor(
        [masks[group_name].sum().item() for group_name in token_groups],
        device=device,
    )
    counts = th.zeros((len(token_groups), num_latents), dtype=th.int64)
    for group_name, group_mask in masks.items():
        for i, (pos, latent) in enumerate(indices):
            if group_mask[pos] and values[i] > EPSILON:
                counts[group_to_idx[group_name]][latent] += 1
    frequencies = counts / total_counts.unsqueeze(-1)
    ctrl_percentage = counts[group_to_idx["ctrl_tokens"]] / (
        counts[group_to_idx["ctrl_tokens"]] + counts[group_to_idx["non_ctrl_tokens"]]
    )
    assistant_percentage = counts[group_to_idx["assistant_tokens"]] / (
        counts[group_to_idx["assistant_tokens"]] + counts[group_to_idx["user_tokens"]]
    )
    stats = ActivationStats(
        num_latents,
        max_activations=th.ones(num_latents, device=device),
        device=device,
        token_groups=token_groups + [f"ctrl_token_{i}" for i in range(1, 11)],
        bucket_edges=th.tensor([0.1, 1, 2, 4], device=device),
    )
    stats.update((indices, values), masks["ctrl_tokens"], "ctrl_tokens")
    stats.update((indices, values), masks["non_ctrl_tokens"], "non_ctrl_tokens")
    stats.update((indices, values), masks["assistant_tokens"], "assistant_tokens")
    stats.update((indices, values), masks["user_tokens"], "user_tokens")
    stats.update((indices, values), masks["bos"], "bos")
    stats.update((indices, values), masks["all_tokens"], "all_tokens")
    check_array_equality(
        stats._counts.sum(dim=-1)[: len(token_groups)], counts.cpu().numpy(), "Counts"
    )
    act_stats = stats.finish().compute_latent_stats().loc[(-1,)]
    assert np.allclose(
        act_stats["lmsys_ctrl_%"],
        ctrl_percentage.double().cpu().numpy(),
        equal_nan=True,
    )
    check_array_equality(
        act_stats["lmsys_assistant_%"],
        assistant_percentage.double().cpu().numpy(),
        "Assistant percentages",
    )
    
    check_array_equality(
        act_stats["lmsys_freq"],
        frequencies[group_to_idx["all_tokens"]].double().cpu().numpy(),
        "Frequency",
    )
    for group_name in token_groups:
        if group_name == "all_tokens":
            continue
        df_name = group_name.replace("_tokens", "")
        check_array_equality(
            act_stats[f"lmsys_{df_name}_freq"],
            frequencies[group_to_idx[group_name]].double().cpu().numpy(),
            f"Frequency for {group_name}",
        )