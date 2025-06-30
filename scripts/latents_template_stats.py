"""
This script compute the frequency of all latents on different kind of tokens, and the rate at which they are activated on each token group.
Token groups are:
- Control tokens
- Control tokens 1...10
- Non-control tokens but bos
- Assistant mask token
- User tokens
- bos
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
import json
from dataclasses import dataclass

import torch as th
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import plotly.express as px
from transformers import AutoTokenizer

sys.path.append(".")
from tools.utils import (
    tokenize_with_ctrl_ids,
    patch_tokenizer,
    gemma_tokens_to_conv,
    load_latent_df,
    push_latent_df,
)
from tools.cache_utils import LatentActivationCache


def remove_bos(mask):
    mask[:, 0] = False
    return mask


# Define bucket boundaries
BUCKET_EDGES = [0.1, 0.4, 0.7]
EPSILON = 1e-3
TOKEN_GROUPS = (
    ["ctrl_tokens"]
    + [f"ctrl_token_{i}" for i in range(1, 11)]
    + ["non_ctrl_tokens", "assistant_tokens", "user_tokens", "bos"]
)


@dataclass
class ComputedActivationStats:
    stats: pd.DataFrame
    token_counts: dict

    @classmethod
    def load(cls, path, index_cols=["token_group", "latent", "bucket"]):
        """Load stats from a CSV file with the specified index columns"""
        stats_df = pd.read_csv(path / "stats.csv", index_col=index_cols)
        with open(path / "counts.json", "r") as f:
            token_counts = json.load(f)
        return cls(stats_df, token_counts)

    def save(self, path):
        self.stats.to_csv(path / "stats.csv")
        with open(path / "counts.json", "w") as f:
            json.dump(self.token_counts, f)

    def get_group_stats(self, group_name):
        """Helper to get basic stats for a token group"""
        group_stats = self.stats.xs(group_name, level="token_group")
        # First multiply mean by count for the entire dataframe
        group_stats["weighted_mean"] = group_stats["mean"].astype(
            np.float64
        ) * group_stats["nonzero count"].astype(np.float64)
        group_stats["weighted_mean"] = group_stats["weighted_mean"].fillna(0)
        # Now we can use fast sum aggregation
        means = (
            group_stats.groupby(["latent", "bucket"])["weighted_mean"]
            .agg("sum")
            .unstack(level="bucket")
            .values
        ).sum(axis=1)

        counts = (
            group_stats.groupby(["latent", "bucket"])["nonzero count"]
            .agg("sum")
            .unstack(level="bucket")
            .values
        ).sum(axis=1)

        # Vectorized division
        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            means = np.divide(means, counts.astype(np.float64))
            means[counts == 0] = np.nan
        group_stats.drop(columns=["weighted_mean"], inplace=True)
        maxs = (
            group_stats.groupby(["latent", "bucket"])["max"]
            .agg("max")
            .unstack(level="bucket")
            .values
        ).copy()
        nan_maxs = np.isnan(maxs).all(axis=1)
        maxs[nan_maxs] = 0
        maxs = np.nanmax(maxs, axis=1)
        maxs[nan_maxs] = np.nan
        assert (np.isnan(maxs) == np.isnan(means)).all()
        return {
            "stats": group_stats,
            "count": counts,
            "mean": means,
            "max": maxs,
            "total_count": self.token_counts[group_name],
        }

    @pd.option_context("mode.copy_on_write", True)
    def compute_latent_stats(self):
        """
        Compute statistics for different token groups and their interactions.
        """
        # Get stats for each group
        groups_data = {
            "ctrl": self.get_group_stats("ctrl_tokens"),
            "non_ctrl": self.get_group_stats("non_ctrl_tokens"),
            "bos": self.get_group_stats("bos"),
            "assistant": self.get_group_stats("assistant_tokens"),
            "user": self.get_group_stats("user_tokens"),
            **{
                f"ctrl_{i}": self.get_group_stats(f"ctrl_token_{i}")
                for i in range(1, 11)
            },
        }

        # Compute per-bucket stats
        results = []
        for bucket in range(len(BUCKET_EDGES) + 1):
            bucket_stats = {}

            # Get bucket-specific data for each group
            bucket_data = {
                name: group["stats"].xs(bucket, level="bucket")
                for name, group in groups_data.items()
            }

            # Compute frequencies for all groups
            for name, group in groups_data.items():
                bucket_stats[f"lmsys_{name}_freq"] = (
                    bucket_data[name]["nonzero count"] / group["total_count"]
                )

            # Compute percentages relative to total activations for different group pairs
            group_pairs = [
                ("ctrl", "non_ctrl"),  # Control vs non-control
                ("assistant", "user"),  # Assistant vs user
            ]

            for group1, group2 in group_pairs:
                total_acts = sum(
                    bucket_data[name]["nonzero count"] for name in [group1, group2]
                )
                total_acts[total_acts == 0] = np.nan

                bucket_stats[f"lmsys_{group1}_%"] = (
                    bucket_data[group1]["nonzero count"] / total_acts
                )

            # Special case for BOS: percentage relative to all tokens
            total_acts_with_bos = sum(
                bucket_data[name]["nonzero count"]
                for name in ["ctrl", "non_ctrl", "bos"]
            )
            total_acts_with_bos[total_acts_with_bos == 0] = np.nan
            bucket_stats["lmsys_bos_%"] = (
                bucket_data["bos"]["nonzero count"] / total_acts_with_bos
            )

            # Compute percentages for individual ctrl tokens relative to all ctrl tokens
            total_ctrl_acts = bucket_data["ctrl"]["nonzero count"]
            total_ctrl_acts[total_ctrl_acts == 0] = np.nan
            for i in range(1, 11):
                bucket_stats[f"ctrl_{i}_%"] = (
                    bucket_data[f"ctrl_{i}"]["nonzero count"] / total_ctrl_acts
                )

            # Include means and maxs for all groups
            for name in groups_data:
                bucket_stats[f"{name}_mean"] = bucket_data[name]["mean"]
                bucket_stats[f"{name}_max"] = bucket_data[name]["max"]

            bucket_df = pd.DataFrame(bucket_stats)
            bucket_df.index.name = "latent"
            results.append(bucket_df)

        # Combine all buckets
        per_bucket_stats = pd.concat(
            results, keys=range(len(BUCKET_EDGES) + 1), names=["bucket"]
        )
        per_bucket_stats.to_csv("results/per_token_stats/per_bucket_stats.csv")
        results[0].to_csv("results/per_token_stats/per_bucket_stats_0.csv")

        # Compute global stats
        global_stats = {}

        # Total activation counts for different group pairs
        group_pairs_totals = {
            "ctrl_non_ctrl": sum(
                group["count"]
                for name, group in groups_data.items()
                if name in ["ctrl", "non_ctrl"]
            ).astype(np.float64),
            "assistant_user": sum(
                group["count"]
                for name, group in groups_data.items()
                if name in ["assistant", "user"]
            ).astype(np.float64),
        }
        for total in group_pairs_totals.values():
            total[total == 0] = np.nan

        # Compute global stats for all groups
        for name, group in groups_data.items():
            global_counts = group["count"]
            global_stats[f"lmsys_{name}_freq"] = global_counts / group["total_count"]
            global_stats[f"{name}_mean"] = group["mean"]
            global_stats[f"{name}_max"] = group["max"]

            # Compute percentages for relevant groups
            if name in ["ctrl", "non_ctrl"]:
                global_stats[f"lmsys_{name}_%"] = (
                    global_counts / group_pairs_totals["ctrl_non_ctrl"]
                )
            elif name in ["assistant", "user"]:
                global_stats[f"lmsys_{name}_%"] = (
                    global_counts / group_pairs_totals["assistant_user"]
                )
            elif name == "bos":
                total_acts_with_bos = sum(
                    groups_data[g]["count"] for g in ["ctrl", "non_ctrl", "bos"]
                ).astype(np.float64)
                total_acts_with_bos[total_acts_with_bos == 0] = np.nan
                global_stats["lmsys_bos_%"] = global_counts / total_acts_with_bos
            elif name.startswith("ctrl_"):
                ctrl_counts = groups_data["ctrl"]["count"].astype(np.float64)
                ctrl_counts[ctrl_counts == 0] = np.nan
                global_stats[f"lmsys_{name}_%"] = global_counts / ctrl_counts

        # Compute total frequency by summing raw counts and dividing by total count
        total_count = (
            groups_data["ctrl"]["total_count"] + groups_data["non_ctrl"]["total_count"]
        )
        total_raw_counts = (
            groups_data["ctrl"]["count"] + groups_data["non_ctrl"]["count"]
        )
        global_stats["lmsys_freq"] = total_raw_counts / total_count
        # Create global stats dataframe
        global_stats = pd.DataFrame(global_stats)
        global_stats.index.name = "latent"
        global_stats["bucket"] = -1
        global_stats = global_stats.reset_index().set_index(["bucket", "latent"])

        # Combine per-bucket and global stats
        all_stats = pd.concat([per_bucket_stats, global_stats])
        return all_stats


class ActivationStats:
    def __init__(
        self,
        num_latents,
        max_activations,
        device,
        token_groups=TOKEN_GROUPS,
        bucket_edges=BUCKET_EDGES,
    ):
        self.max_activations = max_activations.to(
            device
        )  # Ensure max_activations is on GPU
        self.bucket_edges = th.tensor(bucket_edges, device=device)
        self.num_buckets = len(bucket_edges) + 1
        self.token_groups = token_groups

        # Pre-allocate tensors on GPU
        self._counts = th.zeros(
            (len(token_groups), num_latents, self.num_buckets),
            dtype=th.int64,
            device=device,
        )
        self._means = th.zeros(
            (len(token_groups), num_latents, self.num_buckets),
            dtype=th.float64,
            device=device,
        )
        self._maxs = th.zeros(
            (len(token_groups), num_latents, self.num_buckets),
            dtype=th.float32,
            device=device,
        )
        self._group_to_idx = {group: idx for idx, group in enumerate(token_groups)}
        self.token_counts = {t: 0 for t in token_groups}

    @th.no_grad()
    def update_sparse(self, sparse_acts, group_mask, group_name):
        self.token_counts[group_name] += group_mask.sum().item()
        (indices, activations) = sparse_acts
        noise_mask = activations > EPSILON
        indices = indices[noise_mask]
        activations = activations[noise_mask]
        assert indices.dim() == 2, "batching not supported yet"
        group_mask = group_mask.squeeze(0)
        assert group_mask.dtype == th.bool
        assert (
            group_mask.dim() == 1
        ), "group mask must be 1D, batching not supported yet"
        sorted_latents, sorted_indices = th.sort(indices[:, 1])
        sorted_token_indices = indices[sorted_indices, 0]
        mask = group_mask[sorted_token_indices]
        sorted_latents = sorted_latents[mask]
        activations = activations[sorted_indices][mask]
        rel_acts = activations / self.max_activations[sorted_latents]
        if len(rel_acts) == 0:
            return
        assert mask.dim() == 1, "mask must be 1D, batching not supported yet"
        assert (
            mask.sum().item() == rel_acts.shape[0]
        ), f"Shape mismatch: {mask.sum().item()} <> {rel_acts.shape[0]}"
        active_latents, counts = th.unique_consecutive(
            sorted_latents, return_counts=True
        )
        assert len(active_latents) == len(sorted_latents.unique())
        buckets = th.bucketize(rel_acts, self.bucket_edges)
        group_idx = self._group_to_idx[group_name]
        self._maxs[group_idx, sorted_latents, buckets] = th.maximum(
            self._maxs[group_idx, sorted_latents, buckets], activations
        )
        self._means[group_idx][active_latents] *= self._counts[group_idx][
            active_latents
        ]
        incr_indices = [sorted_latents, buckets]
        self._counts[group_idx].index_put_(
            incr_indices, th.ones_like(sorted_latents), accumulate=True
        )
        assert activations.shape == sorted_latents.shape
        self._means[group_idx].index_put_(
            incr_indices, activations.double(), accumulate=True
        )
        counts = self._counts[group_idx][active_latents]
        nonzero_mask = counts != 0
        self._means[group_idx][active_latents] = th.where(
            nonzero_mask,
            self._means[group_idx][active_latents] / counts,
            self._means[group_idx][active_latents],
        )
        # self._means[group_idx][active_latents][counts != 0] /= counts[counts != 0]
        # add 1 for each sorted_latent, bucket index
        # create a sum of act values that is filled in the same way using values
        # update means with the new counts and sum of act values

    @th.no_grad()
    def update(self, activations, group_mask, group_name):
        # activations can be either:
        # - a dense tensor of shape (batch, seq, latents)
        # - a tuple (indices, values) where:
        #   - indices is a tensor of shape (batch, N, 2) containing (token_idx, latent_idx) pairs
        #   - values is a tensor of shape (batch, N) containing activation values
        # group_mask: boolean tensor of shape (batch, seq)

        if isinstance(activations, th.Tensor):
            # Dense case
            assert (
                activations.shape[0] == group_mask.shape[0]
            ), f"Shape 0 mismatch: {activations.shape} <> {group_mask.shape}"
            assert (
                activations.shape[1] == group_mask.shape[1]
            ), f"Shape 1 mismatch: {activations.shape} <> {group_mask.shape}"
            assert (
                group_mask.dim() == 2
            ), f"Group mask has no batch dim? {group_mask.shape}"
            assert (
                activations.dim() == 3
            ), f"Activations has no batch dim? {activations.shape}"
            group_activations = activations[group_mask]  # shape: (n_tokens, latents)
        else:
            assert isinstance(
                activations, tuple
            ), f"Unknown activations type: {type(activations)}\n{activations}"
            return self.update_sparse(activations, group_mask, group_name)
        if len(group_activations) == 0:
            return
        self.token_counts[group_name] += len(group_activations)

        # Compute buckets on GPU
        rel_activations = group_activations / self.max_activations
        buckets = th.bucketize(rel_activations, self.bucket_edges)
        buckets[group_activations < EPSILON] = -1

        group_idx = self._group_to_idx[group_name]

        for bucket_idx in range(self.num_buckets):
            bucket_mask = buckets == bucket_idx  # shape: (n_tokens, latents)
            assert bucket_mask.shape == group_activations.shape
            counts = bucket_mask.sum(dim=0)  # shape: (latents,)
            update_mask = counts > 0
            counts = counts[update_mask]  # shape: (ft_to_update,)
            if not update_mask.any():
                continue
            bucket_acts = group_activations[
                :, update_mask
            ]  # shape: (n_tokens, ft_to_update)
            bucket_acts = th.where(
                bucket_mask[:, update_mask], bucket_acts, th.zeros_like(bucket_acts)
            )  # shape: (n_tokens, ft_to_update)
            means = (
                bucket_acts.sum(dim=0).to(th.float64) / counts
            )  # shape: (ft_to_update,)
            maxs = th.where(
                bucket_mask[:, update_mask].any(dim=0),
                bucket_acts.max(dim=0).values,  # shape: (ft_to_update,)
                self._maxs[
                    group_idx, update_mask, bucket_idx
                ],  # shape: (ft_to_update,)
            )

            curr_counts = self._counts[
                group_idx, update_mask, bucket_idx
            ]  # shape: (ft_to_update,)
            new_counts = curr_counts + counts  # shape: (ft_to_update,)

            self._means[group_idx, update_mask, bucket_idx] = (
                curr_counts * self._means[group_idx, update_mask, bucket_idx]
                + counts * means
            ) / new_counts  # shape: (ft_to_update,)

            self._counts[group_idx, update_mask, bucket_idx] = new_counts
            self._maxs[group_idx, update_mask, bucket_idx] = th.maximum(
                self._maxs[group_idx, update_mask, bucket_idx], maxs
            )

    def finish(self):
        # Create a mask for non-zero counts
        zeros_mask = self._counts == 0
        self._means[zeros_mask] = th.nan
        self._maxs[zeros_mask] = th.nan
        data = {
            "token_group": np.repeat(
                list(self._group_to_idx.keys()), self._counts[0].numel()
            ),
            "latent": np.tile(
                np.repeat(range(self._counts.shape[1]), self._counts.shape[2]),
                len(self._group_to_idx),
            ),
            "bucket": np.tile(
                range(self._counts.shape[2]),
                len(self._group_to_idx) * self._counts.shape[1],
            ),
            "nonzero count": self._counts.cpu().numpy().flatten(),
            "mean": self._means.cpu().numpy().flatten(),
            "max": self._maxs.cpu().numpy().flatten(),
        }

        stats = pd.DataFrame(data).set_index(["token_group", "latent", "bucket"])
        return ComputedActivationStats(stats, self.token_counts)


def process_stats(
    stats: ComputedActivationStats,
    crosscoder: str,
    save_path,
    verbose=1,
    test=False,
    df_name="feature_df",
):
    if verbose:
        # compute simple statistics, like bucket frequency
        bucket_freq = stats.stats.groupby("bucket")["nonzero count"].sum()
        bucket_freq = bucket_freq / bucket_freq.sum()
        print(f"Bucket frequency: {bucket_freq.round(2)}")
        # Average activation per token group
        avg_per_group = stats.stats.groupby("token_group")["mean"].mean().round(2)
        print("\nAverage activation per token group:")
        print(avg_per_group)

        # Distribution of activations across buckets for ctrl vs non-ctrl tokens
        ctrl_dist = (
            stats.stats[
                stats.stats.index.get_level_values("token_group") == "ctrl_tokens"
            ]
            .groupby("bucket")["nonzero count"]
            .sum()
            .round(2)
        )
        non_ctrl_dist = (
            stats.stats[
                stats.stats.index.get_level_values("token_group") == "non_ctrl_tokens"
            ]
            .groupby("bucket")["nonzero count"]
            .sum()
            .round(2)
        )
        print("\nActivation distribution - Control tokens:")
        print((ctrl_dist / ctrl_dist.sum()).round(2))
        print("\nActivation distribution - Non-control tokens:")
        print((non_ctrl_dist / non_ctrl_dist.sum()).round(2))

        # Top tokens with highest max activations per group
        print("\nTop 5 highest max activations per group:")
        for group in TOKEN_GROUPS:
            group_stats = stats.stats.xs(group, level="token_group")
            top_5 = group_stats.nlargest(5, "max")
            print(f"\n{group}:")
            print(top_5["max"].round(2))

        # Ratio of high vs low activations per group
        print("\nRatio of high (bucket 2-3) vs low (bucket 0-1) activations:")
        for group in TOKEN_GROUPS:
            group_stats = stats.stats.xs(group, level="token_group")
            high_acts = group_stats[group_stats.index.get_level_values("bucket") >= 2][
                "nonzero count"
            ].sum()
            low_acts = group_stats[group_stats.index.get_level_values("bucket") < 2][
                "nonzero count"
            ].sum()
            if low_acts > 0:
                ratio = high_acts / low_acts
                print(f"{group}: {ratio:.2f}")
    latent_stats = stats.compute_latent_stats()
    latent_stats.to_csv(save_path / "latent_stats.csv")
    # plot histograms of different stats
    # Plot histograms of different stats
    # Create directory for plots
    plot_dir = save_path / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # Plot ctrl frequency distribution
    fig = px.histogram(
        latent_stats.reset_index(),
        x="lmsys_ctrl_freq",
        color="bucket",
        barmode="overlay",
        title="Distribution of Control Token Frequencies",
        labels={
            "lmsys_ctrl_freq": "Control Token Frequency",
            "count": "Number of latents",
            "bucket": "Activation Bucket",
        },
        log_y=True,
    )
    fig.write_html(plot_dir / "ctrl_frequency_dist.html")
    fig.write_image(plot_dir / "ctrl_frequency_dist.png", scale=3)

    # Plot non-ctrl frequency distribution
    fig = px.histogram(
        latent_stats.reset_index(),
        x="lmsys_non_ctrl_freq",
        color="bucket",
        barmode="overlay",
        title="Distribution of Non-Control Token Frequencies",
        labels={
            "lmsys_non_ctrl_freq": "Non-Control Token Frequency",
            "count": "Number of latents",
            "bucket": "Activation Bucket",
        },
        log_y=True,
    )
    fig.write_html(plot_dir / "non_ctrl_frequency_dist.html")
    fig.write_image(plot_dir / "non_ctrl_frequency_dist.png", scale=3)

    # Plot ctrl percentage distribution
    fig = px.histogram(
        latent_stats.reset_index(),
        x="lmsys_ctrl_%",
        color="bucket",
        barmode="overlay",
        title="Distribution of Control Token Percentages",
        labels={
            "lmsys_ctrl_%": "Control Token Percentage",
            "count": "Number of latents",
            "bucket": "Activation Bucket",
        },
        log_y=True,
    )
    fig.write_html(plot_dir / "ctrl_percentage_dist.html")
    fig.write_image(plot_dir / "ctrl_percentage_dist.png", scale=3)

    latent_df = load_latent_df(crosscoder, df_name=df_name)
    new_stats = latent_stats.xs(-1, level="bucket")
    intersection_columns = set(new_stats.columns) & set(latent_df.columns)
    for col in intersection_columns:
        try:
            pd.testing.assert_series_equal(new_stats[col], latent_df[col])
        except Exception as e:
            print(f"Mismatch in {col}: {e}")
    latent_df = latent_df[
        [c for c in latent_df.columns if c not in intersection_columns]
    ]
    # select stats for bucket -1
    new_stats = new_stats.merge(latent_df, left_index=True, right_index=True)
    # Reorder columns to group related metrics
    # fmt: off
    new_stats["lmsys_avg_act"] = new_stats["ctrl_mean"] * new_stats["lmsys_ctrl_%"] + new_stats["non_ctrl_mean"] * (1 - new_stats["lmsys_ctrl_%"])
    ordered_cols = [  
        "tag", "dead", "dec_norm_diff", "base uselessness score", "avg_activation", 
        "lmsys_ctrl_%", "lmsys_bos_%", "lmsys_user_%", "lmsys_assistant_%",
        # Frequencies
        "lmsys_dead", "fw_dead","freq","lmsys_freq","lmsys_ctrl_freq", "lmsys_non_ctrl_freq", "fw_freq", "bos_freq",
        # Mean activations
        "lmsys_avg_act", "ctrl_mean", "non_ctrl_mean", "fw_avg_act",
        # Max activations  
        "lmsys_max_act", "ctrl_max", "non_ctrl_max",
        # Cosine similarities
        "dec_cos_sim", "enc_cos_sim",
        # Norm differences
        "enc_norm_diff","dec_base_norm", "dec_instruct_norm", "enc_base_norm", "enc_instruct_norm", 
    ]
    all_cols =  ordered_cols + [col for col in new_stats.columns if col not in ordered_cols]
    all_cols = [col for col in all_cols if col in new_stats.columns]
    # fmt: on
    new_stats = new_stats[all_cols]
    new_stats.to_csv(save_path / "latent_stats_global.csv")
    print(f"Saved to {save_path / 'latent_stats_global.csv'}")
    if not test:
        push_latent_df(new_stats, crosscoder, confirm=False, filename=df_name)


@th.no_grad()
def compute_latents_template_stats(
    tokenizer,
    crosscoder: str,
    latent_activation_cache: LatentActivationCache,
    max_activations,
    save_path,
    max_num_tokens=1_000_000_000,
    test=False,
    df_name="feature_df",
):
    device = "cuda" if th.cuda.is_available() else "cpu"
    latent_activation_cache.to(device)
    stats = ActivationStats(
        latent_activation_cache.dict_size,
        max_activations,
        device=device,
    )

    num_tokens = 0
    max_num_tokens = max_num_tokens if not test else 100_000
    for i in range(len(latent_activation_cache)):
        if (
            tokenizer.start_of_turn_token_id
            in latent_activation_cache.get_sequence(i)[:2]
        ):
            latent_activation_cache.offset = latent_activation_cache.offset + i
            break
    print(f"Using offset {latent_activation_cache.offset}")
    try:
        # dataloader = DataLoader(
        #     latent_activation_cache,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     # num_workers=16,
        # )
        dataloader = latent_activation_cache
        pbar = trange(len(dataloader), desc="Processing batches")
        for i in pbar:
            tokens = latent_activation_cache.get_sequence(i)
            # Convert tokens to conversation using gemma_tokens_to_conv
            tokens, cc_acts = dataloader[i]
            # convs = [
            #     gemma_tokens_to_conv(sample.tolist(), tokenizer) for sample in tokens
            # ]
            convs = [gemma_tokens_to_conv(tokens.tolist(), tokenizer)]
            batch = tokenize_with_ctrl_ids(
                convs,
                tokenizer,
                return_dict=True,
                return_assistant_tokens_mask=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)
            ctrl_mask = batch["ctrl_mask"]
            ctrl_ids = batch["ctrl_ids"]
            assistant_mask = batch["assistant_masks"]
            attn_mask = batch["attention_mask"].bool()
            user_tokens_mask = remove_bos(attn_mask & ~assistant_mask & ~ctrl_mask)
            bos_mask = th.zeros_like(attn_mask, dtype=th.bool)
            bos_mask[:, 0] = True
            all_masks = {
                "ctrl_tokens": ctrl_mask,
                **{f"ctrl_token_{i}": ctrl_ids == i for i in range(1, 11)},
                "non_ctrl_tokens": remove_bos(attn_mask & ~ctrl_mask),
                "assistant_tokens": assistant_mask,
                "user_tokens": user_tokens_mask,
                "bos": bos_mask,
            }
            # assert (
            #     attn_mask.shape == cc_acts.shape[:-1]
            # ), f"Shape mismatch: {attn_mask.shape} <> {cc_acts.shape[:-1]}"
            # assert (
            #     ctrl_mask.shape == cc_acts.shape[:-1]
            # ), f"Shape mismatch: {ctrl_mask.shape} <> {cc_acts.shape[:-1]}"
            # assert (
            #     assistant_mask.shape == cc_acts.shape[:-1]
            # ), f"Shape mismatch: {assistant_mask.shape} <> {cc_acts.shape[:-1]}"
            # assert (
            #     all_masks["ctrl_token_1"].shape == cc_acts.shape[:-1]
            # ), f"Shape mismatch: {all_masks['ctrl_token_1'].shape} <> {cc_acts.shape[:-1]}"
            for group_name, mask in all_masks.items():
                stats.update(cc_acts, mask, group_name)

            num_new_tokens = attn_mask.sum().item()
            num_tokens += num_new_tokens
            pbar.set_postfix_str(f"Tokens: {num_tokens}")
            if num_tokens >= max_num_tokens:
                break
    finally:
        computed_stats = stats.finish()
        if save_path is not None:
            save_path.mkdir(exist_ok=True)
            computed_stats.save(save_path)
    process_stats(
        computed_stats, crosscoder, save_path, verbose=1, test=test, df_name=df_name
    )
    return computed_stats


if __name__ == "__main__":
    # python scripts/latents_template_stats.py SAE-base-gemma-2-2b-L13-k100-x32-lr1e-04-local-shuffling --latent-activation-cache-path $DATASTORE/latent_activations --df-name "feature_df_from_base" --latent-activation-cache-suffix "from_base"
    parser = ArgumentParser()
    parser.add_argument("crosscoder", type=str)
    parser.add_argument(
        "--latent-activation-cache-path", type=Path, default="./data/latent_activations"
    )
    parser.add_argument("--df-name", type=str, default="feature_df")
    parser.add_argument("--latent-activation-cache-suffix", type=str, default="")
    parser.add_argument("--test", "-t", action="store_true")
    parser.add_argument("--use-precomputed-stats", "--skip", action="store_true")
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    if args.use_precomputed_stats:
        stats = ComputedActivationStats.load(Path("results/latents_template_stats"))
        process_stats(stats, verbose=0, test=args.test)
        exit()

    # Create output directory
    output_dir = Path("results/latents_template_stats") / args.crosscoder
    if args.test:
        output_dir = output_dir / "test"
    if args.name:
        output_dir = output_dir / args.name
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load tokenizer and patch it
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    patch_tokenizer(tokenizer, "gemma-2-2b-it")

    # Load latent activation cache and max activations
    l_act_path = args.latent_activation_cache_path / args.crosscoder
    if args.latent_activation_cache_suffix:
        l_act_path = l_act_path / args.latent_activation_cache_suffix
    latent_activation_cache = LatentActivationCache(l_act_path, expand=False)
    max_activations = latent_activation_cache.max_activations

    stats = compute_latents_template_stats(
        tokenizer,
        args.crosscoder,
        latent_activation_cache,
        max_activations,
        save_path=output_dir,
        test=args.test,
        df_name=args.df_name,
    )
