"""
This script compute the frequency of all features on different kind of tokens, and the rate at which they are activated on each token group.
Token groups are:
- Control tokens
- Control tokens 1...10
- Non-control tokens but bos
- Assistant mask token
- User tokens
- bos
"""

import sys

sys.path.append(".")

from argparse import ArgumentParser
from pathlib import Path
import json
from dataclasses import dataclass

from nnterp import load_model
from nnterp.nnsight_utils import get_layer_output, get_layer
import torch as th
from huggingface_hub import hf_hub_download
import pandas as pd
import einops
from dictionary_learning import CrossCoder
import numpy as np
from tqdm import tqdm, trange
from datasets import load_dataset

from utils import tokenize_with_ctrl_ids, chat_template


def remove_bos(mask):
    mask[:, 0] = False
    return mask


# Define bucket boundaries
BUCKET_EDGES = [0.1, 0.4, 0.7]
EPSILON = 1e-8
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
    def load(cls, path, index_cols=["token_group", "feature", "bucket"]):
        """Load stats from a CSV file with the specified index columns"""
        stats_df = pd.read_csv(path / "stats.csv", index_col=index_cols)
        with open(path / "counts.json", "r") as f:
            token_counts = json.load(f)
        return cls(stats_df, token_counts)

    def save(self, path):
        self.stats.to_csv(path / "stats.csv")
        with open(path / "counts.json", "w") as f:
            json.dump(self.token_counts, f)


class ActivationStats:
    def __init__(
        self,
        num_features,
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
            (len(token_groups), num_features, self.num_buckets),
            dtype=th.int64,
            device=device,
        )
        self._means = th.zeros(
            (len(token_groups), num_features, self.num_buckets),
            dtype=th.float64,
            device=device,
        )
        self._maxs = th.zeros(
            (len(token_groups), num_features, self.num_buckets),
            dtype=th.float32,
            device=device,
        )
        self._group_to_idx = {group: idx for idx, group in enumerate(token_groups)}
        self.token_counts = {t: 0 for t in token_groups}

    @th.no_grad()
    def update(self, activations, group_mask, group_name):
        # activations: tensor of shape (batch, seq, features)
        # group_mask: boolean tensor of shape (batch, seq)
        assert activations.shape[0] == group_mask.shape[0]
        assert activations.shape[1] == group_mask.shape[1]
        assert group_mask.dim() == 2
        group_activations = activations[group_mask]  # shape: (n_tokens, features)
        self.token_counts[group_name] += len(group_activations)

        if len(group_activations) == 0:
            return

        # Compute buckets on GPU
        rel_activations = group_activations / self.max_activations
        buckets = th.bucketize(rel_activations, self.bucket_edges)
        buckets[group_activations < EPSILON] = -1

        group_idx = self._group_to_idx[group_name]

        for bucket_idx in range(self.num_buckets):
            bucket_mask = buckets == bucket_idx  # shape: (n_tokens, features)
            assert bucket_mask.shape == group_activations.shape
            counts = bucket_mask.sum(dim=0)  # shape: (features,)
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
            "feature": np.tile(
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

        stats = pd.DataFrame(data).set_index(["token_group", "feature", "bucket"])
        return ComputedActivationStats(stats, self.token_counts)


@th.no_grad()
def main(
    base_model,
    it_model,
    crosscoder: CrossCoder,
    crosscoder_device,
    dataset,
    max_activations,
    save_path,
    layer=13,
    max_num_tokens=10_000_000,
    batch_size=8,
    test=False,
):
    stats = ActivationStats(
        crosscoder.dict_size,
        max_activations,
        device=crosscoder_device,
    )

    def get_feature(batch):
        with base_model.trace(batch):
            base_acts = (
                get_layer_output(base_model, layer).to(crosscoder_device).save()
            )  # (batch, seq_len, d_model)
            get_layer_output(base_model, layer).stop()
        with it_model.trace(batch):
            it_acts = (
                get_layer_output(it_model, layer).to(crosscoder_device).save()
            )  # (batch, seq_len, d_model)
            get_layer_output(it_model, layer).stop()
        cc_input = th.stack([base_acts, it_acts], dim=2).float()  # b, seq, 2, d
        cc_input = einops.rearrange(cc_input, "b s m d -> (b s) m d")
        cc_acts = crosscoder.get_activations(cc_input)
        cc_acts = einops.rearrange(cc_acts, "(b s) f -> b s f", b=it_acts.shape[0])
        del cc_input, base_acts, it_acts
        th.cuda.empty_cache()
        return cc_acts

    num_tokens = 0
    max_num_tokens = max_num_tokens if not test else 100_000
    pbar = tqdm(total=max_num_tokens, desc="Processing tokens")
    if test:
        dataset = dataset[:8]
    for i in trange(0, len(dataset), batch_size):
        conv_batch = dataset[i : min(i + batch_size, len(dataset))]
        batch = tokenize_with_ctrl_ids(
            conv_batch,
            it_model.tokenizer,
            return_dict=True,
            return_assistant_tokens_mask=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
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
        cc_acts = get_feature(batch)
        for group_name, mask in all_masks.items():
            stats.update(cc_acts, mask, group_name)
        num_new_tokens = attn_mask.sum().item()
        num_tokens += num_new_tokens
        pbar.update(num_new_tokens)
        if num_tokens >= max_num_tokens:
            break

    computed_stats = stats.finish()
    if save_path is not None:
        computed_stats.save(save_path)
    return computed_stats


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base-device", "-b", type=str, default="auto")
    parser.add_argument("--it-device", "-i", type=str, default="auto")
    parser.add_argument("--crosscoder-device", "-c", type=str, default="cpu")
    parser.add_argument("--test", "-t", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path("results/per_token_stats")
    output_dir.mkdir(exist_ok=True, parents=True)

    repo_id = "Butanium/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04"
    df_path = hf_hub_download(
        repo_id=repo_id, filename="feature_df.csv", repo_type="dataset"
    )
    df = pd.read_csv(df_path, index_col=0)
    max_activations = th.from_numpy(df["max_activation_lmsys"].values)
    base_model = load_model("google/gemma-2-2b", device_map=args.base_device)
    it_model = load_model(
        "google/gemma-2-2b-it",
        tokenizer_kwargs={"padding_side": "right"},
        device_map=args.it_device,
    )
    it_model.tokenizer.chat_template = chat_template
    crosscoder = CrossCoder.from_pretrained(
        "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
        from_hub=True,
        device=args.crosscoder_device,
    )
    dataset = load_dataset(
        "jkminder/lmsys-chat-1m-gemma-formatted", split="validation"
    )["conversation"]
    stats = main(
        base_model,
        it_model,
        crosscoder,
        args.crosscoder_device,
        dataset,
        max_activations,
        save_path=output_dir,
        batch_size=args.batch_size,
        test=args.test,
    )

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
        stats.stats[stats.stats.index.get_level_values("token_group") == "ctrl_tokens"]
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
