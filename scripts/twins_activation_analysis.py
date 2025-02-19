from pathlib import Path
import json
import argparse
import sys
import os
import torch as th
from tqdm.auto import tqdm

sys.path.append(".")
from tools.utils import load_latent_df, load_crosscoder, load_activation_dataset


@th.no_grad()
def compute_feature_metrics(
    dataloader,
    crosscoder_name,
    twins_file,
    device="cpu",
    abosulte_max_act=False,
):
    """Compute bucketed activation statistics for twin features"""

    with open(twins_file, "r") as f:
        twins = json.load(f)
    all_latents = sum(twins, [])
    left_latents = [p[0] for p in twins]
    right_latents = [p[1] for p in twins]

    # Load max activations from latent df
    df = load_latent_df(crosscoder_name)
    crosscoder = load_crosscoder(crosscoder_name).to(device)
    if abosulte_max_act:
        max_acts = th.maximum(
            th.from_numpy(df["max_act_train"][left_latents].values),
            th.from_numpy(df["max_act_train"][right_latents].values),
        ).clamp(
            min=1e-6
        )  # shape: (len(twins))
        max_acts = sum([[mact] * 2 for mact in max_acts], [])
        max_acts = th.tensor(max_acts, device=device)

    else:
        max_acts = (
            th.tensor(
                df["max_act_train"],
                device=device,
            )
            .to(device)
            .clamp(min=1e-6)[all_latents]
        )
    assert not max_acts.isnan().any(), "Max activations contain NaNs"
    assert max_acts.shape == (len(all_latents),), "Max activations have wrong shape"
    # Create bucket boundaries based on max activations
    bucket_edges = th.tensor([1e-6, 0.1, 0.4, 0.7], device=device)

    results = {}
    n_buckets = len(bucket_edges) + 1
    buckets = th.zeros((len(twins), n_buckets, n_buckets), device=device)
    count_total = th.zeros(len(twins), device=device)

    for batch_data in tqdm(dataloader, desc="Computing bucketed activations"):
        batch_data = batch_data.to(device)
        rel_batch_features = (
            crosscoder.get_activations(batch_data, select_features=all_latents)
            / max_acts
        )

        for i in range(len(twins)):
            # Get activations for both features
            acts_A = rel_batch_features[:, 2 * i]
            acts_B = rel_batch_features[:, 2 * i + 1]

            # Compute bucket indices
            bucket_idx_A = th.bucketize(acts_A, bucket_edges)
            bucket_idx_B = th.bucketize(acts_B, bucket_edges)
            assert bucket_idx_A.max() < n_buckets
            assert bucket_idx_B.max() < n_buckets
            assert acts_A.shape == bucket_idx_A.shape
            assert acts_B.shape == bucket_idx_B.shape
            # Update bucket counts using torch operations
            buckets[i].index_put_(
                (bucket_idx_A, bucket_idx_B),
                th.ones_like(bucket_idx_A, dtype=buckets[i].dtype),
                accumulate=True,
            )

            count_total[i] += batch_data.size(0)

    results["buckets"] = buckets.cpu().numpy().tolist()
    results["bucket_edges"] = bucket_edges.cpu().numpy().tolist()
    results["count_total"] = count_total.cpu().numpy().tolist()
    return results


def main(n, batch_size, twins_file, data_store, crosscoder_name, split):
    print("CUDA available: ", th.cuda.is_available())

    BASE_MODEL = "gemma-2-2b"
    INSTRUCT_MODEL = "gemma-2-2b-it"
    DATA_STORE = Path(data_store)
    activation_store_dir = DATA_STORE / "activations"

    fw_dataset, lmsys_dataset = load_activation_dataset(
        activation_store_dir,
        base_model=BASE_MODEL,
        instruct_model=INSTRUCT_MODEL,
        layer=13,
        split=split,
    )
    if n is not None:
        fw_dataset = th.utils.data.Subset(fw_dataset, th.arange(0, n))
        lmsys_dataset = th.utils.data.Subset(lmsys_dataset, th.arange(0, n))
    num_works = min(10, os.cpu_count())
    # Create DataLoader for batch processing
    fw_dataloader = th.utils.data.DataLoader(
        fw_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_works,
    )
    lmsys_dataloader = th.utils.data.DataLoader(
        lmsys_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_works,
    )

    fw_results = {
        ("abs_max_act" if ama else "rel_max_act"): compute_feature_metrics(
            fw_dataloader,
            crosscoder_name,
            twins_file,
            device="cuda:0",
            abosulte_max_act=ama,
        )
        for ama in [True, False]
    }
    lmsys_results = {
        ("abs_max_act" if ama else "rel_max_act"): compute_feature_metrics(
            lmsys_dataloader,
            crosscoder_name,
            twins_file,
            device="cuda:0",
            abosulte_max_act=ama,
        )
        for ama in [True, False]
    }
    results = {
        "fw_results": fw_results,
        "lmsys_results": lmsys_results,
    }

    save_path = Path("results") / "twin_stats"
    save_path.mkdir(parents=True, exist_ok=True)
    # make paths
    str_n = f"N{n}" if n is not None else "all"
    file = save_path / f"{split}_twins-{crosscoder_name}_stats_{str_n}.json"
    with open(file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(
        description="Compute feature metrics on model activations"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of activations to load from validation set (default: 100,000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size over N dimension (default: 4096)",
    )
    parser.add_argument(
        "--twins-file",
        type=str,
        help="Path to file containing twins",
        default="data/twins.json",
    )
    parser.add_argument(
        "--data-store", type=str, help="Path to data store", default="."
    )
    parser.add_argument(
        "--crosscoder-name",
        type=str,
        help="Name of crosscoder",
        default="l13_crosscoder",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split to use",
        default="validation",
    )
    args = parser.parse_args()

    main(
        args.n,
        args.batch_size,
        args.twins_file,
        args.data_store,
        args.crosscoder_name,
        args.split,
    )
