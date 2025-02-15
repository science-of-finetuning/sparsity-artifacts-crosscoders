from pathlib import Path
import json
import argparse
import sys
import torch as th
import numpy as np
from tqdm.auto import tqdm
from coolname import generate_slug

sys.path.append(".")
from tools.utils import load_latent_df, load_crosscoder, load_activation_dataset


@th.no_grad()
def compute_feature_metrics(
    dataloader,
    crosscoder_name,
    twins_file,
    device="cpu",
):
    """Compute bucketed activation statistics for twin features"""

    with open(twins_file, "r") as f:
        twins = json.load(f)
    all_latents = sum(twins, [])

    # Load max activations from latent df
    df = load_latent_df(crosscoder_name)
    crosscoder = load_crosscoder(crosscoder_name).to(device)
    max_acts = (
        th.tensor(
            np.maximum(df["lmsys_ctrl_max_act"], df["lmsys_non_ctrl_max_act"]),
            device=device,
        )
        .to(device)
        .nan_to_num(nan=1e-6)
        .clamp(min=1e-6)[all_latents]
    )

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

    # Create DataLoader for batch processing
    fw_dataloader = th.utils.data.DataLoader(
        fw_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=10,
    )
    lmsys_dataloader = th.utils.data.DataLoader(
        lmsys_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=10,
    )

    fw_results = compute_feature_metrics(
        fw_dataloader,
        crosscoder_name,
        twins_file,
        device="cuda:0",
    )
    lmsys_results = compute_feature_metrics(
        lmsys_dataloader,
        crosscoder_name,
        twins_file,
        device="cuda:0",
    )
    results = {
        "fw_results": fw_results,
        "lmsys_results": lmsys_results,
    }

    save_path = Path("results") / "twin_stats"
    save_path.mkdir(parents=True, exist_ok=True)
    # make paths
    file = (
        save_path
        / f"{split}_twins{generate_slug(2)}-{crosscoder_name}_activation_statistics_N{n}.json"
    )
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
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size over N dimension (default: 1024)",
    )
    parser.add_argument(
        "--twins-file",
        type=str,
        help="Path to file containing twins",
        default="data/twins.json",
    )
    parser.add_argument(
        "--data-store", type=str, required=True, help="Path to data store"
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
