from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import torch as th
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import json
import argparse

from dictionary_learning import CrossCoder


def compute_feature_metrics(
    dataloader,
    features,
    crosscoder,
    twins_file,
    batch_size=1000,
    d_batch_size=1000,
    device="cpu",
):
    """ """

    with open(twins_file, "r") as f:
        twins = json.load(f)

    results = {}
    N = len(dataloader.dataset)
    D = features.shape[0]

    count_joint = th.zeros(len(twins), device=device)
    count_A = th.zeros(len(twins), device=device)
    count_B = th.zeros(len(twins), device=device)
    count_A_B = th.zeros(len(twins), device=device)
    count_B_A = th.zeros(len(twins), device=device)
    count_total = th.zeros(len(twins), device=device)
    m2_A = th.zeros(len(twins), device=device)
    m2_B = th.zeros(len(twins), device=device)
    covar_A = th.zeros(len(twins), device=device)
    covar_B = th.zeros(len(twins), device=device)
    mean_A = th.zeros(len(twins), device=device)
    mean_B = th.zeros(len(twins), device=device)
    th.cuda.empty_cache()

    with th.no_grad():
        crosscoder.to(device)

        # First pass: compute all projections and store them
        for batch_idx, batch_data in enumerate(
            tqdm(dataloader, desc="Computing feature activations and projections")
        ):
            batch_data = batch_data.to(device)
            batch_base = batch_data[:, 0]  # First column is base model data
            batch_it = batch_data[:, 1]  # Second column is instruction model data

            batch_features = crosscoder.encode(batch_data)
            for i, twin in enumerate(twins):
                A, B = twin
                count_joint[i] += (
                    (batch_features[:, A] > 0) & (batch_features[:, B] > 0)
                ).sum()
                count_A[i] += (batch_features[:, A] > 0).sum()
                count_B[i] += (batch_features[:, B] > 0).sum()
                count_A_B[i] += ((batch_features[:, A] > batch_features[:, B])).sum()
                count_B_A[i] += ((batch_features[:, B] > batch_features[:, A])).sum()
                count_total[i] += batch_base.size(0)

                delta_A = batch_features[:, A] - mean_A[i]
                delta_B = batch_features[:, B] - mean_B[i]

                mean_A[i] += delta_A.sum() / count_total[i]
                mean_B[i] += delta_B.sum() / count_total[i]

                delta2_A = batch_features[:, A] - mean_A[i]
                delta2_B = batch_features[:, B] - mean_B[i]

                m2_A[i] += (delta_A * delta2_A).sum()
                m2_B[i] += (delta_B * delta2_B).sum()

                covar_A[i] += (delta_A * delta_B).sum()
                covar_B[i] += (delta_B * delta_A).sum()

    results["count_joint"] = count_joint.cpu()
    results["count_A"] = count_A.cpu()
    results["count_B"] = count_B.cpu()
    results["count_A_B"] = count_A_B.cpu()
    results["count_B_A"] = count_B_A.cpu()
    results["count_total"] = count_total.cpu()
    valid_count = count_total > 1
    results["correAB"] = (
        covar_A[valid_count]
        / th.sqrt(m2_A[valid_count] * m2_B[valid_count])
        / (count_total[valid_count] - 1)
    )
    results["correBA"] = (
        covar_B[valid_count]
        / th.sqrt(m2_B[valid_count] * m2_A[valid_count])
        / (count_total[valid_count] - 1)
    )
    return results


def main(n, batch_size, d_batch_size, twins_file, data_store):
    print("CUDA available: ", th.cuda.is_available())

    crosscoder = CrossCoder.from_pretrained(
        "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04", from_hub=True
    )

    BASE_MODEL = "gemma-2-2b"
    INSTRUCT_MODEL = "gemma-2-2b-it"
    DATA_STORE = Path(data_store)
    activation_store_dir = DATA_STORE / "activations"

    base_model_dir = activation_store_dir / BASE_MODEL
    instruct_model_dir = activation_store_dir / INSTRUCT_MODEL

    base_model_fineweb = base_model_dir / "fineweb-1m-sample" / "validation"
    base_model_lmsys_chat = (
        base_model_dir / "lmsys-chat-1m-gemma-formatted" / "validation"
    )
    instruct_model_fineweb = instruct_model_dir / "fineweb-1m-sample" / "validation"
    instruct_model_lmsys_chat = (
        instruct_model_dir / "lmsys-chat-1m-gemma-formatted" / "validation"
    )

    submodule_name = f"layer_13_out"

    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_chat_cache = PairedActivationCache(
        base_model_lmsys_chat / submodule_name,
        instruct_model_lmsys_chat / submodule_name,
    )

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    validation_size = 10**6
    train_dataset, validation_dataset = th.utils.data.random_split(
        dataset, [len(dataset) - validation_size, validation_size]
    )

    crosscoder.decoder.weight.shape
    it_decoder = crosscoder.decoder.weight[1, :, :].clone()
    base_decoder = crosscoder.decoder.weight[0, :, :].clone()

    test_idx = th.randint(0, len(validation_dataset), (n,))
    # Get the text indices from the validation dataset
    dataset = th.utils.data.Subset(validation_dataset, test_idx)

    print("Number of activations: ", len(dataset))

    # Create DataLoader for batch processing
    dataloader = th.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10
    )

    results = compute_feature_metrics(
        dataloader,
        it_decoder,
        crosscoder,
        twins_file,
        d_batch_size=d_batch_size,
        device="cuda:0",
    )
    save_path = DATA_STORE / "results"
    # make paths
    name = twins_file.split("/")[-1].split(".")[0]
    save_path.mkdir(parents=True, exist_ok=True)
    file = save_path / f"{name}_activation_statistics_N{n}.pt"
    th.save(results, file)


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(
        description="Compute feature metrics on model activations"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=100_000,
        help="Number of activations to load from validation set (default: 100,000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size over N dimension (default: 1024)",
    )
    parser.add_argument(
        "--d_batch_size",
        type=int,
        default=2048,
        help="Batch size over D dimension (default: 1024)",
    )
    parser.add_argument(
        "--twins-file", type=str, required=True, help="Path to file containing twins"
    )
    parser.add_argument(
        "--data-store", type=str, required=True, help="Path to data store"
    )
    args = parser.parse_args()

    main(args.n, args.batch_size, args.d_batch_size, args.twins_file, args.data_store)
