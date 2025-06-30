from pathlib import Path
from argparse import ArgumentParser
import sys
import warnings
from tqdm import tqdm, trange
import torch as th
import numpy as np
from dictionary_learning.cache import PairedActivationCache
from dictionary_learning import CrossCoder
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from tools.utils import (
    save_json,
    load_activation_dataset,
    load_dictionary_model,
    load_latent_df,
    push_latent_df,
)
from tools.cache_utils import LatentActivationCache

# th.set_float32_matmul_precision("high")


@th.no_grad()
def compute_stats_from_latent_cache(
    latent_cache: LatentActivationCache,
    device,
):
    """
    Compute maximum activation values and frequencies for each latent feature using a LatentActivationCache.

    Args:
        latent_cache: LatentActivationCache containing the latent activations
        device: Device to perform computations on

    Returns:
        Tuple containing:
        - Tensor of shape (dict_size,) containing the maximum activation value for each latent feature
        - Tensor of shape (dict_size,) containing the frequency of each feature (non-zero activations / total tokens)
        - int: total number of tokens in the cache
    """
    latent_cache.to(device)
    dict_size = latent_cache.dict_size
    max_activations = th.zeros(dict_size, device=device)
    nonzero_counts = th.zeros(dict_size, device=device, dtype=th.long)
    total_tokens = 0

    # Iterate through all samples in the cache
    pbar = trange(len(latent_cache), desc="Computing max activations")
    for i in pbar:
        # Get latent activations for this sample
        tokens, activations = latent_cache[i]
        total_tokens += len(tokens)
        # If using sparse tensor format
        if isinstance(activations, th.Tensor) and activations.is_sparse:
            values = activations.values()
            indices = activations.indices()
            latent_indices = indices[1, :]  # Second row contains latent indices
            max_activations.index_reduce_(0, latent_indices, values, "amax")
            # Count non-zero activations per feature
            unique_indices, counts = latent_indices.unique(return_counts=True)
            nonzero_counts.index_add_(0, unique_indices, counts)

        # If using dense format
        elif isinstance(activations, th.Tensor):
            batch_max = activations.max(dim=0).values
            max_activations = th.maximum(max_activations, batch_max)
            nonzero_counts += (activations != 0).sum(dim=0)

        # If using sparse tuple format (indices, values)
        else:
            indices, values = activations
            latent_indices = indices[:, 1]
            max_activations.index_reduce_(0, latent_indices, values, "amax")
            # Count non-zero activations
            unique_indices, counts = latent_indices.unique(return_counts=True)
            nonzero_counts.index_add_(0, unique_indices, counts)

    frequencies = nonzero_counts / total_tokens
    return max_activations.cpu(), frequencies.cpu(), total_tokens


@th.no_grad()
def compute_stats(
    crosscoder: CrossCoder,
    cache: PairedActivationCache,
    device,
    batch_size: int = 2048,
    num_workers: int = 16,
):
    dataloader = th.utils.data.DataLoader(
        cache, batch_size=batch_size, num_workers=num_workers
    )
    max_activations = th.zeros(crosscoder.dict_size, device=device)
    nonzero_counts = th.zeros(crosscoder.dict_size, device=device)
    total_tokens = 0

    for batch in tqdm(dataloader):
        activations = crosscoder.get_activations(
            batch.to(device)
        )  # (batch_size, dict_size)
        assert activations.shape == (len(batch), crosscoder.dict_size)
        max_activations = th.max(max_activations, activations.max(dim=0).values)
        nonzero_counts += (activations != 0).sum(dim=0)
        total_tokens += activations.shape[0]

    frequencies = nonzero_counts / total_tokens
    assert max_activations.shape == (crosscoder.dict_size,)
    return max_activations.cpu(), frequencies.cpu(), total_tokens


def main(
    crosscoder: str,
    latent_activation_cache_suffix: str = "",
    activation_cache_path: Path = Path("./activations"),
    device: str = "cuda",
    layer: int = 13,
    batch_size: int = 2048,
    num_workers: int = 16,
    confirm: bool = False,
    latent_activation_cache_path: Path | None = None,
    latent_activation_cache: LatentActivationCache | None = None,
    use_sparse_tensor: bool = False,
    push_to_hub: bool = True,
    df_filename: str = "feature_df",
) -> tuple[pd.DataFrame, dict]:
    results = {}

    # Use LatentActivationCache if path is provided
    if latent_activation_cache_path or latent_activation_cache:
        print(f"Using LatentActivationCache from {latent_activation_cache_path}")
        split = "validation"
        warnings.warn(
            "LatentActivationCache is faster but only supports validation split",
        )
        if latent_activation_cache is None:
            split_path = latent_activation_cache_path / crosscoder
            if latent_activation_cache_suffix:
                split_path = split_path / latent_activation_cache_suffix

            print(f"Processing {split} split")
            latent_activation_cache = LatentActivationCache(
                split_path, expand=True, use_sparse_tensor=use_sparse_tensor
            )
        max_activations, frequencies, total_tokens = compute_stats_from_latent_cache(
            latent_activation_cache, device
        )
        results[split] = {
            "max_activations": max_activations.tolist(),
            "frequencies": frequencies.tolist(),
            "total_tokens": total_tokens,
        }
    # Otherwise use the original PairedActivationCache approach
    else:
        print("Using PairedActivationCache")
        crosscoder_model = load_dictionary_model(crosscoder).to(device)
        for split in ["train", "validation"]:
            fw_dataset, lmsys_dataset = load_activation_dataset(
                activation_cache_path,
                base_model="gemma-2-2b",
                instruct_model="gemma-2-2b-it",
                layer=layer,
                split=split,
            )

            max_activations_fw, frequencies_fw, total_tokens_fw = compute_stats(
                crosscoder_model, fw_dataset, device, batch_size, num_workers
            )
            max_activations_lmsys, frequencies_lmsys, total_tokens_lmsys = (
                compute_stats(
                    crosscoder_model,
                    lmsys_dataset,
                    device,
                    batch_size,
                    num_workers,
                )
            )
            results[split] = {
                "max_activations_fw": max_activations_fw.tolist(),
                "max_activations_lmsys": max_activations_lmsys.tolist(),
                "frequencies_fw": frequencies_fw.tolist(),
                "frequencies_lmsys": frequencies_lmsys.tolist(),
                "total_tokens_fw": total_tokens_fw,
                "total_tokens_lmsys": total_tokens_lmsys,
            }

    path = Path("results") / f"max_acts_{crosscoder}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, path)

    # Update latent dataframe with max activation values
    df = load_latent_df(crosscoder, df_name=df_filename)

    if latent_activation_cache_path or latent_activation_cache:
        # For LatentActivationCache approach
        if "train" in results:
            df["max_act_train"] = results["train"]["max_activations"]
            df["freq_train"] = results["train"]["frequencies"]
        if "validation" in results:
            df["max_act_val"] = results["validation"]["max_activations"]
            df["freq_val"] = results["validation"]["frequencies"]
        message = "Added max activations and frequencies for validation split"
    else:
        # For original approach with separate FW and LMSYS datasets
        if "train" in results:
            df["max_act_train"] = np.maximum(
                results["train"]["max_activations_fw"],
                results["train"]["max_activations_lmsys"],
            )
            df["freq_train"] = (
                results["train"]["frequencies_fw"] * results["train"]["total_tokens_fw"]
                + results["train"]["frequencies_lmsys"]
                * results["train"]["total_tokens_lmsys"]
            ) / (
                results["train"]["total_tokens_fw"]
                + results["train"]["total_tokens_lmsys"]
            )
            df["max_act_lmsys_train"] = results["train"]["max_activations_lmsys"]
            df["max_act_fw_train"] = results["train"]["max_activations_fw"]
            df["freq_lmsys_train"] = results["train"]["frequencies_lmsys"]
            df["freq_fw_train"] = results["train"]["frequencies_fw"]

        if "validation" in results:
            df["max_act_val"] = np.maximum(
                results["validation"]["max_activations_fw"],
                results["validation"]["max_activations_lmsys"],
            )
            df["freq_val"] = (
                results["validation"]["frequencies_fw"]
                * results["validation"]["total_tokens_fw"]
                + results["validation"]["frequencies_lmsys"]
                * results["validation"]["total_tokens_lmsys"]
            ) / (
                results["validation"]["total_tokens_fw"]
                + results["validation"]["total_tokens_lmsys"]
            )
            df["max_act_lmsys_val"] = results["validation"]["max_activations_lmsys"]
            df["max_act_fw_val"] = results["validation"]["max_activations_fw"]
            df["freq_lmsys_val"] = results["validation"]["frequencies_lmsys"]
            df["freq_fw_val"] = results["validation"]["frequencies_fw"]
        message = "Added max activations and frequencies for all splits and datasets"

    if push_to_hub:
        push_latent_df(
            df,
            crosscoder=crosscoder,
            commit_message=message,
            confirm=confirm,
            filename=df_filename,
        )
    return df, results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("crosscoder", type=str)
    parser.add_argument(
        "--activation-cache-path", "-p", type=Path, default="./activations"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--confirm", action="store_true", help="Confirm before pushing")
    parser.add_argument(
        "--latent-activation-cache-path",
        type=Path,
        help="Path to latent activations directory. If provided, will use LatentActivationCache instead.",
    )
    parser.add_argument(
        "--latent-activation-cache-suffix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--use-sparse-tensor",
        action="store_true",
        help="Use sparse tensor representation for latent activations",
    )
    parser.add_argument(
        "--df-filename",
        type=str,
        default="feature_df",
        help="Filename for the feature dataframe",
    )
    args = parser.parse_args()

    main(
        crosscoder=args.crosscoder,
        activation_cache_path=args.activation_cache_path,
        device=args.device,
        layer=args.layer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        confirm=args.confirm,
        latent_activation_cache_path=args.latent_activation_cache_path,
        latent_activation_cache_suffix=args.latent_activation_cache_suffix,
        use_sparse_tensor=args.use_sparse_tensor,
        df_filename=args.df_filename,
    )
