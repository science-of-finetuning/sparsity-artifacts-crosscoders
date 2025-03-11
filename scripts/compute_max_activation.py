from pathlib import Path
from argparse import ArgumentParser
import sys
import warnings
from tqdm import tqdm, trange
import torch as th
import numpy as np
from dictionary_learning.cache import PairedActivationCache
from dictionary_learning import CrossCoder

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
def compute_max_activation_from_latent_cache(
    latent_cache: LatentActivationCache,
    device,
):
    """
    Compute maximum activation values for each latent feature using a LatentActivationCache.

    Args:
        latent_cache: LatentActivationCache containing the latent activations
        device: Device to perform computations on

    Returns:
        Tensor of shape (dict_size,) containing the maximum activation value for each latent feature
    """
    latent_cache.to(device)
    dict_size = latent_cache.dict_size
    max_activations = th.zeros(dict_size, device=device)

    # Iterate through all samples in the cache
    pbar = trange(len(latent_cache), desc="Computing max activations")
    for i in pbar:
        # Get latent activations for this sample
        _, activations = latent_cache[i]

        # If using sparse tensor format
        if isinstance(activations, th.Tensor) and activations.is_sparse:
            # For sparse tensors, we can get the values directly
            values = activations.values()
            indices = activations.indices()
            # Update max activations for the corresponding latent features
            latent_indices = indices[1, :]  # Second row contains latent indices
            max_activations.index_reduce_(0, latent_indices, values, "amax")

        # If using dense format
        elif isinstance(activations, th.Tensor):
            # For dense tensors, compute max across sequence dimension
            batch_max = activations.max(dim=0).values
            max_activations = th.maximum(max_activations, batch_max)

        # If using sparse tuple format (indices, values)
        else:
            indices, values = activations
            # Extract latent indices (second column)
            latent_indices = indices[:, 1]
            # Update max activations for the corresponding latent features
            max_activations.index_reduce_(0, latent_indices, values, "amax")

    return max_activations.cpu()


@th.no_grad()
def compute_max_activation(
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
    for batch in tqdm(dataloader):
        activations = crosscoder.get_activations(
            batch.to(device)
        )  # (batch_size, dict_size)
        max_activations = th.max(max_activations, activations.max(dim=0).values)
    assert max_activations.shape == (crosscoder.dict_size,)
    return max_activations.cpu()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--activation-cache-path", "-p", type=Path, default="./activations"
    )
    parser.add_argument("--crosscoder", type=str, default="l13_crosscoder")
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
        "--use-sparse-tensor",
        action="store_true",
        help="Use sparse tensor representation for latent activations",
    )
    args = parser.parse_args()

    results = {}

    # Use LatentActivationCache if path is provided
    if args.latent_activation_cache_path:
        print(f"Using LatentActivationCache from {args.latent_activation_cache_path}")
        split = "validation"
        warnings.warn(
            "LatentActivationCache is faster but only supports validation split",
        )
        split_path = args.latent_activation_cache_path / args.crosscoder

        print(f"Processing {split} split")
        latent_cache = LatentActivationCache(
            split_path, expand=True, use_sparse_tensor=args.use_sparse_tensor
        )
        max_activations = compute_max_activation_from_latent_cache(
            latent_cache, args.device
        )
        results[split] = {
            "max_activations": max_activations.tolist(),
        }
    # Otherwise use the original PairedActivationCache approach
    else:
        print("Using PairedActivationCache")
        crosscoder = load_dictionary_model(args.crosscoder).to(args.device)
        for split in ["train", "validation"]:
            fw_dataset, lmsys_dataset = load_activation_dataset(
                args.activation_cache_path,
                base_model="gemma-2-2b",
                instruct_model="gemma-2-2b-it",
                layer=args.layer,
                split=split,
            )

            max_activations_fw = compute_max_activation(
                crosscoder, fw_dataset, args.device, args.batch_size, args.num_workers
            )
            max_activations_lmsys = compute_max_activation(
                crosscoder,
                lmsys_dataset,
                args.device,
                args.batch_size,
                args.num_workers,
            )
            results[split] = {
                "max_activations_fw": max_activations_fw.tolist(),
                "max_activations_lmsys": max_activations_lmsys.tolist(),
            }

    path = Path("results") / f"max_acts_{args.crosscoder}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, path)

    # Update latent dataframe with max activation values
    df = load_latent_df(args.crosscoder)

    if args.latent_activation_cache_path:
        # For LatentActivationCache approach
        if "train" in results:
            df["max_act_train"] = results["train"]["max_activations"]
        if "validation" in results:
            df["max_act_val"] = results["validation"]["max_activations"]
    else:
        # For original approach with separate FW and LMSYS datasets
        if "train" in results:
            df["max_act_train"] = np.maximum(
                results["train"]["max_activations_fw"],
                results["train"]["max_activations_lmsys"],
            )
            df["max_act_lmsys_train"] = results["train"]["max_activations_lmsys"]
            df["max_act_fw_train"] = results["train"]["max_activations_fw"]

        if "validation" in results:
            df["max_act_val"] = np.maximum(
                results["validation"]["max_activations_fw"],
                results["validation"]["max_activations_lmsys"],
            )
            df["max_act_lmsys_val"] = results["validation"]["max_activations_lmsys"]
            df["max_act_fw_val"] = results["validation"]["max_activations_fw"]

    message = "Added max activations for all splits and datasets"
    if args.latent_activation_cache_path:
        message = "Added max activations for validation split"
    push_latent_df(
        df,
        crosscoder=args.crosscoder,
        commit_message=message,
        confirm=args.confirm,
    )
