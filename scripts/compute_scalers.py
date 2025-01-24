# %%
import sys
sys.path.append(".")
import torch as th
from typing import Callable
from dictionary_learning import CrossCoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import numpy as np
from loguru import logger
import argparse
import os
th.set_grad_enabled(False)

from tools.latent_scaler.closed_form import (
    remove_latents,
    closed_form_scalars,
    run_tests,
)

def compute_max_activations(dataloader, cc, device):
    max_activations = th.zeros(cc.dict_size, device=device)
    for batch in tqdm(dataloader, desc="Computing max activations"):
        batch = batch.to(device)
        latent_activations = cc.encode(batch)
        # Update max values and immediately free the intermediate tensor
        max_activations = th.max(max_activations, latent_activations.max(dim=0).values)
        # Explicitly clear intermediate tensors
        del latent_activations
        # Force GPU memory cleanup
        if device == "cuda":
            th.cuda.empty_cache()
    return max_activations

def load_base_activation(batch, **kwargs):
    return batch[:, 0, :]


def load_base_error(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return batch[:, 0, :] - remove_latents(
        reconstruction[:, 0, :],
        latent_activations[:, latent_indices],
        base_decoder[latent_indices],
    )


def load_base_reconstruction(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return remove_latents(
        reconstruction[:, 0, :], latent_activations[:, latent_indices], latent_vectors
    )


def load_chat_reconstruction(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return remove_latents(
        reconstruction[:, 0, :], latent_activations[:, latent_indices], latent_vectors
    )


def load_chat_error(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return batch[:, 0, :] - remove_latents(
        reconstruction[:, 0, :], latent_activations[:, latent_indices], latent_vectors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--crosscoder-path",
        type=str,
        default="Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
    )
    parser.add_argument(
        "--activation-store-dir", type=Path, default="/workspace/data/activations/"
    )
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("-N", "--num-samples", type=int, default=20_000_000)
    parser.add_argument("--base-model", type=str, default="gemma-2-2b")
    # parser.add_argument("--latent-df-path", type=str, default="Butanium/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04")
    parser.add_argument("--instruct-model", type=str, default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default="/workspace/data/results/closed_form_scalars",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument(
        "--threshold-active-latents",
        type=float,
        default=None,
        help="If not None, only consider latents with more than this percentage of active the max activation.",
    )
    args = parser.parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = th.device(args.device)

    # Run tests first
    # run_tests(verbose=True)

    if args.threshold_active_latents is not None:
        assert args.threshold_active_latents > 0 and args.threshold_active_latents < 1, "Threshold must be between 0 and 1"
    
    # Loading latent dataframe
    # if os.path.exists(args.latent_df_path):
    #     latent_df = pd.read_csv(args.latent_df_path)
    # else:
    #     latent_df = pd.read_csv(hf_hub_download(args.latent_df_path,filename="feature_df.csv", repo_type="dataset"))

    # Load crosscoder
    cc = CrossCoder.from_pretrained(args.crosscoder_path, from_hub=True)
    cc = cc.to(device)

    # Setup paths
    activation_store_dir = Path(args.activation_store_dir)
    base_model_dir = activation_store_dir / args.base_model
    instruct_model_dir = activation_store_dir / args.instruct_model

    base_model_fineweb = base_model_dir / "fineweb-1m-sample" / args.dataset_split
    base_model_lmsys_chat = (
        base_model_dir / "lmsys-chat-1m-gemma-formatted" / args.dataset_split
    )
    instruct_model_fineweb = (
        instruct_model_dir / "fineweb-1m-sample" / args.dataset_split
    )
    instruct_model_lmsys_chat = (
        instruct_model_dir / "lmsys-chat-1m-gemma-formatted" / args.dataset_split
    )

    submodule_name = f"layer_{args.layer}_out"

    # Load caches and create dataset
    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_chat_cache = PairedActivationCache(
        base_model_lmsys_chat / submodule_name,
        instruct_model_lmsys_chat / submodule_name,
    )

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    # Get decoder weights
    global it_decoder, base_decoder, chat_only_indices
    it_decoder = cc.decoder.weight[1, :, :].clone()
    base_decoder = cc.decoder.weight[0, :, :].clone()

    n_per_dataset = args.num_samples // 2
    test_idx = th.cat(
        [th.arange(n_per_dataset), th.arange(n_per_dataset) + len(fineweb_cache)]
    )
    dataset = th.utils.data.Subset(dataset, test_idx)
    logger.info(f"Number of activations: {len(dataset)}")

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    chat_only_indices = th.load(
        activation_store_dir / ".." / "only_it_decoder_feature_indices.pt"
    )


    latent_activation_postprocessing_fn = None
    if args.threshold_active_latents is not None:
        max_act_path = activation_store_dir / ".." / f"max_activations_N{args.num_samples}.pt"
        if not os.path.exists(max_act_path):
            # Compute max activations
            max_activations = compute_max_activations(dataloader, cc, device)
            assert max_activations.shape == (cc.dict_size,)
            th.save(max_activations, max_act_path)
        else:
            max_activations = th.load(max_act_path)

        threshold = max_activations * args.threshold_active_latents

        def jumprelu_latent_activations(latent_activations):
            # latent_activations: (batch_size, dict_size)
            # Set latent activations to 0 if their value lies below 10% of the max act.
            latent_activations = latent_activations.masked_fill(latent_activations < threshold, 0)
            return latent_activations
        latent_activation_postprocessing_fn = jumprelu_latent_activations
    # Create results directory
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Run all computations
    for name, loader_fn in [
        ("base_reconstruction", load_base_reconstruction),
        ("base_error", load_base_error),
        ("it_reconstruction", load_chat_reconstruction),
        ("it_error", load_chat_error),
    ]:
        if latent_activation_postprocessing_fn is not None:
            name += f"_jumprelu{args.threshold_active_latents}"
        logger.info(f"Computing {name}")
        betas, count_active = closed_form_scalars(
            it_decoder[chat_only_indices],
            chat_only_indices,
            dataloader,
            cc,
            loader_fn,
            device=device,
            latent_activation_postprocessing_fn=latent_activation_postprocessing_fn,
        )
        th.save(betas, args.results_dir / f"betas_{name}.pt")
        th.save(count_active, args.results_dir / f"count_active_{name}.pt")


if __name__ == "__main__":
    main()
