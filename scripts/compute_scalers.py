# %%
import sys

sys.path.append(".")
import torch as th
from typing import Callable, Union
from dictionary_learning import CrossCoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import numpy as np
from functools import partial
from loguru import logger
import argparse
from tools.utils import load_activation_dataset

import os

th.set_grad_enabled(False)
th.set_float32_matmul_precision("highest")

from tools.latent_scaler.closed_form import (
    remove_latents,
    closed_form_scalars,
    run_tests,
)
from tools.utils import load_connor_crosscoder


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


def load_chat_activation(batch, **kwargs):
    return batch[:, 1, :]


def load_base_error(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    base_decoder: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return batch[:, 0, :] - remove_latents(
        reconstruction[:, 0, :],
        latent_activations[:, latent_indices],
        base_decoder[latent_indices],
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
    return batch[:, 1, :] - remove_latents(
        reconstruction[:, 1, :], latent_activations[:, latent_indices], latent_vectors
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
    return reconstruction[:, 0, :]


def load_chat_reconstruction(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return reconstruction[:, 1, :]


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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("-N", "--num-samples", type=int, default=20_000_000)
    parser.add_argument("--base-model", type=str, default="gemma-2-2b")
    parser.add_argument("--instruct-model", type=str, default="gemma-2-2b-it")
    parser.add_argument(
        "--chat-only-indices-path",
        type=Path,
        default="/workspace/data/only_it_decoder_feature_indices.pt",
    )
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
    parser.add_argument("--chat-error", action="store_true")
    parser.add_argument("--chat-reconstruction", action="store_true")
    parser.add_argument("--base-error", action="store_true")
    parser.add_argument("--base-reconstruction", action="store_true")
    parser.add_argument("--random-vectors", action="store_true")
    parser.add_argument("--random-indices", action="store_true")
    parser.add_argument("--connor-crosscoder", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("-SRD", "--special-results-dir", type=str, default="", help="Addon to the results directory. Results will be saved in results_dir/SRD/model_name/")
    parser.add_argument("--n-offset", type=int, default=0, help="Offset for the number of samples. If non-zero, the start index will be n_offset * num_samples")
    parser.add_argument("--shuffle-within-dataset", action="store_true")
    parser.add_argument("--lmsys-subfolder", type=str, default=None, help="Subfolder for the LMSYS dataset")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64", "bfloat16"],
        help="Data type for computations",
    )
    args = parser.parse_args()

    if (
        not args.chat_error
        and not args.chat_reconstruction
        and not args.base_error
        and not args.base_reconstruction
    ):
        logger.info("No computations selected, running all")
        args.chat_error = True
        args.chat_reconstruction = True
        args.base_error = True
        args.base_reconstruction = True

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = th.device(args.device)

    # Setup dtype
    dtype_map = {"float32": th.float32, "float64": th.float64, "bfloat16": th.bfloat16}
    dtype = dtype_map[args.dtype]
    logger.info(f"Using dtype: {dtype}")

    # Run tests first
    print("Running tests...")
    run_tests(verbose=True)

    if args.threshold_active_latents is not None:
        assert (
            args.threshold_active_latents > 0 and args.threshold_active_latents < 1
        ), "Threshold must be between 0 and 1"

    # Load crosscoder
    if args.connor_crosscoder:
        cc = load_connor_crosscoder()
        lmsys_split = f"{args.dataset_split}-coltext_base_format"
        args.crosscoder_path = "ckkissane/crosscoder-gemma-2-2b-model-diff"
    else:
        cc = CrossCoder.from_pretrained(args.crosscoder_path, from_hub=True)
        lmsys_split = args.dataset_split
    cc = cc.to(device).to(dtype)

    # Setup paths
    # Load validation dataset
    activation_store_dir = Path(args.activation_store_dir)

    fineweb_cache, lmsys_cache = load_activation_dataset(
        activation_store_dir,
        base_model=args.base_model,
        instruct_model=args.instruct_model,
        layer=args.layer,
        split=args.dataset_split,
        lmsys_subfolder=args.lmsys_subfolder
    )

    num_samples_per_dataset = args.num_samples // 2
    dataset = th.utils.data.ConcatDataset(
        [
            th.utils.data.Subset(fineweb_cache, th.arange(args.n_offset * num_samples_per_dataset, (args.n_offset + 1) * num_samples_per_dataset)),
            th.utils.data.Subset(lmsys_cache, th.arange(args.n_offset * num_samples_per_dataset, (args.n_offset + 1) * num_samples_per_dataset)),
        ]
    )

    if args.epochs > 1:
        dataset = th.utils.data.ConcatDataset([dataset] * args.epochs)

    if args.shuffle_within_dataset:
        dataset = th.utils.data.Subset(dataset, th.randperm(len(dataset)))

    # Get decoder weights
    it_decoder = cc.decoder.weight[1, :, :].clone().to(dtype)
    assert it_decoder.shape == (cc.dict_size, cc.activation_dim)
    base_decoder = cc.decoder.weight[0, :, :].clone().to(dtype)
    assert base_decoder.shape == (cc.dict_size, cc.activation_dim)

    logger.info(f"Number of activations: {len(dataset)}")

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    chat_only_indices = th.load(args.chat_only_indices_path, weights_only=True)

    latent_activation_postprocessing_fn = None
    if args.threshold_active_latents is not None:
        max_act_path = (
            activation_store_dir / ".." / f"max_activations_N{args.num_samples}.pt"
        )
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
            latent_activations = latent_activations.masked_fill(
                latent_activations < threshold, 0
            )
            return latent_activations

        latent_activation_postprocessing_fn = jumprelu_latent_activations
    
    # Create results directory
    if args.special_results_dir:
        results_dir = args.results_dir / args.special_results_dir / args.crosscoder_path.replace("/", "_")
    else:
        results_dir = args.results_dir / args.crosscoder_path.replace("/", "_")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Saving results to ", results_dir)
    computations = []
    if args.base_reconstruction:
        computations.append(("base_reconstruction", load_base_reconstruction))
    if args.base_error:
        computations.append(
            ("base_error", partial(load_base_error, base_decoder=base_decoder))
        )
    if args.chat_reconstruction:
        computations.append(("it_reconstruction", load_chat_reconstruction))
    if args.chat_error:
        computations.append(("it_error", load_chat_error))

    latent_vectors = it_decoder[chat_only_indices].clone()
    if args.random_vectors:
        random_vectors = th.randn(
            len(chat_only_indices), cc.activation_dim, device=device
        )
        assert random_vectors.shape == (len(chat_only_indices), cc.activation_dim)
        # Scale random vectors to match the norm of the IT decoder vectors
        it_decoder_norm = th.norm(latent_vectors, dim=1)
        print(it_decoder_norm.shape)
        print(th.norm(random_vectors, dim=1, keepdim=True).shape)
        assert it_decoder_norm.shape == (len(chat_only_indices),)
        random_vectors = random_vectors * (
            it_decoder_norm / th.norm(random_vectors, dim=1)
        ).unsqueeze(1)
        assert random_vectors.shape == (len(chat_only_indices), cc.activation_dim)
        latent_vectors = random_vectors

    if args.random_indices:
        random_indices = th.randint(
            0, cc.dict_size, (len(chat_only_indices),), device=device
        )
        assert random_indices.shape == (len(chat_only_indices),)
        # Scale random indices to match the norm of the IT decoder vectors
        random_indices_vectors = it_decoder[random_indices].clone()
        assert random_indices_vectors.shape == (
            len(chat_only_indices),
            cc.activation_dim,
        )
        # Scale random vectors to match the norm of the IT decoder vectors
        it_decoder_norm = th.norm(latent_vectors, dim=1)
        assert it_decoder_norm.shape == (len(chat_only_indices),)
        random_indices_vectors = random_indices_vectors * (
            it_decoder_norm / th.norm(random_indices_vectors, dim=1)
        ).unsqueeze(1)
        assert random_indices_vectors.shape == (
            len(chat_only_indices),
            cc.activation_dim,
        )
        latent_vectors = random_indices_vectors

    # Run all computations
    for name, loader_fn in computations:
        name += f"_N{args.num_samples}_n_offset{args.n_offset}"
        if latent_activation_postprocessing_fn is not None:
            name += f"_jumprelu{args.threshold_active_latents}"
        if args.random_vectors:
            name += f"_random_vectors_s{args.seed}"
        if args.random_indices:
            name += f"_random_indices_s{args.seed}"
        if args.name:
            name += f"_{args.name}"
        logger.info(f"Computing {name}")
        betas, count_active, nominator, norm_f, norm_d = closed_form_scalars(
            latent_vectors,
            chat_only_indices,
            dataloader,
            cc,
            loader_fn,
            device=device,
            dtype=dtype,
            latent_activation_postprocessing_fn=latent_activation_postprocessing_fn,
        )
        th.save(betas.cpu(), results_dir / f"betas_{name}.pt")
        th.save(count_active.cpu(), results_dir / f"count_active_{name}.pt")
        th.save(nominator.cpu(), results_dir / f"nominator_{name}.pt")
        th.save(norm_f.cpu(), results_dir / f"norm_f_{name}.pt")
        th.save(norm_d.cpu(), results_dir / f"norm_d_{name}.pt")

        if args.random_indices or args.random_vectors:
            th.save(latent_vectors.cpu(), results_dir / f"latent_vectors_{name}.pt")
        if args.random_indices:
            th.save(random_indices.cpu(), results_dir / f"random_indices_{name}.pt")

if __name__ == "__main__":
    main()
