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
from tools.cc_utils import load_dictionary_model
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
import os

th.set_grad_enabled(False)
th.set_float32_matmul_precision("highest")

from tools.latent_scaler.closed_form import (
    remove_latents,
    closed_form_scalars,
    run_tests,
    identity_fn,
)


def load_base_activation(batch, **kwargs):
    return batch[:, 0, :]


def load_chat_activation(batch, **kwargs):
    return batch[:, 1, :]


def load_base_activation_no_bias(batch, crosscoder: CrossCoder, **kwargs):
    return batch[:, 0, :] - crosscoder.decoder.bias[0, :]


def load_chat_activation_no_bias(batch, crosscoder: CrossCoder, **kwargs):
    return batch[:, 1, :] - crosscoder.decoder.bias[1, :]


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
        "--dictionary-model",
        type=str,
        default="Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
    )
    parser.add_argument(
        "--activation-store-dir", type=Path, default="/workspace/data/activations/"
    )
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("-N", "--num-samples", type=int, default=50_000_000)
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument(
        "--latent-indices-path",
        type=Path,
        default=None,
        help="Path to the latent indices file. If not provided, all latents are considered.",
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
    parser.add_argument("--base-activation", action="store_true")
    parser.add_argument("--chat-activation", action="store_true")
    parser.add_argument("--base-activation-no-bias", action="store_true")
    parser.add_argument("--chat-activation-no-bias", action="store_true")
    parser.add_argument("--random-vectors", action="store_true")
    parser.add_argument("--random-indices", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "-SRD",
        "--special-results-dir",
        type=str,
        default="",
        help="Addon to the results directory. Results will be saved in results_dir/SRD/model_name/",
    )
    parser.add_argument(
        "--n-offset",
        type=int,
        default=0,
        help="Offset for the number of samples. If non-zero, the start index will be n_offset * num_samples",
    )
    parser.add_argument("--shuffle-within-dataset", action="store_true")
    parser.add_argument(
        "--lmsys-subfolder",
        type=str,
        default=None,
        help="Subfolder for the LMSYS dataset",
    )
    parser.add_argument(
        "--lmsys-split",
        type=str,
        default=None,
        help="Split for the LMSYS dataset. If not provided, the default split will be used.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64", "bfloat16"],
        help="Data type for computations",
    )
    parser.add_argument(
        "--max-activations-path",
        type=Path,
        default=None,
        help="Path to the max activations file. ",
    )
    parser.add_argument("--run-tests", action="store_true", help="Run tests first")
    args = parser.parse_args()

    if str(args.latent_indices_path).strip() == "None":
        args.latent_indices_path = None
        latent_indices_name = "all_latents"
    else:
        latent_indices_name = Path(args.latent_indices_path).name.split(".")[0]

    if args.lmsys_split is None:
        args.lmsys_split = args.dataset_split

    if (
        not args.chat_error
        and not args.chat_reconstruction
        and not args.base_error
        and not args.base_reconstruction
        and not args.chat_activation
        and not args.base_activation
        and not args.base_activation_no_bias
        and not args.chat_activation_no_bias
    ):
        logger.info("No computations selected, running all")
        args.chat_error = True
        args.chat_reconstruction = True
        args.base_error = True
        args.base_reconstruction = True
        args.base_activation = True
        args.base_activation_no_bias = True
        args.chat_activation = True
        args.chat_activation_no_bias = True
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
    if args.run_tests:
        print("Running tests...")
        run_tests(verbose=True)

    if args.threshold_active_latents is not None:
        assert (
            args.threshold_active_latents > 0 and args.threshold_active_latents < 1
        ), "Threshold must be between 0 and 1"

    # Load dictionary model
    print(f"Loading dictionary model from {args.dictionary_model}")
    dict_model = load_dictionary_model(args.dictionary_model)
    dict_model = dict_model.to(device).to(dtype)

    # If crosscoder is a local path, replace with only the directory name (e.g. /path/to/crosscoder/model_final.pt -> crosscoder)
    if Path(args.dictionary_model).exists():
        args.dictionary_model = Path(args.dictionary_model).parent.name
    print(f"Using dictionary model name: {args.dictionary_model}")

    # Setup paths
    # Load validation dataset
    activation_store_dir = Path(args.activation_store_dir)

    base_model_stub = args.base_model.split("/")[-1]
    chat_model_stub = args.chat_model.split("/")[-1]
    fineweb_cache, lmsys_cache = load_activation_dataset(
        activation_store_dir,
        base_model=base_model_stub,
        instruct_model=chat_model_stub,
        layer=args.layer,
        split=args.dataset_split,
        lmsys_subfolder=args.lmsys_subfolder,
        lmsys_split=args.lmsys_split,
    )

    num_samples_per_dataset = args.num_samples // 2
    dataset = th.utils.data.ConcatDataset(
        [
            th.utils.data.Subset(
                fineweb_cache,
                th.arange(
                    args.n_offset * num_samples_per_dataset,
                    (args.n_offset + 1) * num_samples_per_dataset,
                ),
            ),
            th.utils.data.Subset(
                lmsys_cache,
                th.arange(
                    args.n_offset * num_samples_per_dataset,
                    (args.n_offset + 1) * num_samples_per_dataset,
                ),
            ),
        ]
    )

    if args.epochs > 1:
        dataset = th.utils.data.ConcatDataset([dataset] * args.epochs)

    if args.shuffle_within_dataset:
        dataset = th.utils.data.Subset(dataset, th.randperm(len(dataset)))

    # Get decoder weights
    if isinstance(dict_model, CrossCoder):
        chat_decoder = dict_model.decoder.weight[1, :, :].clone().to(dtype)
        assert chat_decoder.shape == (dict_model.dict_size, dict_model.activation_dim)
        base_decoder = dict_model.decoder.weight[0, :, :].clone().to(dtype)
        assert base_decoder.shape == (dict_model.dict_size, dict_model.activation_dim)
    else:
        chat_decoder = dict_model.decoder.weight.clone().to(dtype).T
        print("chat_decoder.shape", chat_decoder.shape)
        assert chat_decoder.shape == (dict_model.dict_size, dict_model.activation_dim)
        base_decoder = dict_model.decoder.weight.clone().to(dtype).T
        assert base_decoder.shape == (dict_model.dict_size, dict_model.activation_dim)

    logger.info(f"Number of activations: {len(dataset)}")

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    print(
        "args.latent_indices_path",
        args.latent_indices_path,
        args.latent_indices_path is None,
    )
    if args.latent_indices_path is not None:
        latent_indices = th.load(args.latent_indices_path, weights_only=True)
    else:
        print("No latent indices provided, using all latents")
        latent_indices = th.arange(dict_model.dict_size)

    latent_activation_postprocessing_fn = None
    if args.threshold_active_latents is not None:
        max_act_path = args.max_activations_path
        if not os.path.exists(max_act_path):
            raise ValueError(
                f"Provided max activations path {max_act_path} does not exist"
            )
        max_activations = th.load(max_act_path).to(device)
        assert max_activations.shape == (dict_model.dict_size,)

        threshold = max_activations * args.threshold_active_latents

        def jumprelu_latent_activations(latent_activations):
            # latent_activations: (batch_size, dict_size)
            # Set latent activations to 0 if their value lies below x% of the max act.
            latent_activations = latent_activations.masked_fill(
                latent_activations < threshold, 0
            )
            return latent_activations

        latent_activation_postprocessing_fn = jumprelu_latent_activations

    # Create results directory
    if args.special_results_dir:
        results_dir = (
            args.results_dir
            / args.special_results_dir
            / args.dictionary_model.replace("/", "_")
        )
    else:
        results_dir = args.results_dir / args.dictionary_model.replace("/", "_")
    results_dir = results_dir / latent_indices_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Saving results to ", results_dir)
    encode_activation_fn = identity_fn
    if isinstance(dict_model, BatchTopKSAE):
        # Deal with BatchTopKSAE
        print(
            "BatchTopKSAE detected, using load_chat_activation as encode_activation_fn"
        )
        encode_activation_fn = load_chat_activation
    computations = []
    if args.base_activation:
        computations.append(("base_activation", load_base_activation))
    if args.chat_activation:
        computations.append(("chat_activation", load_chat_activation))
    if args.base_reconstruction:
        computations.append(("base_reconstruction", load_base_reconstruction))
    if args.base_error:
        assert isinstance(
            dict_model, CrossCoder
        ), "Base error only supported for CrossCoder"
        computations.append(
            ("base_error", partial(load_base_error, base_decoder=base_decoder))
        )
    if args.base_activation_no_bias:
        computations.append(("base_activation_no_bias", load_base_activation_no_bias))
    if args.chat_activation_no_bias:
        computations.append(("chat_activation_no_bias", load_chat_activation_no_bias))
    if args.chat_reconstruction:
        computations.append(("it_reconstruction", load_chat_reconstruction))
    if args.chat_error:
        computations.append(("it_error", load_chat_error))

    latent_vectors = chat_decoder[latent_indices].clone()
    print("latent_vectors.shape", latent_vectors.shape)
    if args.random_vectors:
        random_vectors = th.randn(
            len(latent_indices), dict_model.activation_dim, device=device
        )
        assert random_vectors.shape == (len(latent_indices), dict_model.activation_dim)
        # Scale random vectors to match the norm of the IT decoder vectors
        it_decoder_norm = th.norm(latent_vectors, dim=1)
        print(it_decoder_norm.shape)
        print(th.norm(random_vectors, dim=1, keepdim=True).shape)
        assert it_decoder_norm.shape == (len(latent_indices),)
        random_vectors = random_vectors * (
            it_decoder_norm / th.norm(random_vectors, dim=1)
        ).unsqueeze(1)
        assert random_vectors.shape == (len(latent_indices), dict_model.activation_dim)
        latent_vectors = random_vectors

    if args.random_indices:
        random_indices = th.randint(
            0, dict_model.dict_size, (len(latent_indices),), device=device
        )
        assert random_indices.shape == (len(latent_indices),)
        # Scale random indices to match the norm of the IT decoder vectors
        random_indices_vectors = chat_decoder[random_indices].clone()
        assert random_indices_vectors.shape == (
            len(latent_indices),
            dict_model.activation_dim,
        )
        # Scale random vectors to match the norm of the IT decoder vectors
        it_decoder_norm = th.norm(latent_vectors, dim=1)
        assert it_decoder_norm.shape == (len(latent_indices),)
        random_indices_vectors = random_indices_vectors * (
            it_decoder_norm / th.norm(random_indices_vectors, dim=1)
        ).unsqueeze(1)
        assert random_indices_vectors.shape == (
            len(latent_indices),
            dict_model.activation_dim,
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
            latent_indices,
            dataloader,
            dict_model,
            loader_fn,
            device=device,
            dtype=dtype,
            encode_activation_fn=encode_activation_fn,
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
