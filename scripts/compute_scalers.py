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
from dictionary_learning.dictionary import BatchTopKSAE, BatchTopKCrossCoder
import os

th.set_grad_enabled(False)
th.set_float32_matmul_precision("highest")

from tools.latent_scaler.closed_form import (
    remove_latents,
    closed_form_scalars,
    run_tests,
    identity_fn,
)
from tools.configs import DATA_ROOT


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


def compute_scalers(
    dictionary_model,
    layer,
    # Model parameters
    base_model: str = "google/gemma-2-2b",
    chat_model: str = "google/gemma-2-2b-it",
    # Data parameters
    activation_store_dir: Path = Path("./data/activations/"),
    results_dir: Path = Path("./results"),
    dataset_split: str = "train",
    lmsys_subfolder: str | None = None,
    lmsys_split: str | None = None,
    latent_indices: th.Tensor | None = None,
    latent_indices_name: str = "all_latents",
    max_activations_path: Path | None = None,
    # Computation parameters
    batch_size: int = 128,
    num_samples: int = 50_000_000,
    num_workers: int = 32,
    device: str = "cuda",
    dtype: str = "float32",
    seed: int = 42,
    epochs: int = 1,
    n_offset: int = 0,
    threshold_active_latents: float | None = None,
    # Output parameters
    special_results_dir: str = "",
    name: str | None = None,
    # Computation flags
    chat_error: bool = False,
    chat_reconstruction: bool = False,
    base_error: bool = False,
    base_reconstruction: bool = False,
    chat_activation: bool = False,
    base_activation: bool = False,
    base_activation_no_bias: bool = False,
    chat_activation_no_bias: bool = False,
    random_vectors: bool = False,
    random_indices: bool = False,
    shuffle_within_dataset: bool = False,
    _run_tests: bool = False,
    target_model_idx: int | None = None,
) -> None:
    """
    ... (todo)
    smaller_batch_size_for_error: Beta on error can take more memory. If this is set to True, a batch size 8x times smaller is used for the beta error computation.
    """
    if latent_indices is not None and latent_indices_name == "all_latents":
        latent_indices_name = f"custom_indices_{len(latent_indices)}"

    results_dir = results_dir / "closed_form_scalars"
    if lmsys_split is None:
        lmsys_split = dataset_split

    if (
        not chat_error
        and not chat_reconstruction
        and not base_error
        and not base_reconstruction
        and not chat_activation
        and not base_activation
        and not base_activation_no_bias
        and not chat_activation_no_bias
    ):
        logger.info("No computations selected, running all")
        chat_error = True
        chat_reconstruction = True
        base_error = True
        base_reconstruction = True
        base_activation = True
        base_activation_no_bias = True
        chat_activation = True
        chat_activation_no_bias = True

    th.manual_seed(seed)
    np.random.seed(seed)

    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    device = th.device(device)

    # Setup dtype
    dtype_map = {"float32": th.float32, "float64": th.float64, "bfloat16": th.bfloat16}
    dtype = dtype_map[dtype]
    logger.info(f"Using dtype: {dtype}")

    # Run tests first
    if _run_tests:
        print("Running tests...")
        run_tests(verbose=True)

    if threshold_active_latents is not None:
        assert (
            threshold_active_latents > 0 and threshold_active_latents < 1
        ), "Threshold must be between 0 and 1"

    # Load dictionary model
    print(f"Loading dictionary model from {dictionary_model}")
    dict_model = load_dictionary_model(dictionary_model)
    dict_model = dict_model.to(device).to(dtype)

    # If crosscoder is a local path, replace with only the directory name (e.g. /path/to/crosscoder/model_final.pt -> crosscoder)
    if Path(dictionary_model).exists():
        dictionary_model = Path(dictionary_model).parent.name
    print(f"Using dictionary model name: {dictionary_model}")

    # Setup paths
    # Load validation dataset
    activation_store_dir = Path(activation_store_dir)

    base_model_stub = base_model.split("/")[-1]
    chat_model_stub = chat_model.split("/")[-1]
    fineweb_cache, lmsys_cache = load_activation_dataset(
        activation_store_dir,
        base_model=base_model_stub,
        instruct_model=chat_model_stub,
        layer=layer,
        split=dataset_split,
        lmsys_subfolder=lmsys_subfolder,
        lmsys_split=lmsys_split,
    )

    num_samples_per_dataset = num_samples // 2
    dataset = th.utils.data.ConcatDataset(
        [
            th.utils.data.Subset(
                fineweb_cache,
                th.arange(
                    n_offset * num_samples_per_dataset,
                    (n_offset + 1) * num_samples_per_dataset,
                ),
            ),
            th.utils.data.Subset(
                lmsys_cache,
                th.arange(
                    n_offset * num_samples_per_dataset,
                    (n_offset + 1) * num_samples_per_dataset,
                ),
            ),
        ]
    )

    if epochs > 1:
        dataset = th.utils.data.ConcatDataset([dataset] * epochs)

    if shuffle_within_dataset:
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
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    if latent_indices is None:
        print("No latent indices provided, using all latents")
        latent_indices = th.arange(dict_model.dict_size)

    latent_activation_postprocessing_fn = None
    if threshold_active_latents is not None:
        max_act_path = max_activations_path
        if not os.path.exists(max_act_path):
            raise ValueError(
                f"Provided max activations path {max_act_path} does not exist"
            )
        max_activations = th.load(max_act_path).to(device)
        assert max_activations.shape == (dict_model.dict_size,)

        threshold = max_activations * threshold_active_latents

        def jumprelu_latent_activations(latent_activations):
            # latent_activations: (batch_size, dict_size)
            # Set latent activations to 0 if their value lies below x% of the max act.
            latent_activations = latent_activations.masked_fill(
                latent_activations < threshold, 0
            )
            return latent_activations

        latent_activation_postprocessing_fn = jumprelu_latent_activations
    if isinstance(dict_model, BatchTopKCrossCoder) and dict_model.decoupled_code:
        if target_model_idx is None:
            raise ValueError(
                "target_model_idx must be provided if using a decoupled code This is needed to specify which code to use for computing betas"
            )
        if latent_activation_postprocessing_fn is None:

            def latent_activation_postprocessing_fn(x):
                return x[:, target_model_idx]

        else:
            prev_postprocessing_fn = latent_activation_postprocessing_fn

            def latent_activation_postprocessing_fn(x):
                return prev_postprocessing_fn(x[:, target_model_idx])

    # Create results directory
    if special_results_dir:
        results_dir = (
            results_dir / special_results_dir / dictionary_model.replace("/", "_")
        )
    else:
        results_dir = results_dir / dictionary_model.replace("/", "_")
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
    if base_activation:
        computations.append(("base_activation", load_base_activation))
    if chat_activation:
        computations.append(("chat_activation", load_chat_activation))
    if base_reconstruction:
        computations.append(("base_reconstruction", load_base_reconstruction))
    if base_error:
        assert isinstance(
            dict_model, CrossCoder
        ), "Base error only supported for CrossCoder"
        computations.append(
            ("base_error", partial(load_base_error, base_decoder=base_decoder))
        )
    if base_activation_no_bias:
        computations.append(("base_activation_no_bias", load_base_activation_no_bias))
    if chat_activation_no_bias:
        computations.append(("chat_activation_no_bias", load_chat_activation_no_bias))
    if chat_reconstruction:
        computations.append(("chat_reconstruction", load_chat_reconstruction))
    if chat_error:
        computations.append(("chat_error", load_chat_error))

    latent_vectors = chat_decoder[latent_indices].clone()
    print("latent_vectors.shape", latent_vectors.shape)
    if random_vectors:
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

    if random_indices:
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
    for exp_name, loader_fn in computations:
        exp_name += f"_N{num_samples}_n_offset{n_offset}"
        if threshold_active_latents is not None:
            exp_name += f"_jumprelu{threshold_active_latents}"
        if random_vectors:
            exp_name += f"_random_vectors_s{seed}"
        if random_indices:
            exp_name += f"_random_indices_s{seed}"
        if name:
            exp_name += f"_{name}"
        logger.info(f"Computing {exp_name}")
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
        th.save(betas.cpu(), results_dir / f"betas_{exp_name}.pt")
        th.save(count_active.cpu(), results_dir / f"count_active_{exp_name}.pt")
        th.save(nominator.cpu(), results_dir / f"nominator_{exp_name}.pt")
        th.save(norm_f.cpu(), results_dir / f"norm_f_{exp_name}.pt")
        th.save(norm_d.cpu(), results_dir / f"norm_d_{exp_name}.pt")

        if random_indices or random_vectors:
            th.save(latent_vectors.cpu(), results_dir / f"latent_vectors_{exp_name}.pt")
        if random_indices:
            th.save(random_indices.cpu(), results_dir / f"random_indices_{exp_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dictionary-model",
        type=str,
        default="Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
    )
    parser.add_argument(
        "--activation-store-dir", type=Path, default=DATA_ROOT / "activations/"
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
        default=DATA_ROOT / "results/closed_form_scalars",
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

    # Load latent indices here if path is provided
    latent_indices = None
    if (
        args.latent_indices_path is not None
        and str(args.latent_indices_path).strip() != "None"
    ):
        print(f"Loading latent indices from {args.latent_indices_path}")
        latent_indices = th.load(args.latent_indices_path, weights_only=True)

    compute_scalers(
        dictionary_model=args.dictionary_model,
        base_model=args.base_model,
        chat_model=args.chat_model,
        layer=args.layer,
        activation_store_dir=args.activation_store_dir,
        dataset_split=args.dataset_split,
        lmsys_subfolder=args.lmsys_subfolder,
        lmsys_split=args.lmsys_split,
        latent_indices=latent_indices,
        max_activations_path=args.max_activations_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        epochs=args.epochs,
        n_offset=args.n_offset,
        threshold_active_latents=args.threshold_active_latents,
        results_dir=args.results_dir,
        special_results_dir=args.special_results_dir,
        name=args.name,
        chat_error=args.chat_error,
        chat_reconstruction=args.chat_reconstruction,
        base_error=args.base_error,
        base_reconstruction=args.base_reconstruction,
        chat_activation=args.chat_activation,
        base_activation=args.base_activation,
        base_activation_no_bias=args.base_activation_no_bias,
        chat_activation_no_bias=args.chat_activation_no_bias,
        random_vectors=args.random_vectors,
        random_indices=args.random_indices,
        shuffle_within_dataset=args.shuffle_within_dataset,
        _run_tests=args.run_tests,
    )
