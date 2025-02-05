import torch as th
from pathlib import Path
import numpy as np

from tools.cc_utils import CCLatent


def load_betas(
    betas_dir_path: Path,
    num_samples: int = 50_000_000,
    computation: str = "base_error",
    n_offset: int = 0,
    suffix: str = "",
):
    betas_filename = f"betas_{computation}_N{num_samples}_n_offset{n_offset}{suffix}.pt"
    count_active_filename = (
        f"count_active_{computation}_N{num_samples}_n_offset{n_offset}{suffix}.pt"
    )
    betas = th.load(betas_dir_path / betas_filename, weights_only=True).cpu()
    count_active = th.load(
        betas_dir_path / count_active_filename, weights_only=True
    ).cpu()
    return betas, count_active


def load_stats(
    betas_dir_path: Path,
    num_samples: int,
    n_offset: int = 0,
    suffix: str = "",
    sgd_name: str = "",
    computation: str = "base_error",
    train: bool = False,
):

    split_suffix = "_train" if train else "_validation"
    if sgd_name:
        stats_path = (
            betas_dir_path
            / f"stats_{computation}_sgd_{sgd_name}_N{num_samples}_n_offset{n_offset}{suffix}{split_suffix}.pt"
        )
    else:
        stats_path = (
            betas_dir_path
            / f"stats_{computation}_N{num_samples}_n_offset{n_offset}{suffix}{split_suffix}.pt"
        )
    return th.load(stats_path)


def get_latent_from_idx(
    idx, array: th.Tensor, chat_indices: th.Tensor, return_latent: bool = True
):
    if isinstance(array, th.Tensor):
        array = array.cpu()
    nan_idx_to_original_idx = th.arange(len(array))[~np.isnan(array)]
    if return_latent:
        return CCLatent(chat_indices[nan_idx_to_original_idx[idx]].item())
    else:
        return nan_idx_to_original_idx[idx]


def get_beta_from_index(index: int, betas: th.Tensor, chat_indices: th.Tensor):
    index = chat_indices.tolist().index(index)
    return betas[index]
