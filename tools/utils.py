from pathlib import Path

import torch as th

from dictionary_learning.cache import PairedActivationCache

from tools.compute_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tools.cc_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tools.plotting_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tools.tokenization_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import


def load_activation_dataset(
    activation_store_dir: Path,
    base_model: str = "gemma-2-2b",
    instruct_model: str = "gemma-2-2b-it",
    layer: int = 13,
    split="validation",
):
    """
    Load the saved activations of the base and instruct models for a given layer

    Args:
        activation_store_dir: The directory where the activations are stored
        base_model: The base model to load
        instruct_model: The instruct model to load
        layer: The layer to load
        split: The split to load
    """
    # Load validation dataset
    activation_store_dir = Path(activation_store_dir)
    base_model_dir = activation_store_dir / base_model
    instruct_model_dir = activation_store_dir / instruct_model

    submodule_name = f"layer_{layer}_out"

    # Load validation caches
    base_model_fineweb = base_model_dir / "fineweb-1m-sample" / split
    base_model_lmsys = base_model_dir / "lmsys-chat-1m-gemma-formatted" / split
    instruct_model_fineweb = instruct_model_dir / "fineweb-1m-sample" / split
    instruct_model_lmsys = instruct_model_dir / "lmsys-chat-1m-gemma-formatted" / split

    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_cache = PairedActivationCache(
        base_model_lmsys / submodule_name, instruct_model_lmsys / submodule_name
    )

    return fineweb_cache, lmsys_cache


def mask_k_first_ones_vec(bool_tensor, k):
    """
    Returns a mask where only the k first of each contiguous sequence of ones are kept.

    For each row in the input boolean tensor, each contiguous segment of ones (True)
    is processed such that only the first k ones (by their order in the sequence) remain True.
    If the number of ones in a segment is smaller than k, all remain True.

    Args:
        bool_tensor (torch.Tensor): A boolean tensor of shape (batch_size, sequence_length)
            containing sequences of 0s and 1s (represented as False and True).
        k (int): The number of first ones to keep in each contiguous sequence.

    Returns:
        torch.Tensor: A boolean tensor mask of the same shape as bool_tensor.
    """
    # Validate inputs
    assert bool_tensor.dtype == th.bool, "Input tensor must be of bool type"
    assert isinstance(k, int) and k > 0, "k must be a positive integer"

    batch_size, seq_len = bool_tensor.shape
    # Create an index tensor for the sequence positions.
    idx = (
        th.arange(seq_len, device=bool_tensor.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # For false positions, we store their index.
    # For true positions, assign -1 so that they don't contribute to the "last seen zero".
    temp = th.where(~bool_tensor, idx, th.full_like(idx, -1))

    # Compute the cumulative maximum along the sequence dimension.
    # At each position, this gives the index of the most recent False (or -1 if none).
    last_zero = th.cummax(temp, dim=1)[0]

    # For each position, calculate the relative position from the most recent False.
    # In a contiguous run of True values, the first element has a relative index of 1.
    rel_idx = idx - last_zero

    # Keep only positions within the first k ones in each run.
    result_mask = bool_tensor & (rel_idx <= k)
    return result_mask
