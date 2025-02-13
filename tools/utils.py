from pathlib import Path
from typing import List
from dictionary_learning.cache import PairedActivationCache

from tools.compute_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tools.cc_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tools.plotting_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tools.tokenization_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import

def apply_masks(values: th.Tensor, masks: List[th.Tensor]) -> th.Tensor:
    """
    Apply the masks to the indices.
    """
    if isinstance(masks, th.Tensor):
        masks = [masks]
    for mask in masks:
        values = values[mask]
    return values




def load_activation_dataset(
    activation_store_dir: Path,
    base_model: str = "gemma-2-2b",
    instruct_model: str = "gemma-2-2b-it",
    lmsys_subfolder: str = None,
    fineweb_subfolder: str = None,
    layer: int = 13,
    split="validation",
    lmsys_split: str = None,
    fineweb_split: str = None,
    lmsys_name: str = "lmsys-chat-1m-gemma-formatted",
    fineweb_name: str = "fineweb-1m-sample",
):
    """
    Load the saved activations of the base and instruct models for a given layer

    Args:
        activation_store_dir: The directory where the activations are stored
        base_model: The base model to load
        instruct_model: The instruct model to load
        lmsys_subfolder: The subfolder to find the lmsys dataset in (activation_store_dir/lmsys_subfolder/model/split)
        fineweb_subfolder: The subfolder to find the fineweb dataset in (activation_store_dir/fineweb_subfolder/model/split)
        layer: The layer to load
        split: The split to load
        lmsys_split: The split to load for the lmsys dataset (overrides split)
        fineweb_split: The split to load for the fineweb dataset (overrides split)
    """
    if lmsys_split is None:
        lmsys_split = split
    if fineweb_split is None:
        fineweb_split = split

    # Load validation dataset
    activation_store_dir = Path(activation_store_dir)
    if lmsys_subfolder is None:
        base_model_dir_lmsys = activation_store_dir / base_model
        instruct_model_dir_lmsys = activation_store_dir / instruct_model
    else:
        base_model_dir_lmsys = activation_store_dir / lmsys_subfolder / base_model
        instruct_model_dir_lmsys = activation_store_dir / lmsys_subfolder / instruct_model

    if fineweb_subfolder is None:
        base_model_dir_fineweb = activation_store_dir / base_model
        instruct_model_dir_fineweb = activation_store_dir / instruct_model
    else:
        base_model_dir_fineweb = fineweb_subfolder / base_model
        instruct_model_dir_fineweb = fineweb_subfolder / instruct_model

    submodule_name = f"layer_{layer}_out"

    # Load validation caches
    base_model_fineweb = base_model_dir_fineweb / fineweb_name / fineweb_split
    instruct_model_fineweb = instruct_model_dir_fineweb / fineweb_name / fineweb_split
    
    base_model_lmsys = base_model_dir_lmsys / lmsys_name / lmsys_split
    instruct_model_lmsys = instruct_model_dir_lmsys / lmsys_name / lmsys_split
    
    print(f"Loading fineweb cache from {base_model_fineweb / submodule_name} and {instruct_model_fineweb / submodule_name}")
    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    print(f"Loading lmsys cache from {base_model_lmsys / submodule_name} and {instruct_model_lmsys / submodule_name}")

    lmsys_cache = PairedActivationCache(
        base_model_lmsys / submodule_name, instruct_model_lmsys / submodule_name
    )

    return fineweb_cache, lmsys_cache
