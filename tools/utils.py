from pathlib import Path

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
