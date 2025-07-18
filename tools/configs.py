from pathlib import Path
import os
from loguru import logger

HF_NAME = os.environ.get("HF_NAME", "science-of-finetuning")
VERSION = "040225"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    Path(os.environ["DATASTORE"]) if "DATASTORE" in os.environ else REPO_ROOT / "data"
)
if "DATASTORE" not in os.environ:
    logger.info(f"DATASTORE not in environment variables, using {DATA_ROOT}")
PLOTS = DATA_ROOT / "plots"

MODEL_CONFIGS = {
    "Qwen/Qwen2.5-1.5B": {
        "ignore_first_n_tokens_per_sample": 21,
        "text_column": "text_qwen2_5",
        "attn_implementation": None,
        "token_level_replacement": None,
    },
    "google/gemma-2-2b": {
        "ignore_first_n_tokens_per_sample": 0,
        "text_column": "text",
        "attn_implementation": "eager",
        "token_level_replacement": None,
    },
    "meta-llama/Meta-Llama-3.1-8B": {
        "ignore_first_n_tokens_per_sample": 25,
        "text_column": "text_llama3",
        "attn_implementation": None,
        "token_level_replacement": None,
    },
    "meta-llama/Llama-3.2-1B": {
        "ignore_first_n_tokens_per_sample": 25,
        "text_column": "text_llama3",
        "attn_implementation": None,
        "token_level_replacement": None,
    },
}
MODEL_CONFIGS["google/gemma-2-2b-it"] = MODEL_CONFIGS["google/gemma-2-2b"]
MODEL_CONFIGS["Qwen/Qwen2.5-1.5B-Instruct"] = MODEL_CONFIGS["Qwen/Qwen2.5-1.5B"]
MODEL_CONFIGS["meta-llama/Meta-Llama-3.1-8B-Instruct"] = MODEL_CONFIGS[
    "meta-llama/Meta-Llama-3.1-8B"
].copy()
MODEL_CONFIGS["meta-llama/Llama-3.2-1B-Instruct"] = MODEL_CONFIGS[
    "meta-llama/Llama-3.2-1B"
].copy()
MODEL_CONFIGS["meta-llama/Meta-Llama-3.1-8B"]["token_level_replacement"] = {
    128006: 1432,
    128009: 827,
    128007: 827,
}  # Llama 3.1 Base doesn't deal well with template tokens
MODEL_CONFIGS["meta-llama/Llama-3.2-1B"]["token_level_replacement"] = {
    128006: 1432,
    128009: 827,
    128007: 827,
}  # Llama 3.2 Base doesn't deal well with template tokens
