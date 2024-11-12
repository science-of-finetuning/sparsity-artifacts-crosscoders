import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.cache import ActivationCache
from datasets import load_from_disk, load_dataset
from loguru import logger
import torch as th
from nnsight import LanguageModel
from pathlib import Path
import os


DATA_PATH = Path(__file__).parent.parent / "datasets"
CHAT_DATA_PATH = DATA_PATH / "lmsys-chat-1m-formatted"

NUM_ROWS = 10**5
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")


# LMSYS Chat
chat_dataset = load_from_disk(CHAT_DATA_PATH)

# FineWeb
base_dataset = load_dataset(
    "HuggingFaceFW/fineweb", name="sample-10BT", split="train"
).select(range(10**6))

# find index for 100m tokens
idx = 0
num_toks = 0
while num_toks < 10**8:
    num_toks += len(
        tokenizer(base_dataset[idx]["text"], max_length=1024, truncation=True)[
            "input_ids"
        ]
    )
    idx += 1
    print(num_toks)
base_idx = idx
base_dataset.select(range(base_idx + 1, base_idx + 1 + NUM_ROWS)).save_to_disk(
    "datasets/validation/fineweb_100m_sample"
)


# find index for 100m tokens
idx = 0
num_toks = 0
while num_toks < 10**8:
    num_toks += len(
        tokenizer(chat_dataset[idx]["text"], max_length=1024, truncation=True)[
            "input_ids"
        ]
    )
    idx += 1
    print(num_toks)
chat_idx = idx
chat_dataset.select(range(chat_idx + 1, chat_idx + 1 + NUM_ROWS)).save_to_disk(
    "datasets/validation/lmsys_chat"
)
