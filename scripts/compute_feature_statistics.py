import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.cache import ActivationCache
from dictionary_learning.dictionary import CrossCoder
from datasets import load_from_disk, load_dataset
from loguru import logger
import torch as th
from nnsight import LanguageModel
from pathlib import Path
from pycolors import TailwindColorPalette
import os
import argparse
import json
from dlabutils import model_path

from tools.feature_analysis import get_features, feature_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=str, default="checkpoints/l13-mu4.0e-02-lr1e-04/ae_90000.pt")
    parser.add_argument("--layer", type=int, default=13)
    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(model_path("google/gemma-2-2b"), torch_dtype=th.bfloat16, device_map="cuda", attn_implementation="eager")
    instruction_model = AutoModelForCausalLM.from_pretrained(model_path("google/gemma-2-2b-it"), torch_dtype=th.bfloat16, device_map="cuda", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_path("google/gemma-2-2b-it"))

    cc = CrossCoder.from_pretrained(args.cc, device="cuda").to("cuda")

    validation_set_base = load_from_disk("datasets/validation/fineweb_100m_sample")
    validation_set_chat = load_from_disk("datasets/validation/lmsys_chat")

    batch_size = 16
    subset = 10000
    stats_fineweb = feature_statistics(validation_set_base.select_columns("text").select(range(subset)), tokenizer, base_model, instruction_model, cc, args.layer, batch_size=batch_size)
    stats_lmsys = feature_statistics(validation_set_chat.select_columns("text").select(range(subset)), tokenizer, base_model, instruction_model, cc, args.layer, batch_size=batch_size)

    import os
    os.makedirs(f"stats/{args.cc.split('/')[-2]}", exist_ok=True)
    # save args
    with open(f"stats/{args.cc.split('/')[-2]}/args.json", "w") as f:
        json.dump(args.__dict__, f)
    th.save(stats_fineweb, f"stats/{args.cc.split('/')[-2]}/fineweb.pt")
    th.save(stats_lmsys, f"stats/{args.cc.split('/')[-2]}/lmsys.pt")