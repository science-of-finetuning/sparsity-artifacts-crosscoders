import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.cache import ActivationCache
from datasets import load_from_disk, load_dataset
from loguru import logger
import torch as th
from nnsight import LanguageModel
from dlabutils import model_path
from pathlib import Path
import os

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activation-store-dir", type=str, required=True)
    parser.add_argument(
        "--chat-dataset-path",
        type=Path,
        default="./datasets/test/lmsys_chat",
        help="Path to the chat dataset which should contain a 'text' column with chats converted to text using your chat template",
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        model_path(args.model),
        device_map="auto",
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path(args.model))
    model = LanguageModel(model, tokenizer=tokenizer)
    num_layers = int(len(model.model.layers))
    layers = [int(num_layers * i) for i in (0.2, 0.5, 0.8)]
    logger.info(f"Collecting activations from layers: {layers}")

    submodules = [model.model.layers[layer] for layer in layers]
    submodule_names = ["layer_{}".format(layer) for layer in layers]

    d_model = model.config.hidden_size
    logger.info(f"d_model={d_model}")

    store_dir = Path(args.activation_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    # LMSYS Chat
    chat_dataset = load_from_disk(args.chat_dataset_path)
    out_dir = store_dir / "lmsys_chat"
    ActivationCache.collect(
        chat_dataset["text"],
        submodules,
        submodule_names,
        model,
        out_dir,
        shuffle_shards=False,
        io="out",
        shard_size=10**6,
        batch_size=16,
        context_len=1024,
        d_model=d_model,
        last_submodule=submodules[-1],
    )

    # # FineWeb
    base_dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train"
    ).select(range(10**6))
    out_dir = store_dir / "fineweb"
    ActivationCache.collect(
        base_dataset["text"],
        submodules,
        submodule_names,
        model,
        out_dir,
        shuffle_shards=False,
        io="out",
        shard_size=10**6,
        batch_size=16,
        context_len=1024,
        d_model=d_model,
        last_submodule=submodules[-1],
    )
