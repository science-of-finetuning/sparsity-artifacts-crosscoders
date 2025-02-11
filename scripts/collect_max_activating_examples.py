"""
Script to collect max activating examples for a base and chat only latents of a CrossCoder.
"""

from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
import heapq
import argparse
import gc
import sqlite3
import json
import sys
import os
import torch as th
from torch.utils.data import DataLoader
from dictionary_learning import CrossCoder
from datasets import load_dataset
from nnterp.nnsight_utils import get_layer_output, get_layer
from nnterp import load_model
from tqdm import tqdm
from huggingface_hub import hf_api, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import wandb

sys.path.append(".")

from tools.utils import load_crosscoder, load_latent_df, df_hf_repo


def max_act_exs_to_db(max_activating_examples, db_path: Path, overwrite=False):
    """Convert max activating examples to a database."""
    if not db_path.exists() or overwrite:
        if db_path.exists():
            os.remove(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS data_table (
                    key INTEGER PRIMARY KEY,
                    examples TEXT
                )"""
            )
            for key, examples in max_activating_examples.items():
                cursor.execute(
                    "INSERT INTO data_table (key, examples) VALUES (?, ?)",
                    (key, json.dumps(examples)),
                )
            conn.commit()


def sort_max_act_exs(max_activating_examples):
    """Sort max activating examples by activation value."""
    for feature_idx in max_activating_examples:
        max_activating_examples[feature_idx] = sorted(
            [(t[0], t[2], t[3]) for t in max_activating_examples[feature_idx]],
            key=lambda x: x[0],
            reverse=True,
        )
    return max_activating_examples


def cleanup_max_act_exs(max_activating, tokenizer, max_seq_len):
    """Clean up max activating examples by removing padding values and truncating sequences."""
    for feature_idx, examples in tqdm(max_activating.items()):
        for i, (ex_val, ex_str, ex_act) in enumerate(examples):
            # Remove padding values (-1) from the beginning of ex_act
            while ex_act and ex_act[0] == -1:
                ex_act.pop(0)
            examples[i] = (ex_val, ex_str, ex_act)
            # Truncate the end of ex_str to match the length of ex_act
            tokens = tokenizer.tokenize(ex_str, add_special_tokens=True)[:max_seq_len]
            tokens = [s.replace("▁", " ") for s in tokens]
            if len(tokens) != len(ex_act):
                print(
                    f"Warning: length of tokens {len(tokens)} does not match length of activation values {len(ex_act)} for example {i} of feature {feature_idx}"
                )
            examples[i] = (ex_val, tokens, ex_act)
    return max_activating


def merge_max_examples(*max_dicts, n=None):
    """Merge max activating examples from multiple dictionaries."""
    merged_dict = {}
    if n is None:
        raise ValueError("n must be provided")
    for d in max_dicts:
        for k, v in d.items():
            merged_dict[k] = merged_dict.get(k, []) + v
    # sort each list
    for k in merged_dict:
        merged_dict[k] = sorted(merged_dict[k], key=lambda x: x[0], reverse=True)[:n]
    return merged_dict


@th.no_grad()
def compute_max_activating_examples(
    dataset,
    feature_indices,
    crosscoder: CrossCoder,
    *,
    base_model,
    chat_model,
    save_path: Path,
    model_batch_size=64,
    crosscoder_batch_size=2048,
    n=100,
    layer=13,
    cc_device="cuda",
    workers=12,
    max_seq_len=1024,
    name="max_activating_examples",
    gc_collect_every=1000,
    checkpoint_every=1000,
) -> None:
    """Compute examples that maximally activate each feature in a CrossCoder model.

    Args:
        dataset: Dataset to search through for examples
        feature_indices: List of feature indices to find max activating examples for
        crosscoder: CrossCoder model to use for computing feature activations
        base_model: Base language model to get activations from
        chat_model: Chation-tuned model to get activations from
        save_path: Path to save results to
        model_batch_size: Batch size for running examples through base/chat models
        crosscoder_batch_size: Batch size for running activations through CrossCoder
        n: Number of max activating examples to find per feature
        layer: Which model layer to get activations from
        device: Device to run models on
        workers: Number of worker processes for data loading
        max_seq_len: Maximum sequence length to consider
        name: Name for saving results
        gc_collect_every: How often to run garbage collection
        checkpoint_every: How often to save checkpoints

    Returns:
        None. Results are saved to save_path/name.
    """
    if save_path is not None:
        save_path = save_path / name
        save_path.mkdir(parents=True, exist_ok=True)

    def dict_update_worker(
        queue, n, feature_indices, save_path, name, tokenizer, max_seq_len
    ):
        entry_id = 0  # entry id ensures we never compare the feat_act
        max_activating_examples = {k: [] for k in feature_indices}
        next_gb = gc_collect_every
        next_checkpoint = checkpoint_every
        num_samples = 0
        while True:
            item = queue.get()
            if item is None:  # Poison pill to stop the worker
                break
            max_activations, batch, feature_activations = item
            next_checkpoint -= 1
            next_gb -= 1
            num_samples += len(batch)
            if next_checkpoint <= 0 and save_path is not None:
                th.save(max_activating_examples, save_path / f"{num_samples}.pt")
                next_checkpoint = checkpoint_every
            if next_gb <= 0:
                gc.collect()
                next_gb = gc_collect_every
            # Dictionary update logic (runs in separate process)
            for idx, feature_idx in enumerate(feature_indices):
                batch_values = max_activations[:, idx]
                entries = list(
                    zip(
                        batch_values.tolist(),
                        batch,
                        [feat_act[:, idx] for feat_act in feature_activations],
                    )
                )

                if len(max_activating_examples[feature_idx]) < n:
                    threshold = float("-inf")
                else:
                    threshold = max_activating_examples[feature_idx][0][0]

                potential_entries = [entry for entry in entries if entry[0] > threshold]

                for entry in potential_entries:
                    if len(max_activating_examples[feature_idx]) < n:
                        heapq.heappush(
                            max_activating_examples[feature_idx],
                            (entry[0], entry_id, entry[1], entry[2].tolist()),
                        )
                        entry_id += 1
                    else:
                        heapq.heappushpop(
                            max_activating_examples[feature_idx],
                            (entry[0], entry_id, entry[1], entry[2].tolist()),
                        )
                        entry_id += 1

        max_activating_examples = sort_max_act_exs(max_activating_examples)
        max_activating_examples = cleanup_max_act_exs(
            max_activating_examples, tokenizer, max_seq_len
        )
        th.save(max_activating_examples, save_path / f"{name}_final.pt")

    # Setup multiprocessing with bounded queue
    queue = Queue(maxsize=10)
    update_process = Process(
        target=dict_update_worker,
        args=(
            queue,
            n,
            feature_indices,
            save_path,
            name,
            chat_model.tokenizer,
            max_seq_len,
        ),
    )
    update_process.start()

    crosscoder.encoder.to(cc_device)
    dataloader = DataLoader(dataset, batch_size=model_batch_size, num_workers=workers)
    dec_weight = (
        crosscoder.decoder.weight.norm(dim=2).sum(dim=0, keepdim=True).to(cc_device)
    )
    num_tokens = 0

    # Main loop - now just collecting activations and queuing updates
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Check if the worker process is still alive
        if not update_process.is_alive():
            raise RuntimeError("dict_update_worker process crashed unexpectedly.")

        bs = len(batch)
        tokens = base_model.tokenizer(
            batch,
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(base_model.device)
        attention_mask = tokens["attention_mask"]

        with base_model.trace(tokens):
            base_activations = get_layer_output(base_model, layer).to(cc_device).save()
            get_layer(base_model, layer).output.stop()
        with chat_model.trace(tokens):
            chat_activations = get_layer_output(chat_model, layer).to(cc_device).save()
            get_layer(chat_model, layer).output.stop()

        base_activations = base_activations.reshape(-1, base_activations.shape[-1])
        chat_activations = chat_activations.reshape(-1, chat_activations.shape[-1])
        merged_activations = th.stack([base_activations, chat_activations], dim=1)

        feature_activations = []
        for act_batch in merged_activations.split(crosscoder_batch_size):
            feature_activations.append(
                (crosscoder.encoder(act_batch.float()) * dec_weight)[:, feature_indices]
            )
        feature_activations = th.cat(feature_activations, dim=0)
        feature_activations = feature_activations.reshape(bs, -1, len(feature_indices))
        feature_activations = feature_activations.masked_fill(
            ~attention_mask.bool().unsqueeze(-1), -1
        )

        max_activations, _ = feature_activations.max(dim=1)
        num_tokens += attention_mask.sum().item()
        # Log metrics to wandb
        wandb.log(
            {
                "batch": batch_idx,
                "mean_activation": max_activations.mean().item(),
                "max_activation": max_activations.max().item(),
                "min_activation": max_activations.min().item(),
                "queue_size": queue.qsize(),
                "num_tokens": num_tokens,
            }
        )

        # Queue the data for processing
        queue.put((max_activations.cpu(), batch, feature_activations.cpu()))

    # Signal the worker to finish and get results
    queue.put(None)
    update_process.join()

    # Check if the worker process crashed
    if update_process.exitcode != 0:
        raise RuntimeError(
            "dict_update_worker process crashed with exit code: "
            f"{update_process.exitcode}"
        )

    return


# python scripts/collect_max_activating_examples.py l13_crosscoder --base-device "cuda:0" --chat-device "cuda:1" --crosscoder-batch-size 512 --model-batch-size 16 --latent-file "notebooks/results/shared_not_shared_indices.json"  --no-upload
# python scripts/collect_max_activating_examples.py connor --base-device "cuda:0" --chat-device "cuda:1" --lmsys-format base  --crosscoder-batch-size 512 --model-batch-size 16
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("crosscoder", type=str)
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--cc-device", type=str, default="cuda")
    parser.add_argument("--base-device", type=str, default="cuda")
    parser.add_argument("--chat-device", type=str, default="cuda")
    parser.add_argument("--validation-size", type=int, default=10**6)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--crosscoder-batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--only-upload", action="store_true")
    parser.add_argument("--no-upload", action="store_false", dest="upload")
    parser.add_argument("--latent-file", type=Path, default=None)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("results/max_activating_examples"),
    )
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--lmsys-format", default="chat", choices=["chat", "base"])
    args = parser.parse_args()
    save_path = args.save_path
    if args.testing:
        save_path = save_path / "testing"
    save_path = save_path / args.crosscoder
    if not args.only_upload:
        if args.workers is None:
            args.workers = cpu_count()

        # Initialize wandb
        wandb.init(project="max-activating-examples", config=vars(args))

        crosscoder = load_crosscoder(args.crosscoder)
        if args.latent_file is not None:
            selected_indices = json.load(args.latent_file.open("r"))
        else:
            df = load_latent_df(args.crosscoder)
            selected_latents = df[
                (df["tag"].isin(["IT only", "Base only", "Chat only"]))
            ]
            selected_indices = selected_latents.index.tolist()

        # Load datasets
        test_set_base = load_dataset(
            "science-of-finetuning/fineweb-1m-sample", split="validation"
        )["text"]
        lmsys_column = "text" if args.lmsys_format == "chat" else "text_base_format"
        test_set_chat = load_dataset(
            "science-of-finetuning/lmsys-chat-1m-gemma-formatted",
            split="validation",
        )[lmsys_column]
        test_set_base = test_set_base[: len(test_set_base) // 2]
        test_set_chat = test_set_chat[: len(test_set_chat) // 2]
        if args.testing:
            test_set_base = test_set_base[:100]
            test_set_chat = test_set_chat[:100]

        # Load models
        base_model = load_model(
            args.base_model,
            torch_dtype=th.bfloat16,
            attn_implementation="eager",
            dispatch=True,
            device_map=args.base_device,
        )
        chat_model = load_model(
            args.chat_model,
            torch_dtype=th.bfloat16,
            attn_implementation="eager",
            dispatch=True,
            device_map=args.chat_device,
        )

        # Create save directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate and save max activating examples
        print("Generating mini examples...")
        compute_max_activating_examples(
            test_set_chat[:100],
            selected_indices,
            crosscoder,
            model_batch_size=args.model_batch_size,
            crosscoder_batch_size=args.crosscoder_batch_size,
            n=args.n,
            base_model=base_model,
            chat_model=chat_model,
            layer=args.layer,
            cc_device=args.cc_device,
            workers=args.workers,
            max_seq_len=args.seq_len,
            save_path=save_path,
            checkpoint_every=1,
            name="mini-chat",
        )

        print("Generating chat examples...")
        compute_max_activating_examples(
            test_set_chat,
            selected_indices,
            crosscoder,
            model_batch_size=args.model_batch_size,
            crosscoder_batch_size=args.crosscoder_batch_size,
            n=args.n,
            base_model=base_model,
            chat_model=chat_model,
            layer=args.layer,
            cc_device=args.cc_device,
            workers=args.workers,
            max_seq_len=args.seq_len,
            save_path=save_path,
            name="chat",
        )

        print("Generating base examples...")
        compute_max_activating_examples(
            test_set_base,
            selected_indices,
            crosscoder,
            model_batch_size=args.model_batch_size,
            crosscoder_batch_size=args.crosscoder_batch_size,
            n=args.n,
            base_model=base_model,
            chat_model=chat_model,
            layer=args.layer,
            cc_device=args.cc_device,
            workers=args.workers,
            max_seq_len=args.seq_len,
            save_path=save_path,
            name="base",
        )

        wandb.finish()
    repo = df_hf_repo[args.crosscoder]

    # Load new examples
    chat_examples = th.load(save_path / "chat/chat_final.pt")
    base_examples = th.load(save_path / "base/base_final.pt")
    chat_base_examples = merge_max_examples(chat_examples, base_examples, n=args.n)
    # Upload files to HF
    for file, file_name, examples in [
        ("base/base_final", "base_examples", base_examples),
        ("chat/chat_final", "chat_examples", chat_examples),
        ("chat_base_examples", "chat_base_examples", chat_base_examples),
    ]:
        # Try to merge with existing examples from HF
        try:
            dl_path = hf_hub_download(
                repo_id=repo, filename=f"{file_name}.pt", repo_type="dataset"
            )
            existing_examples = th.load(dl_path)
            examples = merge_max_examples(examples, existing_examples, n=args.n)
        except EntryNotFoundError:
            print(f"No existing file for {file_name}")

        # Save updated results
        th.save(examples, save_path / f"{file}.pt")
        max_act_exs_to_db(examples, save_path / f"{file}.db", overwrite=True)
        if args.upload and not args.testing:
            # Upload to HF
            for ftype in ["pt", "db"]:
                file_path = save_path / f"{file}.{ftype}"
                hf_api.upload_file(
                    repo_id=repo,
                    repo_type="dataset",
                    path_or_fileobj=file_path,
                    path_in_repo=f"{file_name}.{ftype}",
                )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
