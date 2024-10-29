import torch as th
from pathlib import Path
from torch.utils.data import DataLoader
from dictionary_learning import CrossCoder
from datasets import load_from_disk
from nnterp.nnsight_utils import get_layer_output, get_layer
from nnterp import load_model
from multiprocessing import Process, Queue
import heapq
from tqdm import tqdm
import argparse
import pandas as pd
import wandb

from multiprocessing import Process, Queue
from torch.utils.data import DataLoader
from dictionary_learning.dictionary import CrossCoder
import heapq
import gc
from transformers import AutoTokenizer


def clean_up_max_activating_examples(max_activating_examples):
    # Sort before saving
    for feature_idx in max_activating_examples:
        max_activating_examples[feature_idx] = sorted(
            [(t[0], t[2], t[3]) for t in max_activating_examples[feature_idx]],
            key=lambda x: x[0],
            reverse=True,
        )
    return max_activating_examples


def clean_up_max_activating_examples2(max_activating, tokenizer, max_seq_len):
    # clean up the max_activating_ex_mini
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


def merge_max_examples(*max_dicts):
    merged_dict = {}
    for d in max_dicts:
        for k, v in d.items():
            merged_dict[k] = merged_dict.get(k, []) + v
    # sort each list
    for k in merged_dict:
        merged_dict[k] = sorted(merged_dict[k], key=lambda x: x[0], reverse=True)
    return merged_dict


@th.no_grad()
def compute_max_activating_examples(
    dataset,
    feature_indices,
    crosscoder: CrossCoder,
    *,
    base_model,
    instruct_model,
    save_path: Path,
    model_batch_size=64,
    crosscoder_batch_size=2048,
    n=100,
    layer=13,
    device="cuda",
    workers=12,
    max_seq_len=1024,
    name="max_activating_examples",
    gc_collect_every=1000,
    checkpoint_every=1000,
) -> None:
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

        max_activating_examples = clean_up_max_activating_examples(
            max_activating_examples
        )
        max_activating_examples = clean_up_max_activating_examples2(
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
            instruct_model.tokenizer,
            max_seq_len,
        ),
    )
    update_process.start()

    crosscoder.encoder.to(device)
    dataloader = DataLoader(
        dataset["text"], batch_size=model_batch_size, num_workers=workers
    )
    dec_weight = (
        crosscoder.decoder.weight.norm(dim=2).sum(dim=0, keepdim=True).to(device)
    )

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
            base_activations = get_layer_output(base_model, layer).save()
            get_layer(base_model, layer).output.stop()
        with instruct_model.trace(tokens):
            instruct_activations = get_layer_output(instruct_model, layer).save()
            get_layer(instruct_model, layer).output.stop()

        base_activations = base_activations.reshape(-1, base_activations.shape[-1])
        instruct_activations = instruct_activations.reshape(
            -1, instruct_activations.shape[-1]
        )
        merged_activations = th.stack([base_activations, instruct_activations], dim=1)

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

        # Log metrics to wandb
        wandb.log(
            {
                "batch": batch_idx,
                "mean_activation": max_activations.mean().item(),
                "max_activation": max_activations.max().item(),
                "min_activation": max_activations.min().item(),
                "queue_size": queue.qsize(),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crosscoder-path",
        type=str,
        default="/dlabscratch1/jminder/repositories/representation-structure-comparison/checkpoints/l13-mu4.1e-02-lr1e-04/ae_final.pt",
    )
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--instruct-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--validation-size", type=int, default=10**6)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--crosscoder-batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path(
            "/dlabscratch1/cdumas/representation-structure-comparison/results-runai/max_activating_examples"
        ),
    )
    parser.add_argument(
        "--feature-df-path",
        type=Path,
        default=Path(
            "/dlabscratch1/cdumas/representation-structure-comparison/notebooks/results/eval_crosscoder/l13-mu4.1e-02-lr1e-04_ae_final/data/feature_df.csv"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "/dlabscratch1/jminder/repositories/representation-structure-comparison/datasets"
        ),
    )

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="max-activating-examples", config=vars(args))

    # Load CrossCoder
    crosscoder = CrossCoder.from_pretrained(args.crosscoder_path)

    # Load feature dataframe and get selected indices
    df = pd.read_csv(args.feature_df_path)
    selected_features = df[
        (df["tag"].isin(["IT only", "Base only"])) & (df["dead"] == False)
    ]
    selected_indices = selected_features.index.tolist()

    # Load datasets
    test_set_base = load_from_disk(args.dataset_dir / "test/fineweb_100m_sample")
    test_set_chat = load_from_disk(args.dataset_dir / "test/lmsys_chat")
    test_set_base = test_set_base.select(range(len(test_set_base) // 2))
    test_set_chat = test_set_chat.select(range(len(test_set_chat) // 2))

    # Load models
    base_model = load_model(
        args.base_model,
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
        dispatch=True,
        device_map=args.device,
    )
    instruct_model = load_model(
        args.instruct_model,
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
        dispatch=True,
        device_map=args.device,
    )

    # Create save directory if it doesn't exist
    args.save_path.mkdir(parents=True, exist_ok=True)

    # Generate and save max activating examples
    print("Generating mini examples...")
    compute_max_activating_examples(
        test_set_chat.select(range(100)),
        selected_indices,
        crosscoder,
        model_batch_size=args.model_batch_size,
        crosscoder_batch_size=args.crosscoder_batch_size,
        n=args.n,
        base_model=base_model,
        instruct_model=instruct_model,
        layer=args.layer,
        device=args.device,
        workers=args.workers,
        max_seq_len=args.seq_len,
        save_path=args.save_path,
        checkpoint_every=1,  # check if this works
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
        instruct_model=instruct_model,
        layer=args.layer,
        device=args.device,
        workers=args.workers,
        max_seq_len=args.seq_len,
        save_path=args.save_path,
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
        instruct_model=instruct_model,
        layer=args.layer,
        device=args.device,
        workers=args.workers,
        max_seq_len=args.seq_len,
        save_path=args.save_path,
        name="base",
    )

    wandb.finish()


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
