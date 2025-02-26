from pathlib import Path
from argparse import ArgumentParser
import sys

from tqdm import tqdm
import torch as th
import numpy as np
from dictionary_learning.cache import PairedActivationCache
from dictionary_learning import CrossCoder

sys.path.append(str(Path(__file__).parent.parent))
from tools.utils import (
    save_json,
    load_activation_dataset,
    load_crosscoder,
    load_latent_df,
    push_latent_df,
)

# th.set_float32_matmul_precision("high")


@th.no_grad()
def compute_max_activation(
    crosscoder: CrossCoder, cache: PairedActivationCache, device, batch_size: int = 2048
):
    dataloader = th.utils.data.DataLoader(cache, batch_size=batch_size)
    max_activations = th.zeros(crosscoder.dict_size, device=device)
    for batch in tqdm(dataloader):
        activations = crosscoder.get_activations(
            batch.to(device)
        )  # (batch_size, dict_size)
        max_activations = th.max(max_activations, activations.max(dim=0).values)
    assert max_activations.shape == (crosscoder.dict_size,)
    return max_activations.cpu()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--activation-cache-path", "-p", type=Path, default="./activations"
    )
    parser.add_argument("--dataset", default="lmsys-chat-1m-chat-formatted")
    parser.add_argument("--crosscoder", type=str, default="l13_crosscoder")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    crosscoder = load_crosscoder(args.crosscoder).to(args.device)
    results = {}
    for split in ["train", "validation"]:
        fw_dataset, lmsys_dataset = load_activation_dataset(
            args.activation_cache_path,
            base_model="gemma-2-2b",
            instruct_model="gemma-2-2b-it",
            layer=args.layer,
            split=split,
        )

        max_activations_fw = compute_max_activation(
            crosscoder, fw_dataset, args.device, args.batch_size
        )
        max_activations_lmsys = compute_max_activation(
            crosscoder, lmsys_dataset, args.device, args.batch_size
        )
        results[split] = {
            "max_activations_fw": max_activations_fw.tolist(),
            "max_activations_lmsys": max_activations_lmsys.tolist(),
        }
    path = Path("results") / f"max_acts_{args.crosscoder}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, path)
    df = load_latent_df()
    # add max_act_train, max_act_val, max_act_train_lmsys, max_act_val_lmsys, max_act_train_fw, max_act_val_fw
    df["max_act_train"] = np.maximum(
        results["train"]["max_activations_fw"],
        results["train"]["max_activations_lmsys"],
    )
    df["max_act_val"] = np.maximum(
        results["validation"]["max_activations_fw"],
        results["validation"]["max_activations_lmsys"],
    )
    df["max_act_lmsys_train"] = results["train"]["max_activations_lmsys"]
    df["max_act_lmsys_val"] = results["validation"]["max_activations_lmsys"]
    df["max_act_fw_train"] = results["train"]["max_activations_fw"]
    df["max_act_fw_val"] = results["validation"]["max_activations_fw"]
    push_latent_df(
        df,
        commit_message="Added max activations for all splits and datasets",
        confirm=False,
    )
