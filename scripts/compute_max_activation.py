from pathlib import Path
from argparse import ArgumentParser

from dictionary_learning.cache import PairedActivationCache
from dictionary_learning import CrossCoder
import torch as th
from tqdm import tqdm

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
    parser.add_argument("--dataset", default="lmsys-chat-1m-gemma-formatted")
    parser.add_argument(
        "--model",
        type=str,
        default="Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
    )
    parser.add_argument("--local", action="store_false", dest="from_hub")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    model = CrossCoder.from_pretrained(
        args.model, from_hub=args.from_hub, device=args.device
    )

    cache = PairedActivationCache(
        args.activation_cache_path
        / "gemma-2-2b"
        / args.dataset
        / "validation"
        / f"layer_{args.layer}_out",
        args.activation_cache_path
        / "gemma-2-2b-it"
        / args.dataset
        / "validation"
        / f"layer_{args.layer}_out",
    )
    max_activations = compute_max_activation(model, cache, args.device, args.batch_size)
    path = Path("results") / f"max_activations_{args.model}_{args.dataset}_{args.layer}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    th.save(max_activations, path)
