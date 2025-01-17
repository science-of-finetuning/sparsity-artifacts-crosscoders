from pathlib import Path

from dictionary_learning.cache import PairedActivationCache
from dictionary_learning import CrossCoder
import torch as th
from tqdm import tqdm
th.set_float32_matmul_precision("high")

def compute_max_activation(crosscoder: CrossCoder, cache: PairedActivationCache, device, batch_size: int = 2048):
    dataloader = th.utils.data.DataLoader(cache, batch_size=batch_size)
    max_activations = th.zeros(cache.num_tokens, device=device)
    for batch in tqdm(dataloader):
        activations = crosscoder.get_activations(batch.to(device))  # (batch_size, features)
        max_activations = th.max(max_activations, activations.max(dim=0).values)
    return max_activations


