import torch as th
import argparse
from pathlib import Path    
from dictionary_learning.cache import PairedActivationCache
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer
from dictionary_learning.training import trainSAE
from torchmetrics import MeanMetric, MinMaxMetric
import os
import einops
SEED = 42

th.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)

th.set_float32_matmul_precision('high')
device = "cuda" if th.cuda.is_available() else "cpu"    
device = "cpu"

def compute_alphas(batch, cc, indices):
    alphas = cc.encode(batch)
    alphas = alphas * cc.decoder.weight.norm(dim=2).sum(dim=0, keepdim=True)
    return alphas[:, indices]

def get_D(X, f, cc, feature_idx):
    FMAX = (f > 0).int().sum(dim=1).max()
    D = th.zeros((X.shape[0], FMAX+1, X.shape[1])).to(X.device)
    for i in range(X.shape[0]):
        f_nonzero = f[i].nonzero().flatten()
        D[i, th.arange(1, min(FMAX, f_nonzero.shape[0])+1), :] = cc.decoder.weight.data[0, f_nonzero[:min(FMAX, f_nonzero.shape[0])]]
        D[i, 0, :] = cc.decoder.weight.data[1, feature_idx]
    return D

def lstsq(dataset, cc, high_threshold_indices, feature_idx):
    print(f"Feature {feature_idx} has {len(high_threshold_indices)} high threshold indices.")
    X = [dataset[i] for i in high_threshold_indices]
    X = th.stack(X, dim=0).to(device) 

    f = cc.encode(X)
    print("num nonzero:",   (f[:, feature_idx] > 0).sum())
    X = X[:, 0] # base activations 
    D = get_D(X, f, cc, feature_idx)
    
    # rearrange D to be (d, f, b)
    D = einops.rearrange(D, "b f d -> b d f")
    # rearrange X to be (d, b)
    X = X - cc.decoder.bias[0].to(device)
    X = einops.rearrange(X, "b d -> b d 1")

    assert not D.isnan().any()
    assert not X.isnan().any()
    assert not D.isinf().any()
    assert not X.isinf().any()

    A, residuals, rank, s = th.linalg.lstsq(D.to(device), X.to(device), rcond=1e-5)
   
    gammas = A[0].cpu()
    non_zero_average = A[1:].flatten()[A[1:].flatten() > 0].mean()
    average = A[1:].flatten().mean()
    return gammas, non_zero_average, average

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-coder-path", type=str, required=True)
    parser.add_argument("--indices-path", type=str, required=True)
    parser.add_argument("--activation-cache-path", type=str, default="activations")
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()

    activation_store_dir = Path(args.activation_cache_path)
    cross_coder_path = Path(args.cross_coder_path)
    assert cross_coder_path.exists(), f"Cross-coder checkpoint not found at {cross_coder_path}."
    indices_path = Path(args.indices_path)
    assert indices_path.exists(), f"Indices file not found at {indices_path}."

    cc = CrossCoder.from_pretrained(cross_coder_path).to(device)
    it_only_indices = th.load(indices_path, weights_only=True)

    total_features = cc.decoder.weight.shape[1]
    # sample 100 features uniformly at random
    sampled_indices = th.randperm(total_features)
    random_indices = []
    i = 0
    while len(random_indices) < 100:
        random_indices.append(sampled_indices[i])
        i += 1

    random_indices = th.tensor(random_indices)
    indices = th.cat([random_indices, it_only_indices])
    base_model_dir = activation_store_dir / "gemma-2-2b"
    instruct_model_dir = activation_store_dir / "gemma-2-2b-it"

    base_model_fineweb = base_model_dir / "fineweb"
    base_model_lmsys_chat = base_model_dir / "lmsys_chat"
    instruct_model_fineweb = instruct_model_dir / "fineweb"
    instruct_model_lmsys_chat = instruct_model_dir / "lmsys_chat"

    submodule_name = f"layer_13_out"

    fineweb_cache = PairedActivationCache(base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name)
    lmsys_chat_cache = PairedActivationCache(base_model_lmsys_chat / submodule_name, instruct_model_lmsys_chat / submodule_name)

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    validation_size = 10**7
    train_dataset, validation_dataset = th.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size])
    print(validation_dataset[0])
    print(f"Validation set size: {len(validation_dataset)} token activations.")

    if not Path("high_threshold_indices.pt").exists():
        chkpoint_steps = 5000
        dataloader = th.utils.data.DataLoader(validation_dataset, batch_size=2**10, shuffle=False, num_workers=32, pin_memory=True)
        high_threshold_indices = [[] for _ in range(len(indices))]
        with th.no_grad():
            idx = 0
            high_threshold = 5
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                alphas = compute_alphas(batch, cc, indices)
                for i in range(len(indices)):
                    alphas_i = alphas[:, i] > high_threshold
                    alphas_i = alphas_i.nonzero().flatten()
                    for j in alphas_i:
                        high_threshold_indices[i].append((idx + j).item())
                idx += len(batch)
                if idx % chkpoint_steps == 0:
                    print(high_threshold_indices)
                    th.save(high_threshold_indices, f"high_threshold_indices_{idx}.pt")
        th.save(high_threshold_indices, "high_threshold_indices.pt")
    else:
        print("Loading high threshold indices from file.")
        high_threshold_indices = th.load("high_threshold_indices.pt")

    random_high_threshold_indices = high_threshold_indices[:100]
    it_only_high_threshold_indices = high_threshold_indices[100:]

    metrics = []
    with th.no_grad():
        print("Computing gammas for random indices.")
        # for high_threshold_indices, feature_idx in zip(random_high_threshold_indices, random_indices):
        #     if len(high_threshold_indices) < 100:
        #         metrics.append((None, None, None))
        #         continue
        #     elif len(high_threshold_indices) > 1000:
        #         high_threshold_indices = high_threshold_indices[:1000]
        #     gammas, non_zero_average, average = lstsq(validation_dataset, cc, high_threshold_indices, feature_idx)

        #     print(gammas.shape)
        #     gamma_mean = gammas.mean()
        #     gamma_std = gammas.std()
        #     gamma_median = gammas.median()
        #     gamma_10 = th.quantile(gammas, 0.1)
        #     gamma_90 = th.quantile(gammas, 0.9)
        #     gammas = gammas[gammas > gamma_10]
        #     gammas = gammas[gammas < gamma_90]
        #     gamma_no_outliers_mean = gammas.mean()  
        #     gamma_no_outliers_std = gammas.std()

        #     res = (gamma_mean, gamma_std, gamma_median, gamma_10, gamma_90, gamma_no_outliers_mean, gamma_no_outliers_std, non_zero_average, average)
        #     th.cuda.empty_cache()
        #     print(res)
        #     metrics.append(res)

        #     metrics_df = pd.DataFrame(metrics, columns=["mean", "std", "median", "10%", "90%", "no_outliers_mean", "no_outliers_std", "non_zero_average", "average"])
        #     metrics_df.to_csv("least_squares_metrics_random.csv")

        metrics = []
        print("Computing gammas for IT-only indices.")
        j = 0
        for high_threshold_indices, feature_idx in tqdm(zip(it_only_high_threshold_indices, it_only_indices)):
            if j < args.start_idx:
                j += 1
                continue
            if len(high_threshold_indices) < 100:
                metrics.append((None, None, None))
                continue
            elif len(high_threshold_indices) > 500:
                high_threshold_indices = high_threshold_indices[:500]
            gammas, non_zero_average, average = lstsq(validation_dataset, cc, high_threshold_indices, feature_idx)

            gamma_mean = gammas.mean()
            gamma_std = gammas.std()
            gamma_median = th.median(gammas)
            gamma_10 = th.quantile(gammas, 0.1)
            gamma_90 = th.quantile(gammas, 0.9)
            gammas = gammas[gammas > gamma_10]
            gammas = gammas[gammas < gamma_90]
            gamma_no_outliers_mean = gammas.mean()  
            gamma_no_outliers_std = gammas.std()

            res = (feature_idx, gamma_mean, gamma_std, gamma_median, gamma_10, gamma_90, gamma_no_outliers_mean, gamma_no_outliers_std, non_zero_average, average)
            metrics.append(res)
            th.cuda.empty_cache()
            print(res)
            metrics_df = pd.DataFrame(metrics, columns=["feature_idx", "mean", "std", "median", "10%", "90%", "no_outliers_mean", "no_outliers_std", "non_zero_average", "average"])
            metrics_df.to_csv(f"least_squares_metrics_it_only_{args.start_idx}.csv")
