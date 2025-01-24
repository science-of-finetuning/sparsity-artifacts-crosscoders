# %%
import torch as th
from typing import Callable
from dictionary_learning import CrossCoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache
import numpy as np
from loguru import logger

def remove_latents(activation: th.Tensor, latent_activations: th.Tensor, latent_vectors: th.Tensor) -> th.Tensor:
    # activation: N x dim_model
    # latent_vectors: num_latent_vectors x dim_model
    # latent_activations: N x num_latent_vectors
    # return: num_latent_vectors x N x dim_model
    num_latent_vectors = latent_vectors.shape[0]
    dim_model = latent_vectors.shape[1]
    N = activation.shape[0]

    assert latent_activations.shape == (N, num_latent_vectors)
    assert activation.shape == (N, dim_model)
    assert latent_vectors.shape == (num_latent_vectors, dim_model)

    # stack the activations num_latent_vectors times -> (num_latent_vectors, N, dim_model)
    activation_stacked = activation.unsqueeze(0).repeat(num_latent_vectors, 1, 1)
    assert activation_stacked.shape == (num_latent_vectors, N, dim_model)

    # remove the latents from the activation
    latent_vectors_reshaped = latent_vectors.unsqueeze(1).repeat(1, N, 1)
    assert latent_vectors_reshaped.shape == (num_latent_vectors, N, dim_model)
    # scale by latent_activations
    latent_activations_reshaped = latent_activations.T.unsqueeze(-1) # (num_latent_vectors, N, 1)
    assert latent_activations_reshaped.shape == (num_latent_vectors, N, 1)

    # remove the latents
    activation_stacked = activation_stacked - (latent_vectors_reshaped * latent_activations_reshaped)
    assert activation_stacked.shape == (num_latent_vectors, N, dim_model)
    return activation_stacked


# %%
@th.no_grad()
def closed_form_scalars(
    latent_vectors: th.Tensor,
    latent_indices: th.Tensor,
    dataloader: DataLoader,
    crosscoder: CrossCoder,
    activation_postprocessing_fn: Callable[[th.Tensor], th.Tensor],
    device: th.device = th.device("cuda"),
) -> th.Tensor:
    # beta = (latent_vector.T @ (data.T @ latent_vector) / ((latent_vector.norm() ** 2) * (latent_activations.norm() ** 2))
    # data: N x dim_model
    # latent_vector: dim_model
    # latent_activations: N
    # beta = (latent_vector.T @ A) / (C * D)
    # beta = (B) / (C * D)
    # A = (data.T @ latent_activations) # (dim_model)
    # B = (latent_vector.T @ A) # (scalar)
    # C = (latent_activations.norm() ** 2) # (scalar)
    # D = (latent_vector.norm() ** 2) # (scalar)

    # We do this for num_latent_vectors latent vectors, so
    # latent_vectors: num_latent_vectors x dim_model
    # latent_activations: N x num_latent_vectors
    # A = (data.T @ latent_activations) # (dim_model, N) x (N, num_latent_vectors) -> (dim_model, num_latent_vectors)
    # B = diag(latent_vectors.T @ A) # (num_latent_vectors, dim_model) x (dim_model, num_latent_vectors) -> (num_latent_vectors, num_latent_vectors)
    # C = (latent_activations.norm() ** 2) # (num_latent_vectors)
    # D = (latent_vector.norm() ** 2) # (num_latent_vectors)
    # beta = (B) / (C * D) # (num_latent_vectors)

    # batched_data -> one data for each latent vector
    # data: num_latent_vectors x N x dim_model
    # only A is different then
    # A = th.matmul(data.transpose(1, 2), latent_activations.T.unsqueeze(-1))  (num_latent_vectors, dim_model, N) x (N, num_latent_vectors) -> (dim_model, num_latent_vectors)

    assert latent_vectors.ndim == 2  # (D, dim_model)
    latent_vectors = latent_vectors.to(device)

    dim_model = latent_vectors.size(1)
    num_latent_vectors = latent_vectors.size(0)
    dict_size = crosscoder.dict_size

    A = th.zeros(
        (dim_model, num_latent_vectors), device=device
    )  # data.T @ latent_activations
    C = th.zeros(num_latent_vectors, device=device)  # latent_activations.norm() ** 2
    D = th.zeros(num_latent_vectors, device=device)  # latent_vectors.norm() ** 2

    count_active = th.zeros(num_latent_vectors, device=device)

    for batch in tqdm(dataloader):
        batch_size_current = batch.shape[0]
        batch = batch.to(device)
        latent_activations = crosscoder.encode(batch)
        assert latent_activations.shape == (batch_size_current, dict_size)
        Y_batch = activation_postprocessing_fn(
            batch, crosscoder=crosscoder, latent_activations=latent_activations
        )
        if len(Y_batch.shape) == 3:
            assert Y_batch.shape == (num_latent_vectors, batch_size_current, dim_model)
        else:
            assert Y_batch.shape == (batch_size_current, dim_model)
        latent_activations = latent_activations[:, latent_indices]
        assert latent_activations.shape == (batch_size_current, num_latent_vectors)

        non_zero_mask = (latent_activations != 0).sum(dim=0)
        assert non_zero_mask.shape == (num_latent_vectors,)
        count_active += non_zero_mask
        non_zero_mask = non_zero_mask > 0

        non_zero_elements = non_zero_mask.sum()

        if len(Y_batch.shape) == 3:
            # one data vector per latent vector
            # use batched logic
            A_update = (
                th.matmul(
                    Y_batch[non_zero_mask].transpose(1, 2),
                    latent_activations[:, non_zero_mask].T.unsqueeze(-1),
                )
                .squeeze(-1)
                .T
            )  # (num_latent_vectors, dim_model, N) x (N, num_latent_vectors) -> (num_latent_vectors, dim_model)
        else:
            A_update = Y_batch.T @ latent_activations[:, non_zero_mask]
        assert A_update.shape == (dim_model, non_zero_elements)
        A[:, non_zero_mask] += A_update

        C_update = (latent_activations[:, non_zero_mask] ** 2).sum(dim=0)
        assert C_update.shape == (non_zero_elements,)
        C[non_zero_mask] += C_update

    D = th.sum(latent_vectors**2, dim=1)
    assert D.shape == (num_latent_vectors,)

    B = th.sum(A.T * latent_vectors, dim=1)  # diag(latent_vector.T @ A)
    assert B.shape == (num_latent_vectors,)

    betas = B / (C * D)
    assert betas.shape == (num_latent_vectors,)

    return betas, count_active


def test_closed_form_scalars(
    dim_model,
    num_latent_vectors,
    N,
    separate_data_per_latent_vector=False,
    batch_size=25,
    dtype=th.float64,
    device=th.device("cuda"),
    verbose=False,
):
    latent_vectors = []
    for i in range(num_latent_vectors):
        v = th.randn(dim_model, dtype=dtype, device=device)
        v = v / v.norm()  # Normalize to unit vector

        if not separate_data_per_latent_vector:
            # Make it orthogonal to the previous vectors
            for j in range(i):
                v = v - (v @ latent_vectors[j]) * latent_vectors[j]

            # Verify orthogonality
            for j in range(i):
                assert (
                    th.abs(v @ latent_vectors[j]) < 1e-6
                ), "Vectors are not orthogonal"
        v = v / v.norm()
        latent_vectors.append(v)

    latent_vectors = th.stack(latent_vectors, dim=0)  # (num_latent_vectors, dim_model)

    if separate_data_per_latent_vector:
        # Sample N random vectors
        v_train = th.randn(num_latent_vectors, N, dim_model, dtype=dtype, device=device)

        for i in range(num_latent_vectors):
            P = th.outer(latent_vectors[i], latent_vectors[i])
            v_train[i] = v_train[i] - (P @ v_train[i].T).T

        # Verify orthogonality
        for i in range(num_latent_vectors):
            assert th.all(
                th.abs(v_train[i] @ latent_vectors[i]) < 1e-6
            ), "Vectors are not orthogonal"
    else:
        # Sample N random vectors
        v_train = th.randn(N, dim_model, dtype=dtype, device=device)

        # Project out the two vectors
        for i in range(num_latent_vectors):
            P = th.outer(latent_vectors[i], latent_vectors[i])
            v_train = v_train - (P @ v_train.T).T

        # Verify orthogonality
        for i in range(num_latent_vectors):
            assert th.all(
                th.abs(v_train @ latent_vectors[i]) < 1e-6
            ), "Vectors are not orthogonal"

    latent_activations = th.randn(N, num_latent_vectors, dtype=dtype, device=device)

    # randomly scale the latent vectors
    for i in range(num_latent_vectors):
        latent_vectors[i] = latent_vectors[i] * th.randn(1, dtype=dtype, device=device)

    beta_ground_truth = (
        th.randn(num_latent_vectors, dtype=dtype, device=device) * 5
    )  # scale by 5

    scaled_activations = (
        latent_activations * beta_ground_truth
    )  # (N, num_latent_vectors)

    if separate_data_per_latent_vector:
        v_train_combined = v_train
        for i in range(num_latent_vectors):
            scaled_target_vectors = th.outer(
                scaled_activations[:, i], latent_vectors[i]
            )  # (N) * (dim_model) -> (N, dim_model)
            v_train_combined[i] = v_train_combined[i] + scaled_target_vectors
    else:
        scaled_target_vectors = (
            scaled_activations @ latent_vectors
        )  # (N, num_latent_vectors) @ (num_latent_vectors, dim_model) -> (N, dim_model)
        v_train_combined = v_train + scaled_target_vectors

    class ToyCrosscoder(CrossCoder):
        def __init__(self, ground_truth_latent_activations: th.Tensor):
            self.ground_truth_latent_activations = ground_truth_latent_activations
            self.batch_index = 0
            self.dict_size = num_latent_vectors
            pass

        def encode(self, x: th.Tensor) -> th.Tensor:
            out = self.ground_truth_latent_activations[self.batch_index, :]
            self.batch_index += 1
            return out

    assert N % batch_size == 0, "N must be divisible by batch_size"
    if separate_data_per_latent_vector:
        v_train_combined_batched = v_train.reshape(
            num_latent_vectors, N // batch_size, batch_size, dim_model
        )
        v_train_combined_batched = v_train_combined_batched.permute(
            1, 2, 0, 3
        )  # (num_batches, batch_size, num_latent_vectors, dim_model)
        processor = lambda x, **kwargs: x.permute(
            1, 0, 2
        )  # x argument is (batch_size, num_latent_vectors, dim_model)
    else:
        v_train_combined_batched = v_train_combined.reshape(
            N // batch_size, batch_size, dim_model
        )
        processor = lambda x, **kwargs: x

    latent_activations_batched = latent_activations.reshape(
        N // batch_size, batch_size, num_latent_vectors
    )

    crosscoder = ToyCrosscoder(latent_activations_batched.to(device))

    beta, count_active = closed_form_scalars(
        latent_vectors.to(device),
        th.arange(num_latent_vectors).to(device),
        v_train_combined_batched.to(device),
        crosscoder,
        processor,
        device=device,
    )

    beta = beta.cpu()
    beta_ground_truth = beta_ground_truth.cpu()
    assert th.allclose(beta, beta_ground_truth)
    if verbose:
        print("Test passed!")
        print("Max error: ", th.max(th.abs(beta - beta_ground_truth)))


# %%
if __name__ == "__main__":
    # %%
    verbose = True
    test_closed_form_scalars(
        dim_model=10, num_latent_vectors=2, N=100, batch_size=25, verbose=verbose
    )
    test_closed_form_scalars(
        dim_model=100, num_latent_vectors=2, N=1000, batch_size=50, verbose=verbose
    )
    test_closed_form_scalars(
        dim_model=100, num_latent_vectors=10, N=1000, batch_size=50, verbose=verbose
    )
    test_closed_form_scalars(
        dim_model=1000, num_latent_vectors=128, N=10000, batch_size=100, verbose=verbose
    )
    test_closed_form_scalars(
        dim_model=1000, num_latent_vectors=128, N=10000, batch_size=200, verbose=verbose
    )
    test_closed_form_scalars(
        dim_model=10,
        num_latent_vectors=2,
        N=100,
        batch_size=25,
        separate_data_per_latent_vector=True,
        verbose=verbose,
    )
    test_closed_form_scalars(
        dim_model=100,
        num_latent_vectors=2,
        N=1000,
        batch_size=50,
        separate_data_per_latent_vector=True,
        verbose=verbose,
    )
    test_closed_form_scalars(
        dim_model=100,
        num_latent_vectors=10,
        N=1000,
        batch_size=50,
        separate_data_per_latent_vector=True,
        verbose=verbose,
    )
    test_closed_form_scalars(
        dim_model=1000,
        num_latent_vectors=128,
        N=10000,
        batch_size=100,
        separate_data_per_latent_vector=True,
        verbose=verbose,
    )
    test_closed_form_scalars(
        dim_model=1000,
        num_latent_vectors=128,
        N=10000,
        batch_size=200,
        separate_data_per_latent_vector=True,
        verbose=verbose,
    )
    # %%
    cc = CrossCoder.from_pretrained(
        "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04", from_hub=True
    )
    cc = cc.to("cuda")

    # %%
    activation_store_dir = Path("/workspace/data/activations/")
    dataset_split = "train"
    batch_size = 256
    n = 20_000_000

    BASE_MODEL = "gemma-2-2b"
    INSTRUCT_MODEL = "gemma-2-2b-it"
    activation_store_dir = Path(activation_store_dir)

    base_model_dir = activation_store_dir / BASE_MODEL
    instruct_model_dir = activation_store_dir / INSTRUCT_MODEL

    base_model_fineweb = base_model_dir / "fineweb-1m-sample" / dataset_split
    base_model_lmsys_chat = (
        base_model_dir / "lmsys-chat-1m-gemma-formatted" / dataset_split
    )
    instruct_model_fineweb = instruct_model_dir / "fineweb-1m-sample" / dataset_split
    instruct_model_lmsys_chat = (
        instruct_model_dir / "lmsys-chat-1m-gemma-formatted" / dataset_split
    )

    submodule_name = f"layer_13_out"

    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_chat_cache = PairedActivationCache(
        base_model_lmsys_chat / submodule_name,
        instruct_model_lmsys_chat / submodule_name,
    )

    dataset = th.utils.data.ConcatDataset([fineweb_cache, lmsys_chat_cache])

    # %%
    it_decoder = cc.decoder.weight[1, :, :].clone()
    base_decoder = cc.decoder.weight[0, :, :].clone()

    n_per_dataset = n // 2
    test_idx = th.cat(
        [th.arange(n_per_dataset), th.arange(n_per_dataset) + len(fineweb_cache)]
    )
    # Get the text indices from the validation dataset
    dataset = th.utils.data.Subset(dataset, test_idx)
    print("Number of activations: ", len(dataset))

    dataloader = th.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10
    )

    # %%
    chat_only_indices = th.load(
        activation_store_dir / ".." / "only_it_decoder_feature_indices.pt"
    )

    # %%
    def load_base_activation(batch, **kwargs):
        return batch[:, 0, :]


    # %%
    def load_base_error(
        batch, crosscoder: CrossCoder, latent_activations: th.Tensor, **kwargs
    ):
        reconstruction = crosscoder.decode(latent_activations)
        return batch[:, 0, :] - remove_latents(reconstruction[:, 0, :], latent_activations, base_decoder[chat_only_indices])


    def load_base_reconstruction(
        batch, crosscoder: CrossCoder, latent_activations: th.Tensor, **kwargs
    ):
        reconstruction = crosscoder.decode(latent_activations)
        return remove_latents(reconstruction[:, 0, :], latent_activations[:, chat_only_indices], base_decoder[chat_only_indices])

    def load_chat_reconstruction(
        batch, crosscoder: CrossCoder, latent_activations: th.Tensor, **kwargs
    ):
        reconstruction = crosscoder.decode(latent_activations)
        return remove_latents(reconstruction[:, 0, :], latent_activations[:, chat_only_indices], it_decoder[chat_only_indices])

    def load_chat_error(
        batch, crosscoder: CrossCoder, latent_activations: th.Tensor, **kwargs
    ):
        reconstruction = crosscoder.decode(latent_activations)
        return batch[:, 0, :] - remove_latents(reconstruction[:, 0, :], latent_activations[:, chat_only_indices], it_decoder[chat_only_indices])

    results_dir = Path("/workspace/data/results/") / "closed_form_scalars"
    results_dir.mkdir(parents=True, exist_ok=True)
    # %%
    betas_base_reconstruction, count_active_base_reconstruction = closed_form_scalars(
        it_decoder[chat_only_indices],
        chat_only_indices,
        dataloader,
        cc,
        load_base_reconstruction,
        device=th.device("cuda"),
    )
    th.save(betas_base_reconstruction, results_dir / "betas_base_reconstruction.pt")
    th.save(count_active_base_reconstruction, results_dir / "count_active_base_reconstruction.pt")
    # %%
    betas_base_error, count_active_base_error = closed_form_scalars(
        it_decoder[chat_only_indices],
        chat_only_indices,
        dataloader,
        cc,
        load_base_error,
        device=th.device("cuda"),
    )
    th.save(betas_base_error, results_dir / "betas_base_error.pt")
    th.save(count_active_base_error, results_dir / "count_active_base_error.pt")
    # %%
    betas_it_reconstruction, count_active_it_reconstruction = closed_form_scalars(
        it_decoder[chat_only_indices],
        chat_only_indices,
        dataloader,
        cc,
        load_chat_reconstruction,
        device=th.device("cuda"),
    )
    th.save(betas_it_reconstruction, results_dir / "betas_it_reconstruction.pt")
    th.save(count_active_it_reconstruction, results_dir / "count_active_it_reconstruction.pt")
    # %%
    betas_it_error, count_active_it_error = closed_form_scalars(
        it_decoder[chat_only_indices],
        chat_only_indices,
        dataloader,
        cc,
        load_chat_error,
        device=th.device("cuda"),
    )
    th.save(betas_it_error, results_dir / "betas_it_error.pt")
    th.save(count_active_it_error, results_dir / "count_active_it_error.pt")

    # %%


# # %%
# print("Number of nans: ", (betas.isnan()).sum())
# # %%
# non_nan_betas = betas[~betas.isnan()]
# # %%
# # Plot the betas
# import matplotlib.pyplot as plt

# plt.hist(non_nan_betas.cpu().numpy(), bins=100)
# plt.yscale("log")
# plt.show()
# # Plot zoomed histogram around 1
# plt.figure(figsize=(10, 6))
# plt.hist(non_nan_betas[non_nan_betas < 10].cpu().numpy(), bins=100)
# plt.title("Zoomed histogram (betas < 10)")
# plt.xlabel("Beta values")
# plt.ylabel("Count")
# plt.yscale("log")
# plt.tight_layout()
# plt.show()
# # %%
# # plot the count_active
# # Plot full histogram
# plt.figure(figsize=(12, 4))
# plt.subplot(121)

# plt.hist(count_active.cpu().numpy(), bins=100)
# plt.title("Full histogram")


# # Plot zoomed histogram
# plt.subplot(122)
# plt.hist(count_active.cpu().numpy(), bins=100, range=(0, 100))
# plt.title("Zoomed histogram (counts < 100)")

# plt.tight_layout()
# plt.show()
# # %%
# max_beta = np.nanargmax(betas.cpu().numpy())
# print("Max beta index: ", chat_only_indices[max_beta])
# print("Max beta value: ", betas[max_beta])
# print("Max active count: ", count_active[max_beta])
# # %%
# # Create 2D histogram of betas vs active counts for betas > 5
# plt.figure(figsize=(10, 6))

# mask = betas > 1000
# x = betas[mask].cpu().numpy()
# y = count_active[mask].cpu().numpy()

# plt.hist2d(x, y, bins=50, cmap="viridis")
# plt.colorbar(label="Count")

# plt.xlabel("Beta values")
# plt.ylabel("Active count")
# plt.title("2D histogram of betas > 5 vs their active counts")

# plt.tight_layout()
# plt.show()
# # %%


# # %%
