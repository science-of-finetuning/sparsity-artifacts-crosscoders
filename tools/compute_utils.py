import torch as th
from torch.nn.functional import cosine_similarity, cross_entropy, kl_div
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

__all__ = [
    "BucketedStats",
    "compute_chunked_cosine_similarity",
    "compute_kl",
    "compute_cross_entropy",
    "compute_entropy",
    "RunningMeanStd",
    "chunked_kl",
    "chunked_max_cosim",
]


class ComputedBucketedStats:
    """
    Contains stats computed from a BucketedStats object and allows for easy computation of global stats.
    """

    def __init__(self, bucketed_stats: pd.DataFrame):
        self.bucketed_stats = bucketed_stats

    def compute_global_stats(self):
        """
        Compute global stats from the bucketed stats.
        Returns a DataFrame with global statistics for each latent:
        - weighted mean (weighted by counts)
        - global min
        - global max
        - total count
        """
        # Get the stats as numpy arrays for vectorized operations
        means = self.bucketed_stats["mean"].unstack()  # latent x bucket
        counts = self.bucketed_stats["nonzero count"].unstack()
        mins = self.bucketed_stats["min"].unstack()
        maxs = self.bucketed_stats["max"].unstack()

        # Compute weighted means vectorized
        # Replace NaN with 0 in means and corresponding counts with 0
        means_masked = np.nan_to_num(means, 0.0)
        counts_masked = np.where(np.isnan(means), 0, counts)
        sum_counts = counts_masked.sum(axis=1)  # sum per latent
        sum_counts[sum_counts == 0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            weighted_mean = (means_masked * counts_masked).sum(axis=1) / sum_counts

        # Global min/max per latent (ignoring NaN)
        global_min = np.nanmin(mins, axis=1)
        global_max = np.nanmax(maxs, axis=1)
        total_count = counts.sum(axis=1)

        # Create DataFrame with results
        global_stats = pd.DataFrame(
            {
                "weighted_mean": weighted_mean,
                "min": global_min,
                "max": global_max,
                "total_count": total_count,
            }
        )

        return global_stats


class BucketedStats:
    """
    Keep track of a given statistic for each activation bucket.
    """

    def __init__(
        self,
        num_latents,
        max_activations,
        device,
        bucket_edges,
        epsilon=1e-6,
    ):
        self.max_activations = max_activations.to(device)
        self.bucket_edges = th.tensor(bucket_edges, device=device)
        self.num_buckets = len(bucket_edges) + 1
        self.num_latents = num_latents
        self.epsilon = epsilon
        # Pre-allocate tensors on GPU
        self._counts = th.zeros(
            (num_latents, self.num_buckets),
            dtype=th.int64,
            device=device,
        )
        self._means = th.zeros(
            (num_latents, self.num_buckets),
            dtype=th.float64,
            device=device,
        )
        self._maxs = th.full(
            (num_latents, self.num_buckets),
            float("-inf"),
            dtype=th.float32,
            device=device,
        )
        self._min = th.full(
            (num_latents, self.num_buckets),
            float("inf"),
            dtype=th.float32,
            device=device,
        )

    @th.no_grad()
    def update(self, activations, values):
        """
        Update the stats for given activations and values.

        Args:
            activations: tensor of shape (batch, latents) containing the activations of the latents
            values: tensor of shape (batch,)
        """
        # Compute buckets on GPU
        rel_activations = activations / self.max_activations
        buckets = th.bucketize(rel_activations, self.bucket_edges)
        buckets[activations < self.epsilon] = -1  # filter out zero activations
        for bucket_idx in range(self.num_buckets):
            bucket_mask = buckets == bucket_idx  # shape: (batch, latents)
            assert bucket_mask.shape == activations.shape
            counts = bucket_mask.sum(dim=0)  # shape: (latents,)
            update_mask = counts > 0
            if not update_mask.any():
                continue
            counts = counts[update_mask]  # shape: (ft_to_update,)
            bucket_values = values[:, update_mask]  # shape: (batch, ft_to_update)
            # select values for the update mask
            bucket_values = th.where(
                bucket_mask[:, update_mask], bucket_values, th.zeros_like(bucket_values)
            )  # shape: (batch, ft_to_update)
            means = (
                bucket_values.sum(dim=0).to(th.float64) / counts
            )  # shape: (ft_to_update,)
            maxs = th.where(
                bucket_mask[:, update_mask].any(dim=0),
                bucket_values.max(dim=0).values,  # shape: (ft_to_update,)
                self._maxs[update_mask, bucket_idx],  # shape: (ft_to_update,)
            )  # If a value is not in the bucket, use the max of the bucket to avoid any updates
            mins = th.where(
                bucket_mask[:, update_mask].any(dim=0),
                bucket_values.min(dim=0).values,  # shape: (ft_to_update,)
                self._min[update_mask, bucket_idx],  # shape: (ft_to_update,)
            )

            curr_counts = self._counts[
                update_mask, bucket_idx
            ]  # shape: (ft_to_update,)
            new_counts = curr_counts + counts  # shape: (ft_to_update,)

            self._means[update_mask, bucket_idx] = (
                curr_counts * self._means[update_mask, bucket_idx] + counts * means
            ) / new_counts  # shape: (ft_to_update,)

            self._counts[update_mask, bucket_idx] = new_counts
            self._maxs[update_mask, bucket_idx] = th.maximum(
                self._maxs[update_mask, bucket_idx], maxs
            )
            self._min[update_mask, bucket_idx] = th.minimum(
                self._min[update_mask, bucket_idx], mins
            )

    def finish(self):
        """
        Convert the collected stats to a pandas DataFrame.
        """
        zeros_mask = self._counts == 0
        self._means[zeros_mask] = th.nan
        self._maxs[zeros_mask] = th.nan
        self._min[zeros_mask] = th.nan
        data = {
            "latent": np.repeat(range(self.num_latents), self.num_buckets),
            "bucket": np.tile(
                range(self.num_buckets),
                self.num_latents,
            ),
            "nonzero count": self._counts.cpu().numpy().flatten(),
            "mean": self._means.cpu().numpy().flatten(),
            "max": self._maxs.cpu().numpy().flatten(),
            "min": self._min.cpu().numpy().flatten(),
        }
        stats = pd.DataFrame(data).set_index(["latent", "bucket"])
        assert self._counts[0, 1] == stats.loc[0, 1]["nonzero count"]
        return stats


@th.no_grad()
def compute_chunked_cosine_similarity(weights1, weights2, chunk_size=4):
    """
    Compute the cosine similarity between all vectors in weights1 and weights2 in chunks of weights1 to avoid OOM.

    Args:
        weights1: tensor of shape (num_vectors1, dim)
        weights2: tensor of shape (num_vectors2, dim)
        chunk_size: the number of vectors in weights1 to process at a time

    Returns:
        cosim_matrix: tensor of shape (num_vectors1, num_vectors2) containing the cosine similarity between all vectors in weights1 and weights2
    """
    if weights1.dim() != 2:
        raise ValueError("weights1 must be a 2D tensor")
    if weights2.dim() != 2:
        raise ValueError("weights2 must be a 2D tensor")
    # Calculate chunk size
    num_chunks = weights1.shape[0] // chunk_size

    # Create list to store chunk matrices
    cosim_matrices = []

    # Process each chunk
    for i in tqdm(range(num_chunks)):
        # th.cuda.empty_cache()
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else weights1.shape[0]
        chunk = weights1[start_idx:end_idx]

        # Compute cosine similarity for this chunk
        # Use modulo to cycle through available GPUs
        gpu_idx = i % th.cuda.device_count()
        device = f"cuda:{gpu_idx}"
        if gpu_idx == 0:
            # sync
            for _id in range(th.cuda.device_count()):
                th.cuda.synchronize(f"cuda:{_id}")
            th.cpu.synchronize()
        cosim_matrix_chunk = cosine_similarity(
            chunk.unsqueeze(1).to(device, non_blocking=True),
            weights2.unsqueeze(0).to(device, non_blocking=True),
            dim=2,
        ).to("cpu", non_blocking=True)
        cosim_matrices.append(cosim_matrix_chunk)

    # Combine all chunks and move to CPU
    cosim_matrix = th.cat(cosim_matrices, dim=0)

    return cosim_matrix


def compute_kl(
    logits, logit_target, mask=None, average_over_tokens=True, allow_non_bool_mask=False
):
    """
    Compute KL divergence between two logit distributions over assistant tokens.

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size) or (num_tokens, vocab_size)
        logit_target: Logits tensor of shape (batch_size, seq_len, vocab_size) or (num_tokens, vocab_size)
        mask: Boolean mask of shape (batch_size, seq_len) indicating which tokens to include in the KL divergence calculation
        average_over_tokens: If True, average over tokens
        allow_non_bool_mask: If True, allow non-boolean masks
    Returns:
        KL divergence per token (summed over vocab dimension, averaged over tokens if average_over_tokens is True)
    """
    if mask is not None:
        if mask.dtype != th.bool and not allow_non_bool_mask:
            raise ValueError(
                "Mask should probably be a boolean tensor. If you want to allow non-boolean masks, set allow_non_bool_mask=True."
            )
        log_probs = th.log_softmax(logits[mask].float(), dim=-1)
        log_probs_target = th.log_softmax(logit_target[mask].float(), dim=-1)
        if log_probs.dim() != 2 or log_probs_target.dim() != 2:
            raise ValueError(
                "Logits should be 2D, there is probably a mistake in the mask"
            )
    else:
        log_probs = th.log_softmax(logits.float(), dim=-1)
        log_probs_target = th.log_softmax(logit_target.float(), dim=-1)
        if log_probs.dim() != 2 or log_probs_target.dim() != 2:
            raise ValueError(
                "Logits should be 2D, flatten your sequence length dimension"
            )
    if average_over_tokens:
        kl = kl_div(log_probs, log_probs_target, log_target=True, reduction="batchmean")
    else:
        kl = kl_div(log_probs, log_probs_target, log_target=True, reduction="none").sum(
            dim=-1
        )
    return kl


def chunked_kl(logits, logit_target, chunk_size=32):
    pass
    num_samples = logits.shape[0]
    assert logits.dim() == 2, "logits should be (num_tokens, vocab_size)"
    assert logit_target.dim() == 2, "logit_target should be (num_tokens, vocab_size)"
    assert (
        logits.shape == logit_target.shape
    ), "logits and logit_target should have the same shape"
    kl_chunks = []
    for i in range(0, num_samples, chunk_size):
        chunk_logits = logits[i : min(i + chunk_size, logits.shape[0])]
        chunk_logit_target = logit_target[i : min(i + chunk_size, logits.shape[0])]
        kl = compute_kl(chunk_logits, chunk_logit_target, average_over_tokens=False)
        kl_chunks.append(kl)
    return th.cat(kl_chunks, dim=0)


def compute_cross_entropy(batch, model, pred_mask):
    """
    Compute the cross entropy loss for the given tokens and model. A mask is provided to indicate which tokens are predicted.

    Args:
        batch: the batch to compute the cross entropy loss for (dict with input_ids and attention_mask)
        model: the model to compute the cross entropy loss for
        pred_mask: the mask to indicate which tokens are predicted

    Returns:
        loss: shape (pred_mask[:, 1:].sum(),) Tensor of cross entropy loss for each predicted token
    """
    with model.trace(batch):
        logits = model.output.logits[:, :-1][
            pred_mask[:, 1:]
        ].save()  # shift by 1 to the left
    target_tokens = batch["input_ids"][:, 1:][
        pred_mask[:, 1:]
    ]  # actual token with mask = 1
    loss = cross_entropy(logits, target_tokens, reduction="none")
    return loss


def compute_entropy(batch, model, pred_mask):
    """
    Compute the entropy of the logits for the given tokens and model. A mask is provided to indicate which tokens are predicted.

    Args:
        batch: the batch to compute the entropy for (dict with input_ids and attention_mask)
        model: the model to compute the entropy for
        pred_mask: the mask to indicate which tokens are predicted

    Returns:
        entropy: shape (pred_mask.sum(),) Tensor of entropy for each predicted token
    """
    with model.trace(batch):
        logits = model.output.logits[pred_mask].save()  # shift by 1 to the left
    log_probs = th.log_softmax(logits, dim=-1)
    probs = th.exp(log_probs)
    entropy = -th.sum(probs * log_probs, dim=-1)
    return entropy


# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
class RunningMeanStd:
    def __init__(self, keep_samples=False):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = None
        self.var = None
        self.count = 0
        self.keep_samples = keep_samples
        self.samples = []

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd()
        if self.mean is not None:
            new_object.mean = self.mean.clone()
            new_object.var = self.var.clone()
            new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: th.Tensor) -> None:
        if self.keep_samples:
            self.samples.append(arr.cpu())
        batch_mean = arr.double().mean(dim=0)
        batch_var = arr.double().var(dim=0)
        batch_count = arr.shape[0]
        if batch_count == 0:
            return
        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: th.Tensor, batch_var: th.Tensor, batch_count: float
    ) -> None:
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + th.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def compute(
        self, return_dict=False
    ) -> tuple[th.Tensor, th.Tensor, float] | dict[str, float]:
        """
        Compute the running mean and variance and also return the count

        Returns:
            mean, var, count
        """
        if return_dict:
            dic = {
                "mean": self.mean.item(),
                "var": self.var.item(),
                "count": self.count,
            }
            if self.keep_samples:
                dic["samples"] = th.cat(self.samples, dim=0)
            return dic
        if self.keep_samples:
            return self.mean, self.var, self.count, th.cat(self.samples, dim=0)
        return self.mean, self.var, self.count


@th.no_grad()
def chunked_max_cosim(weights1, weights2, chunk_size=4):
    """
    Compute the max cosine similarity between all vectors in weights1 and weights2 in chunks of weights1 to avoid OOM.

    Args:
        weights1: tensor of shape (num_vectors1, dim)
        weights2: tensor of shape (num_vectors2, dim)
        chunk_size: the number of vectors in weights1 to process at a time

    Returns:
        cosim_matrix: tensor of shape (num_vectors1) containing the max cosine similarity for each vector in weights1 with all vectors in weights2
    """
    if weights1.dim() != 2:
        raise ValueError("weights1 must be a 2D tensor")
    if weights2.dim() != 2:
        raise ValueError("weights2 must be a 2D tensor")
    # Calculate chunk size
    num_chunks = weights1.shape[0] // chunk_size

    # Create list to store chunk matrices
    cosim_matrices = []

    # Process each chunk
    for i in tqdm(range(num_chunks)):
        # th.cuda.empty_cache()
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else weights1.shape[0]
        chunk = weights1[start_idx:end_idx]

        # Compute cosine similarity for this chunk
        # Use modulo to cycle through available GPUs
        gpu_idx = i % th.cuda.device_count()
        device = f"cuda:{gpu_idx}"
        if gpu_idx == 0:
            # sync
            for _id in range(th.cuda.device_count()):
                th.cuda.synchronize(f"cuda:{_id}")
            th.cpu.synchronize()
        cosim_matrix_chunk = (
            cosine_similarity(
                chunk.unsqueeze(1).to(device, non_blocking=True),
                weights2.unsqueeze(0).to(device, non_blocking=True),
                dim=2,
            )
            .max(dim=1)
            .values.to("cpu", non_blocking=True)
        )
        cosim_matrices.append(cosim_matrix_chunk)

    # Combine all chunks and move to CPU
    cosim_matrix = th.cat(cosim_matrices, dim=0)
    assert cosim_matrix.shape == (weights1.shape[0],)

    return cosim_matrix
