import torch as th
from tqdm.auto import tqdm
from torch.nn.functional import cosine_similarity
from torch.nn.functional import kl_div
import warnings
from pathlib import Path

from typing import Any, Union

from torch import Tensor
from torchmetrics.aggregation import BaseAggregator


template_path = (
    Path(__file__).parent / "templates" / "gemma_chat_template_ctrl_tokens.jinja"
)
chat_template_path = Path(__file__).parent / "templates" / "gemma_chat_template.jinja"
with open(template_path, "r") as f:
    ctrl_template = f.read()
with open(chat_template_path, "r") as f:
    chat_template = f.read()


class Mean1DMetric(BaseAggregator):

    mean_value: Tensor

    def __init__(
        self,
        size: int,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            th.zeros((size,), dtype=th.get_default_dtype()),
            nan_strategy,
            state_name="mean_value",
            **kwargs,
        )
        self.add_state(
            "weight",
            default=th.tensor(0, dtype=th.get_default_dtype()),
            dist_reduce_fx="sum",
        )

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
        """
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = th.as_tensor(value, dtype=self.dtype, device=self.device)
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel() == 0:
            return
        self.mean_value += value.sum(dim=tuple(range(value.dim() - 1)))
        self.weight += value[..., 0].numel()

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.mean_value / self.weight


def compute_chunked_cosine_similarity(weights1, weights2, chunk_size):
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
            for id in range(th.cuda.device_count()):
                th.cuda.synchronize(f"cuda:{id}")
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


def tokenize_with_ctrl_mask(
    convs: list[list[dict[str, str]]],
    tokenizer,
    **tokenizer_kwargs,
) -> dict:
    """
    Create a mask that is 1 for chat control tokens and 0 for other tokens
    """
    kwargs = tokenizer_kwargs.copy()
    kwargs.update(
        dict(
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            return_dict=True,
            chat_template=ctrl_template,
        )
    )
    ctrl_mask = th.tensor(
        tokenizer.apply_chat_template(
            convs,
            **kwargs,
        )["assistant_masks"],
        dtype=th.bool,
    )
    if "chat_template" in tokenizer_kwargs:
        warnings.warn(
            "chat_template is already set in tokenizer_kwargs, ignoring it"
        )
    tokenizer_kwargs["chat_template"] = chat_template
    tokenizer_kwargs["return_dict"] = True
    tokenizer_kwargs["return_assistant_tokens_mask"] = True
    tok_dict = tokenizer.apply_chat_template(convs, **tokenizer_kwargs)
    tok_dict["ctrl_mask"] = ctrl_mask
    tok_dict["assistant_masks"] = th.tensor(tok_dict["assistant_masks"], dtype=th.bool)
    return tok_dict


def tokenize_with_ctrl_ids(
    convs: list[list[dict[str, str]]],
    tokenizer,
    **tokenizer_kwargs,
) -> dict:
    """
    Same as tokenize_with_ctrl_mask, but labels the control tokens from 1 to 10 instead all True
    """
    tok_dict = tokenize_with_ctrl_mask(convs, tokenizer, **tokenizer_kwargs)
    mask = tok_dict["ctrl_mask"]
    ids = mask.to(th.int)
    n_ctrl_toks = ids.sum()
    rep_1_10 = th.arange(1, 11, dtype=th.int).repeat(n_ctrl_toks // 10 + 1)[
        :n_ctrl_toks
    ]
    ids[mask] = rep_1_10
    tok_dict["ctrl_ids"] = ids
    return tok_dict


from tiny_dashboard.html_utils import (
    create_token_html,
    create_example_html,
    create_base_html,
)
from tiny_dashboard.utils import sanitize_tokens, sanitize_token


def activation_visualization(
    tokens: list[str],
    activations: th.Tensor,
    tokenizer,
    highlight_idx: int | None = None,
    title: str = "",
) -> str:
    """Create HTML with highlighted tokens based on activation values"""
    html_parts = []
    # all_feature_indicies = list(range(activations.shape[0]))
    # Find highlight feature index in the activation tensor
    if highlight_idx is None:
        if activations.dim() == 2:
            raise ValueError(
                "Activations must be 1D unless a highlight feature is specified"
            )
        highlight_acts = activations
        activations = activations.unsqueeze(0)
        other_features = [0]
    else:
        highlight_acts = activations[highlight_idx]
        other_features = [i for i in range(activations.shape[0]) if i != highlight_idx]
    # Normalize activations for color intensity (only for highlight feature)
    max_highlight = highlight_acts.max()
    norm_acts = highlight_acts / (max_highlight + 1e-6)

    # Create HTML spans with activation values
    sanitized_tokens = sanitize_tokens(tokens, non_breaking_space=False)
    for i, (san_token, token) in enumerate(zip(sanitized_tokens, tokens)):

        color = f"rgba(255, 0, 0, {norm_acts[i].item():.3f})"

        # Create tooltip content only for requested features
        tok_id = tokenizer.convert_tokens_to_ids(token)
        tooltip_token = sanitize_token(
            token, keep_newline=False, non_breaking_space=False
        )
        tooltip_lines = [f"Token {tok_id}: '{tooltip_token}'"]
        for feat in other_features:
            act_value = activations[feat, i].item()
            tooltip_lines.append(f"Feature {feat}: {act_value:.3f}")

        tooltip_content = "\n".join(tooltip_lines)
        html_parts.append(create_token_html(san_token, color, tooltip_content))

    html = "".join(html_parts)
    html = create_example_html(max_highlight.item(), html, static=True)
    return create_base_html(title, html)


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


# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = th.zeros(shape, dtype=th.float64)
        self.var = th.ones(shape, dtype=th.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
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
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: th.Tensor, batch_var: th.Tensor, batch_count: float
    ) -> None:
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

    def compute(self) -> tuple[th.Tensor, th.Tensor, float]:
        return self.mean, self.var, self.count
