import json
import warnings
from pathlib import Path
from typing import Any, Union
from collections import defaultdict
from tempfile import TemporaryDirectory

import torch as th
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity, cross_entropy, kl_div
import pandas as pd
from huggingface_hub import hf_hub_download, hf_api
import networkx as nx

from torch import Tensor
from torchmetrics.aggregation import BaseAggregator
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from dictionary_learning import CrossCoder
from dictionary_learning.cache import PairedActivationCache

template_path = (
    Path(__file__).parent.parent / "templates" / "gemma_chat_template_ctrl_tokens.jinja"
)
chat_template_path = (
    Path(__file__).parent.parent / "templates" / "gemma_chat_template.jinja"
)
with open(template_path, "r") as f:
    ctrl_template = f.read()
with open(chat_template_path, "r") as f:
    chat_template = f.read()


def load_activation_dataset(
    activation_store_dir: Path,
    base_model: str = "gemma-2-2b",
    instruct_model: str = "gemma-2-2b-it",
    layer: int = 13,
    num_samples_per_dataset: int = None,
    split = "validation"
):
    # Load validation dataset
    activation_store_dir = Path(activation_store_dir)
    base_model_dir = activation_store_dir / base_model
    instruct_model_dir = activation_store_dir / instruct_model

    submodule_name = f"layer_{layer}_out"

    # Load validation caches
    base_model_fineweb = base_model_dir / "fineweb-1m-sample" / split
    base_model_lmsys = base_model_dir / "lmsys-chat-1m-gemma-formatted" / split
    instruct_model_fineweb = instruct_model_dir / "fineweb-1m-sample" / split
    instruct_model_lmsys = (
        instruct_model_dir / "lmsys-chat-1m-gemma-formatted" / split
    )

    fineweb_cache = PairedActivationCache(
        base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name
    )
    lmsys_cache = PairedActivationCache(
        base_model_lmsys / submodule_name, instruct_model_lmsys / submodule_name
    )

    return fineweb_cache, lmsys_cache


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


def compute_chunked_cosine_similarity(weights1, weights2, chunk_size=4):
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
        warnings.warn("chat_template is already set in tokenizer_kwargs, ignoring it")
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
        """
        Compute the running mean and variance and also return the count

        Returns:
            mean, var, count
        """
        return self.mean, self.var, self.count


dfs = defaultdict(lambda: None)
df_hf_repo = {
    "l13_crosscoder": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
    "connor": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-ckissane",
}


def load_latent_df(crosscoder=None):
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    df_path = hf_hub_download(
        repo_id=df_hf_repo[crosscoder],
        filename="feature_df.csv",
        repo_type="dataset",
    )
    return pd.read_csv(df_path, index_col=0)


def push_latent_df(
    df,
    crosscoder=None,
    force=False,
    allow_remove_columns=None,
    commit_message=None,
    confirm=True,
):
    """
    Push a new feature_df.csv to the hub.

    Args:
        df: the new df to push
        crosscoder: the crosscoder to push the df for
        force: if True, push the df even if there are missing columns
        allow_remove_columns: if not None, a list of columns to allow to be removed
        commit_message: the commit message to use for the push
    """
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    if not force or confirm:
        original_df = load_latent_df(crosscoder)
        original_columns = set(original_df.columns)
        new_columns = set(df.columns)
        allow_remove_columns = (
            set(allow_remove_columns) if allow_remove_columns is not None else set()
        )
        missing_columns = original_columns - new_columns - allow_remove_columns
        added_columns = new_columns - original_columns
        shared_columns = original_columns & new_columns
        if len(missing_columns) > 0:
            if force:
                warnings.warn(f"Missing columns in uploaded df: {missing_columns}")
            else:
                raise ValueError(
                    f"Missing columns in uploaded df: {missing_columns}\n"
                    "If you want to upload the df anyway, set allow_remove_columns=your_removed_columns"
                    " and force=True"
                )

        if len(added_columns) > 0 and not force:
            print(f"Added columns in uploaded df: {added_columns}")

        for column in shared_columns:
            if original_df[column].dtype != df[column].dtype:
                warnings.warn(
                    f"Column {column} has different dtype in original and new df"
                )
            # diff the columns
            if not (original_df[column].equals(df[column])):
                print(f"Column {column} has different values in original and new df")
    if confirm:
        print(f"Commit message: {commit_message}")
        r = input("Would you like to push the df to the hub? y/(n)")
        if r != "y":
            raise ValueError("User cancelled")
    with TemporaryDirectory() as tmpdir:
        df.to_csv(Path(tmpdir) / "feature_df.csv")
        hf_api.upload_file(
            repo_id=df_hf_repo[crosscoder],
            path_or_fileobj=Path(tmpdir) / "feature_df.csv",
            path_in_repo="feature_df.csv",
            repo_type="dataset",
            commit_message=commit_message,
        )


def _feature_df(crosscoder=None):
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    global dfs
    if dfs[crosscoder] is None:
        dfs[crosscoder] = load_latent_df(crosscoder)
    return dfs[crosscoder]


def base_only_latent_indices():
    df = _feature_df()
    # filter for tag = Base only
    return th.tensor(df[df["tag"] == "Base only"].index.tolist())


def it_only_latent_indices():
    df = _feature_df()
    # filter for tag = IT only
    return th.tensor(df[df["tag"] == "IT only"].index.tolist())


def shared_latent_indices():
    df = _feature_df()
    # filter for tag = Shared
    return th.tensor(df[df["tag"] == "Shared"].index.tolist())


class CCLatent:
    def __init__(self, id_: int, crosscoder=None):
        self.id = id_
        self.row = _feature_df(crosscoder).loc[id_]
        self.stats = self.row.to_dict()
        self.dead = False
        for k, v in self.stats.items():
            setattr(self, k.replace(" ", "_").replace("%", "pct"), v)
        if crosscoder is None:
            crosscoder = "l13_crosscoder"
        self.crosscoder = crosscoder

    def is_chat_only(self):
        return self.tag == "Chat only" or self.tag == "IT only"

    def is_base_only(self):
        return self.tag == "Base only"

    def is_shared(self):
        return self.tag == "Shared"

    def is_other(self):
        return self.tag == "Other"

    def __str__(self):
        return self.row.__str__()

    def __repr__(self) -> str:
        return self.row.__repr__()

    def base_decoder_vector(self):
        return _crosscoder(self.crosscoder).decoder.weight[0][self.id]

    def chat_decoder_vector(self):
        return _crosscoder(self.crosscoder).decoder.weight[1][self.id]

    def auto_decoder_vector(self):
        if self.is_chat_only():
            return self.chat_decoder_vector()
        elif self.is_base_only():
            return self.base_decoder_vector()
        else:
            raise ValueError(f"Cannot get auto decoder vector for {self.tag}")


def apply_connor_template(conv):
    if isinstance(conv[0], list):
        return [apply_connor_template(c) for c in conv]
    return "\n".join(
        [
            ("Assistant: " if msg["role"] == "assistant" else "User: ") + msg["content"]
            for msg in conv
        ]
    )


@th.no_grad()
def load_connor_crosscoder():
    path = "blocks.14.hook_resid_pre"
    repo_id = "ckkissane/crosscoder-gemma-2-2b-model-diff"
    # Download config and weights
    config_path = hf_hub_download(repo_id=repo_id, filename=f"{path}/cfg.json")
    weights_path = hf_hub_download(repo_id=repo_id, filename=f"{path}/cc_weights.pt")

    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Load weights
    state_dict = th.load(weights_path, map_location=cfg["device"], weights_only=True)

    crosscoder = CrossCoder(
        activation_dim=cfg["d_in"],
        dict_size=cfg["dict_size"],
        num_layers=2,
    )

    crosscoder.encoder.weight[:] = state_dict["W_enc"]
    crosscoder.encoder.bias[:] = state_dict["b_enc"]
    crosscoder.decoder.weight[:] = state_dict["W_dec"].permute(1, 0, 2)
    crosscoder.decoder.bias[:] = state_dict["b_dec"]
    return crosscoder


def load_crosscoder(crosscoder=None):
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    if crosscoder == "l13_crosscoder":
        return CrossCoder.from_pretrained(
            "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04", from_hub=True
        )
    elif crosscoder == "connor":
        return load_connor_crosscoder()
    else:
        raise ValueError(f"Unknown crosscoder: {crosscoder}")


crosscoders = defaultdict(lambda: None)


def _crosscoder(crosscoder=None):
    global crosscoders
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    if crosscoders[crosscoder] is None:
        crosscoders[crosscoder] = load_crosscoder(crosscoder)
    return crosscoders[crosscoder]


"""
=================================
=                               =
=         Plotting utils        =
=                               =
=================================
"""
from networkx.drawing.nx_pylab import apply_alpha


def draw_networkx_nodes(
    G,
    pos,
    nodelist=None,
    node_size=300,
    node_color="#1f78b4",
    node_shape="o",
    alpha=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ax=None,
    linewidths=None,
    edgecolors=None,
    label=None,
    margins=None,
    hide_ticks=True,
):
    """Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    nodelist : list (default list(G))
        Draw only specified nodes

    node_size : scalar or array (default=300)
        Size of nodes.  If an array it must be the same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or array of floats (default=None)
        The node transparency.  This can be a single alpha value,
        in which case it will be applied to all the nodes of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap (default=None)
        Colormap for mapping intensities of nodes

    vmin,vmax : floats or None (default=None)
        Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence] (default=1.0)
        Line width of symbol border

    edgecolors : [None | scalar | sequence] (default = node_color)
        Colors of node borders. Can be a single color or a sequence of colors with the
        same length as nodelist. Color can be string or rgb (or rgba) tuple of floats
        from 0-1. If numeric values are specified they will be mapped to colors
        using the cmap and vmin,vmax parameters. See `~matplotlib.pyplot.scatter` for more details.

    label : [None | string]
        Label for legend

    margins : float or 2-tuple, optional
        Sets the padding for axis autoscaling. Increase margin to prevent
        clipping for nodes that are near the edges of an image. Values should
        be in the range ``[0, 1]``. See :meth:`matplotlib.axes.Axes.margins`
        for details. The default is `None`, which uses the Matplotlib default.

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nodes = nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_edges
    draw_networkx_labels
    draw_networkx_edge_labels
    """
    from collections.abc import Iterable

    import matplotlib as mpl
    import matplotlib.collections  # call as mpl.collections
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)

    if len(nodelist) == 0:  # empty nodelist, no drawing
        return mpl.collections.PathCollection(None)

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as err:
        raise nx.NetworkXError(f"Node {err} has no position.") from err

    if isinstance(alpha, Iterable):
        node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
        alpha = None

    # Convert node_shape to array if it's not already
    if not isinstance(node_shape, (np.ndarray, list)):
        node_shape = [node_shape] * len(nodelist)
    node_shape = np.asarray(node_shape)

    # Convert node_color to array if it's not already
    if not isinstance(node_color, (np.ndarray, list)):
        node_color = [node_color] * len(nodelist)
    node_color = np.asarray(node_color)

    # Create collections for each unique shape-color combination
    collections = []

    for shape in np.unique(node_shape):
        shape_mask = node_shape == shape
        shape_xy = xy[shape_mask]
        shape_colors = node_color[shape_mask]

        node_collection = ax.scatter(
            shape_xy[:, 0],
            shape_xy[:, 1],
            s=node_size,
            c=shape_colors,
            marker=shape,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            linewidths=linewidths,
            edgecolors=edgecolors,
            label=label,
        )
        collections.append(node_collection)

    if hide_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    if margins is not None:
        if isinstance(margins, Iterable):
            ax.margins(*margins)
        else:
            ax.margins(margins)

    # Set zorder for all collections
    for collection in collections:
        collection.set_zorder(2)

    # Return the last collection for compatibility
    return collections[-1] if collections else None


def plot_component_sizes(G, title=None, save_path=None):
    # bar plot of the size of the connected components
    component_sizes = [len(c) for c in nx.connected_components(G)]
    print(f"found {len(component_sizes)} connected components")
    # Count frequency of each size
    from collections import Counter

    size_counts = Counter(component_sizes)
    # Convert to lists for plotting and sort
    sizes = sorted(list(size_counts.keys()))
    counts = [size_counts[size] for size in sizes]
    fig = px.bar(
        x=sizes,
        y=counts,
        title="Distribution of connected component sizes" if title is None else title,
        labels={"x": "Component size", "y": "Number of components"},
    )
    # Only show x-axis ticks for values that exist in the data
    fig.update_xaxes(tickmode="array", tickvals=sizes, type="category")
    if save_path is not None:
        fig.write_image(save_path / "component_sizes.png")
        fig.write_html(save_path / "component_sizes.html")
    fig.show()
    # bar plot where x is the component size and y is the number of nodes in those components
    y = [size_counts[size] * size for size in sizes]
    fig = px.bar(
        x=sizes,
        y=y,
        title=(
            "Distribution of latent in different component sizes" + "\n"
            if title is None
            else title
        ),
        labels={"x": "Component size", "y": "Number of latents"},
    )
    # Only show x-axis ticks for values that exist in the data
    fig.update_xaxes(tickmode="array", tickvals=sizes, type="category")
    if save_path is not None:
        fig.write_image(save_path / "component_sizes.png")
        fig.write_html(save_path / "component_sizes.html")
    fig.show()


def draw_graph(G, crosscoder, title="", file=None):
    df = _feature_df(crosscoder)
    plt.figure(figsize=(15, 5))
    pos = nx.spring_layout(G, k=0.035)  # reduced k from default

    # Create node color list and prepare for legend
    node_colors = []
    node_shapes = []
    it_only_nodes = []
    base_only_nodes = []
    shared_nodes = []
    unknown_nodes = []
    for node in G.nodes():

        latent = CCLatent(int(node[1:]), crosscoder)
        if latent.is_chat_only():
            node_colors.append("red")
            it_only_nodes.append(node)
        elif latent.is_base_only():
            node_colors.append("blue")
            base_only_nodes.append(node)
        elif latent.is_shared():
            node_colors.append("green")
            shared_nodes.append(node)
        else:
            node_colors.append("gray")
            unknown_nodes.append(node)
        if node[0] == "i":
            if latent.is_base_only():
                raise Exception(f"Base only node: {node}")
            node_shapes.append("*")
        else:
            node_shapes.append("o")
        if latent.dead:
            raise Exception(f"dead node: {node}")
    # Draw nodes with colors
    draw_networkx_nodes(
        G, pos, node_size=15, node_color=np.array(node_colors), node_shape=node_shapes
    )

    # Draw edges with width proportional to weight
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[w * 1 for w in edge_weights], alpha=0.3)

    # Add legend with larger text
    plt.scatter([], [], c="red", label="Chat only", s=15)
    plt.scatter([], [], c="blue", label="Base only", s=15)
    plt.scatter([], [], c="green", label="Shared", s=15)
    plt.scatter([], [], c="gray", label="Other", s=15)
    plt.scatter([], [], c="black", marker="*", label="Chat latents", s=15)
    plt.scatter([], [], c="black", marker="o", label="Base latents", s=15)
    plt.legend(fontsize=16)  # 1.6x larger legend text

    plt.title(title, fontsize=14)  # 1.6x larger title text
    plt.axis("off")
    if file is not None:
        plt.savefig(file, bbox_inches="tight", dpi=300)
    plt.show()


def draw_interactive_graph(G, crosscoder, title=""):
    pos = nx.spring_layout(G, k=0.035)  # reduced k from default

    # Create node color list and prepare for legend
    it_only_nodes = []
    base_only_nodes = []
    shared_nodes = []
    unknown_nodes = []

    # Create lists for node traces
    node_x = []
    node_y = []
    node_colors = []
    node_symbols = []
    node_texts = []  # Added for hover text
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_texts.append(node)  # Add node name for hover

        latent = CCLatent(int(node[1:]), crosscoder)
        if latent.is_chat_only():
            node_colors.append("red")
            it_only_nodes.append(node)
        elif latent.is_base_only():
            node_colors.append("blue")
            base_only_nodes.append(node)
        elif latent.is_shared():
            node_colors.append("green")
            shared_nodes.append(node)
        else:
            node_colors.append("gray")
            unknown_nodes.append(node)

        if node[0] == "i":
            node_symbols.append("star")
        else:
            node_symbols.append("circle")

        if latent.dead:
            raise Exception(f"dead node: {node}")

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G[edge[0]][edge[1]]["weight"])

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="black"),
        hoverinfo="none",
        mode="lines",
        opacity=0.6,
        showlegend=False,
    )

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=node_texts,  # Add hover text
        hoverinfo="text",  # Show the text on hover
        marker=dict(size=10, color=node_colors, symbol=node_symbols),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=[node_trace, edge_trace])

    # Update layout
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Add legend using shapes
    legend_items = [
        ("Chat only", "red", "circle"),
        ("Base only", "blue", "circle"),
        ("Shared", "green", "circle"),
        ("Other", "gray", "circle"),
        ("Chat latents", "black", "star"),
        ("Base latents", "black", "circle"),
    ]

    for i, (name, color, symbol) in enumerate(legend_items):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=dict(size=10, color=color, symbol=symbol),
                showlegend=True,
            )
        )

    fig.show()
    return fig
