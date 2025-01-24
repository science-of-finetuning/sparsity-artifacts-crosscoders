import torch as th
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity, cross_entropy, kl_div
import warnings
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from typing import Any, Union
import networkx as nx

from torch import Tensor
from torchmetrics.aggregation import BaseAggregator
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


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


df = None


def feature_df():
    global df
    if df is None:
        df_path = hf_hub_download(
            repo_id="Butanium/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
            filename="feature_df.csv",
            repo_type="dataset",
        )
        df = pd.read_csv(df_path, index_col=0)
    return df


def base_only_latent_indices():
    df = feature_df()
    # filter for tag = Base only
    return th.tensor(df[df["tag"] == "Base only"].index.tolist())


def it_only_latent_indices():
    df = feature_df()
    # filter for tag = IT only
    return th.tensor(df[df["tag"] == "IT only"].index.tolist())


def shared_latent_indices():
    df = feature_df()
    # filter for tag = Shared
    return th.tensor(df[df["tag"] == "Shared"].index.tolist())


class CCLatent:
    def __init__(self, id_: int):
        self.id = id_
        self.row = feature_df().loc[id_]
        self.stats = self.row.to_dict()
        for k, v in self.stats.items():
            setattr(self, k.replace(" ", "_").replace("%", "pct"), v)

    def is_chat_only(self):
        return self.tag == "Chat only"

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


def apply_connor_template(conv):
    if isinstance(conv[0], list):
        return [apply_connor_template(c) for c in conv]
    return "\n".join(
        [
            ("Assistant: " if msg["role"] == "assistant" else "User: ") + msg["content"]
            for msg in conv
        ]
    )


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


def plot_component_sizes(G, title=None):
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
    fig.show()


def draw_graph(G, title="", file=None):
    plt.figure(figsize=(15, 5))
    pos = nx.spring_layout(G, k=0.035)  # reduced k from default

    # Create node color list and prepare for legend
    node_colors = []
    node_shapes = []
    it_only_nodes = []
    base_only_nodes = []
    shared_nodes = []
    unknown_nodes = []
    df = feature_df()
    for node in G.nodes():

        tag = df.loc[int(node[1:]), "tag"]
        if tag == "IT only":
            node_colors.append("red")
            it_only_nodes.append(node)
        elif tag == "Base only":
            node_colors.append("blue")
            base_only_nodes.append(node)
        elif tag == "Shared":
            node_colors.append("green")
            shared_nodes.append(node)
        else:
            node_colors.append("gray")
            unknown_nodes.append(node)
        if node[0] == "i":
            if tag == "Base only":
                raise Exception(f"Base only node: {node}")
            node_shapes.append("*")
        else:
            node_shapes.append("o")
        if df.loc[int(node[1:]), "dead"]:
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


def draw_interactive_graph(G, title=""):
    df = feature_df()
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

        type = df.loc[int(node[1:]), "tag"]
        if type == "IT only":
            node_colors.append("red")
            it_only_nodes.append(node)
        elif type == "Base only":
            node_colors.append("blue")
            base_only_nodes.append(node)
        elif type == "Shared":
            node_colors.append("green")
            shared_nodes.append(node)
        else:
            node_colors.append("gray")
            unknown_nodes.append(node)

        if node[0] == "i":
            node_symbols.append("star")
        else:
            node_symbols.append("circle")

        if df.loc[int(node[1:]), "dead"]:
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
