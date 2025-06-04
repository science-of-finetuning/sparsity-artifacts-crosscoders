from collections.abc import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_pylab import apply_alpha
import plotly.express as px
import plotly.graph_objects as go

from tools.cc_utils import CCLatent

__all__ = [
    "draw_networkx_nodes",
    "plot_component_sizes",
    "draw_graph",
    "draw_interactive_graph",
]


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

    for name, color, symbol in legend_items:
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


def frequency_plot(df: pd.DataFrame):
    # Get unique tags for grouping
    tags = df["tag"].unique()

    # Create figure
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(111)

    # Colors matching the other plot
    colors = {
        "Chat only": "C0",
        "Base only": "C1",
        "Shared": "C2",
        "Other": "darkgray",
    }

    # Apply log transformation to frequency data
    all_freqs = np.concatenate(
        [
            np.log10(
                df[(df["tag"] == tag) & (df["lmsys_freq"] > 1e-8)]["lmsys_freq"] + 1e-10
            )
            for tag in tags
        ]
    )

    # Determine bin edges in log space
    bins = np.linspace(min(all_freqs), max(all_freqs), 30)
    bin_width = bins[1] - bins[0]

    # Calculate bar width and offsets
    n_tags = len(tags)
    single_bar_width = bin_width / (n_tags)  # Add 1 for spacing
    offsets = np.linspace(
        -bin_width / 2 + single_bar_width / 2,
        bin_width / 2 - single_bar_width / 2,
        n_tags,
    )

    # Plot histogram for each tag
    for tag, offset in zip(tags, offsets):
        tag_data = df[df["tag"] == tag]
        # Apply log transformation to the data
        log_freqs = np.log10(
            tag_data["lmsys_freq"] + 1e-10
        )  # Add small constant to avoid log(0)
        counts, _ = np.histogram(log_freqs, bins=bins)
        normalized_counts = counts / counts.sum()
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.bar(
            bin_centers + offset,
            normalized_counts,
            width=single_bar_width,
            alpha=1.0,
            label=tag.replace("Chat only", "Chat-only").replace(
                "Base only", "Base-only"
            ),
            color=colors[tag],
        )

    # Styling
    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({"font.size": 20})

    ax.grid(True, alpha=0.15)

    # Use more human-readable tick values at nice round numbers
    log_ticks = np.array([-10, -8, -6, -4, -2])  # Powers of 10 for cleaner values
    log_ticks = log_ticks[
        np.logical_and(log_ticks >= min(all_freqs), log_ticks <= max(all_freqs))
    ]
    if len(log_ticks) < 3:  # Ensure we have enough ticks
        log_ticks = np.linspace(min(all_freqs), max(all_freqs), 5)
        log_ticks = np.round(log_ticks)  # Round to integers for cleaner display

    ax.set_xticks(log_ticks)
    ax.set_xticklabels(
        [f"$10^{{{int(x)}}}$" for x in log_ticks]
    )  # Use LaTeX for cleaner display

    ax.set_xlabel("Latent Frequency (log scale)")
    ax.set_ylabel("Density")

    # Move legend below plot
    ax.legend(fontsize=16, loc="upper left")

    plt.savefig("latent_frequency_histogram.pdf", bbox_inches="tight")
    plt.show()

