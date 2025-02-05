from pycolors import TailwindColorPalette
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

COLORS = TailwindColorPalette()
RATIO_COLOR = COLORS.get_shade(5, 500)
BASE_COLOR = COLORS.get_shade(2, 600)
CHAT_COLOR = COLORS.get_shade(7, 600)


def plot_scaler_histograms(
    betas,
    target_type,
    baseline=None,
    title="Scaler Histogram Analysis for Reconstruction",
    xpos_legend=0.02,
    xpos_inset=0.85,
):
    """
    Create histograms comparing base and chat model scaler values and their ratio.

    Args:
        betas (dict): Dictionary containing reconstruction values for base and chat models
        target_type (str): Type of target to plot (e.g., "reconstruction", "latent")
    """
    recon_base = betas["normal"]["base"][target_type]
    recon_chat = betas["normal"]["it"][target_type]

    # Filter out NaN values from both reconstructions
    # Convert to numpy arrays to handle NaN values consistently
    recon_base = recon_base.numpy() if hasattr(recon_base, 'numpy') else recon_base
    recon_chat = recon_chat.numpy() if hasattr(recon_chat, 'numpy') else recon_chat
    
    valid_mask = ~(np.isnan(recon_base) | np.isnan(recon_chat))
    recon_base = recon_base[valid_mask]
    recon_chat = recon_chat[valid_mask]

    ratio = recon_base / recon_chat
    if baseline is not None:
        ratio_baseline = baseline / recon_chat

    # Calculate histogram data for main plots first
    hist_base = np.histogram(recon_base, bins=100)
    hist_chat = np.histogram(recon_chat, bins=100)
    max_count = max(np.max(hist_base[0]), np.max(hist_chat[0]))

    # Create the figure
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Base/Chat Ratio",
            "Base/Chat Ratio (zoom on -1 to 2)",
            "Base and Chat Scaler Values",
        ],
        column_widths=[0.25, 0.375, 0.375],
    )

    # Main histograms
    fig.add_trace(
        go.Histogram(
            x=ratio,
            nbinsx=100,
            marker_color=RATIO_COLOR,
            name="Base/Chat Ratio",
        ),
        row=1,
        col=1,
    )
    if baseline is not None:
        fig.add_trace(
            go.Histogram(
                x=ratio_baseline,
                nbinsx=100,
                marker_color="black",
                name="Base/Chat Ratio (Baseline)",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Histogram(
            x=ratio[(ratio > -1) & (ratio < 2)],
            nbinsx=100,
            marker_color=RATIO_COLOR,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    if baseline is not None:
        fig.add_trace(
            go.Histogram(
                x=ratio_baseline[(ratio_baseline > -1) & (ratio_baseline < 2)],
                nbinsx=100,
                marker_color="black",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Histogram(
            x=recon_base,
            name="Base Activation",
            nbinsx=100,
            marker_color=BASE_COLOR,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Histogram(
            x=recon_chat,
            name="Chat Activation",
            nbinsx=100,
            marker_color=CHAT_COLOR,
        ),
        row=1,
        col=3,
    )

    fig.update_annotations(yshift=-20)
    # Add inset histograms with secondary axes
    fig.add_trace(
        go.Histogram(
            x=recon_base[(recon_base >= -1) & (recon_base <= 2)],
            name="Base Model (zoom)",
            nbinsx=50,
            xaxis="x4",
            yaxis="y4",
            showlegend=False,
            marker_color=BASE_COLOR,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=recon_chat[(recon_chat >= -1) & (recon_chat <= 2)],
            name="Chat Model (zoom)",
            nbinsx=50,
            xaxis="x4",
            yaxis="y4",
            showlegend=False,
            marker_color=CHAT_COLOR,
        )
    )

    # Update layout with inset axes
    fig.update_layout(
        width=1400,
        height=500,
        showlegend=True,
        xaxis4=dict(
            domain=[xpos_inset, xpos_inset + 0.13],
            anchor="y4",
            range=[-1, 2],
            title="Zoom (-1 to 2)",
            showgrid=True,
            linecolor="black",
            linewidth=2,
            mirror=True,
        ),
        yaxis4=dict(
            domain=[0.5, 0.96],
            anchor="x4",
            showgrid=True,
            linecolor="black",
            linewidth=2,
            mirror=True,
        ),
        shapes=[
            dict(
                type="rect",
                xref="x3",
                yref="y3",
                x0=-1,
                x1=2,
                y0=0,
                y1=max_count * 2,
                line=dict(color="black", width=1, dash="dot"),
                fillcolor="rgba(255, 255, 255, 0.1)",
            )
        ],
    )

    # Add legend to leftmost plot in top left
    fig.update_layout(
        legend=dict(
            x=xpos_legend,
            y=0.95,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        )
    )
    fig.update_layout(
        title_text=title,
    )

    fig.update_xaxes(title_text="Beta Ratio", col=1)
    fig.update_xaxes(title_text="Beta Ratio", col=2)
    fig.update_xaxes(title_text="Beta", col=3)
    fig.update_yaxes(title_text="Count", col=1)

    # Increase font sizes
    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=16),
        legend_font=dict(size=14),
    )

    fig.update_xaxes(title_font=dict(size=14))
    fig.update_yaxes(title_font=dict(size=14))

    fig.update_annotations(font_size=14, y=1.1)

    return fig
