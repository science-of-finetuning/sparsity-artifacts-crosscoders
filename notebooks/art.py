# %%
from dotenv import load_dotenv

load_dotenv("./.env")
import IPython

IPython.get_ipython().run_line_magic("load_ext", "autoreload")
IPython.get_ipython().run_line_magic("autoreload", "2")
import sys

sys.path.append("..")
from tools.cc_utils import load_latent_df
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib as mpl

df = load_latent_df("l13_crosscoder")
df = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss")
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 18})
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools.utils import load_json
from tools.cc_utils import load_latent_df
Path("results").mkdir(exist_ok=True)

# %%
"""
==========================
Twin Activation Divergence
==========================
"""


def plot_twin_activation_divergence(buckets, title, stitle):
    both_high = buckets[:, 3:, 3:].sum(axis=(1, 2))
    only_A_high = buckets[:, 3:, :2].sum(axis=(1, 2))
    only_B_high = buckets[:, :2, 3:].sum(axis=(1, 2))
    A_high = buckets[:, 3:, :].sum(axis=(1, 2))
    B_high = buckets[:, :, 3:].sum(axis=(1, 2))

    exclusivity = (only_A_high + only_B_high) / (
        A_high + B_high - buckets[:, 3:, 3:].sum(axis=(1, 2)) + 1e-10
    )

    # Create the histogram with improved styling
    plt.figure(figsize=(8, 4))
    plt.hist(exclusivity, bins=15, color="C3", alpha=0.8, edgecolor="black")

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle="--")

    # Customize axis labels with LaTeX formatting
    plt.xlabel(r"Twin Activation Divergence")  # , fontsize=12)
    plt.ylabel(r"Pair Count")  # , fontsize=12)

    # Add title if desired
    # plt.title("Distribution of Twin Activation Divergence", pad=10)

    # Customize ticks
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 30, 5))

    # Optional: Add mean line
    mean_exclusivity = np.mean(exclusivity)
    plt.axvline(
        mean_exclusivity,
        color="darkred",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {mean_exclusivity:.2f}",
    )
    plt.legend()
    if stitle:
        plt.title(title)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(
        Path("results") / f"twin_activation_divergence_{title}.pdf", bbox_inches="tight"
    )
    plt.show()


val_results = load_json(
    "../results/twin_stats/validation_twins-l13_crosscoder_stats_all.json"
)
fw_buckets = val_results["fw_results"]["abs_max_act"]["buckets"]
fw_buckets = np.array(fw_buckets)
# plot_twin_activation_divergence(fw_buckets, "FineWeb", True)
lmsys_buckets = val_results["lmsys_results"]["abs_max_act"]["buckets"]
lmsys_buckets = np.array(lmsys_buckets)
# plot_twin_activation_divergence(lmsys_buckets, "LMSYS", True)
# merged buckets
merged_buckets = fw_buckets + lmsys_buckets
plot_twin_activation_divergence(merged_buckets, "merged", False)
# %%
"""
========================
Relative Norm Difference
========================
"""

# dec_df = load_latent_df()
dec_df = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss") 
if "dead" not in dec_df.columns:
    print("no dead column")
    dec_df["dead"] = False
green = "limegreen"
dec_ratios = dec_df["dec_norm_diff"][dec_df["dead"] == False]
ratio_error_values = dec_df["beta_ratio_error"][dec_df["dead"] == False]
ratio_reconstruction_values = dec_df["beta_ratio_reconstruction"][dec_df["dead"] == False]
values = 1 - dec_ratios
plt.figure(figsize=(6, 4.0))
hist, bins, _ = plt.hist(values, bins=100, color="lightgray", label="Other", log=True)

# Color specific regions
mask_center = (bins[:-1] >= 0.4) & (bins[:-1] < 0.6)
mask_left = (bins[:-1] >= 0.9) & (bins[:-1] <= 1.0)
mask_right = (bins[:-1] >= 0.0) & (bins[:-1] < 0.1)

plt.hist(values, bins=bins, color="lightgray", log=True)  # Base gray histogram
plt.hist(
    values[((values >= 0.4) & (values < 0.6))],
    bins=bins,
    color="C1",
    label="Shared",
    log=True,
)
plt.hist(values[((values >= 0.9))], bins=bins, color="C0", label="Chat-only", log=True)
# Define a range of thresholds with increasingly bright blue colors
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
blues = ['#000066', '#0000AA', '#0000DD', '#3333FF', '#6666FF', '#9999FF', '#CCCCFF', '#E6E6FF', '#F2F2FF', '#FCFCFF']  # Dark to bright blue

for i, thres in enumerate(thresholds):
    plt.hist(
        values[((values >= 0.9) & (ratio_error_values < thres) & (ratio_reconstruction_values < thres))], 
        bins=bins, 
        color=blues[i], 
        label=f"Chat-only (thres={thres})", 
        log=True
    )

plt.hist(values[(values <= 0.1)], bins=bins, color=green, label="Base-only", log=True)

# Update yticks for log scale
plt.yticks(
    [1, 10, 100, 1000, 4000], ["$10^0$", "$10^1$", "$10^2$", "$10^3$", r"$4\times10^3$"]
)

# Remove the original xticks call
# plt.xticks([0, 0.5, 1]), #["0\n(Base only)", "0.5\n(Shared)", "1\n(Chat only)"])
plt.xticks([0, 0.1, 0.4, 0.5, 0.6, 0.9, 1])
# # Get current axis and tick positions
ax = plt.gca()
# ticks = ax.get_xticks()

# # Create custom text annotations with different colors
# for tick, label in zip(ticks, ["0", "0.5", "1"]):
#     # Add value in black
#     ax.text(tick, -0.05, label,
#             color='black',
#             ha='center',
#             va='top',
#             transform=ax.get_xaxis_transform())

# Add colored descriptions below
# ax.text(0.05, -0.09, "(Base only)",
#         color='C2', ha='center', va='top',
#         transform=ax.get_xaxis_transform())
# ax.text(0.5, -0.09, "(Shared)",
#         color='C1', ha='center', va='top',
#         transform=ax.get_xaxis_transform())
# ax.text(0.95, -0.09, "(Chat only)",
#         color='C0', ha='center', va='top',
#         transform=ax.get_xaxis_transform())
# Add vertical lines at key thresholds
plt.axvline(x=0.1, color="green", linestyle="--", alpha=0.5)
plt.axvline(x=0.4, color="C1", linestyle="--", alpha=0.5)
plt.axvline(x=0.6, color="C1", linestyle="--", alpha=0.5)
plt.axvline(x=0.9, color="C0", linestyle="--", alpha=0.5)
plt.xlabel(
    "Relative Norm Difference",
)  # labelpad=23)
plt.ylabel("Latents")
plt.xlim(0, 1)
# plt.legend(loc="upper left")

plt.tight_layout()
plt.savefig(Path("results") / "decoder_norm_diff.pdf", bbox_inches="tight")

plt.show()
#%%
#%%
def plot_decoder_norm_diff(crosscoder, thres_error=0.3, thres_reconstruction=0.3, no_legend=False, log=False, ylim=None):
    chat_only_color = (0,0.6,1)
    chat_specific_color = (0,0,0.65)
    dec_df = load_latent_df(crosscoder)

    if "dead" not in dec_df.columns:
        print("no dead column")
        dec_df["dead"] = False
    green = "limegreen"
    dec_ratios = dec_df["dec_norm_diff"][dec_df["dead"] == False]
    ratio_error_values = dec_df["beta_ratio_error"][dec_df["dead"] == False]
    ratio_reconstruction_values = dec_df["beta_ratio_reconstruction"][dec_df["dead"] == False]
    values = 1 - dec_ratios
    plt.figure(figsize=(7, 4.))
    hist, bins, _ = plt.hist(values, bins=100, color="lightgray", label=None, log=log)

    # Color specific regions
    mask_center = (bins[:-1] >= 0.4) & (bins[:-1] < 0.6)
    mask_left = (bins[:-1] >= 0.9) & (bins[:-1] <= 1.0)
    mask_right = (bins[:-1] >= 0.0) & (bins[:-1] < 0.1)
    plt.hist(values, bins=bins, color="lightgray", log=log)  # Base gray histogram
    plt.hist(
        values[((values >= 0.4) & (values < 0.6))], bins=bins, color="C1", label="shared", log=log
    )
    plt.hist(values[((values >= 0.9))], bins=bins, color=chat_only_color, label="chat-only", log=log)

    plt.hist(
        values[((values >= 0.9) & (ratio_error_values < thres_error) & (ratio_reconstruction_values < thres_reconstruction))], 
        bins=bins, 
        color=chat_specific_color, 
        label=r'{\raggedleft chat-specific}\\{\fontsize{14pt}{3em}\selectfont{}$\nu^\epsilon < ' + str(thres_error) + r' $ \& $\nu^r < ' + str(thres_reconstruction) + r' $}'
    )
    plt.hist(values[(values <= 0.1)], bins=bins, color=green, label="base-only", log=log)

    if log:
        # Update yticks for log scale
        plt.yticks([1, 10, 100, 1000, 4000], 
                   ["$10^0$", "$10^1$", "$10^2$", "$10^3$", r"$4\times10^3$"])

    plt.xticks([0, 0.1, 0.4, 0.5, 0.6, 0.9, 1])
    ax = plt.gca()

    plt.axvline(x=0.1, color="green", linestyle="--", alpha=0.5)
    plt.axvline(x=0.4, color="C1", linestyle="--", alpha=0.5)
    plt.axvline(x=0.6, color="C1", linestyle="--", alpha=0.5)
    plt.axvline(x=0.9, color=chat_only_color, linestyle="--", alpha=0.5)
    plt.xlabel(
        "Relative Norm Difference $\Delta_\\text{norm}$",
    )
    plt.ylabel("Latents")
    if not no_legend:
        plt.legend(loc="upper left")

    # Create a zoomed-in inset for the 0.9 to 1.0 range
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    # Create the inset axes
    axins = inset_axes(plt.gca(), width="30%", height="30%", loc="upper right")

    # Plot the zoomed region
    bins = np.linspace(0.9, 1.0, 20)
    axins.hist(values, bins=bins, color="lightgray", log=log)
    axins.hist(
        values[((values >= 0.9))], 
        bins=bins, 
        color=chat_only_color, 
        log=log
    )
    axins.hist(
        values[((values >= 0.9) & (ratio_error_values.abs() < thres_error) & (ratio_reconstruction_values.abs() < thres_reconstruction))], 
        bins=bins, 
        color=chat_specific_color, 
        log=log
    )

    # Set the limits for the inset
    axins.set_xlim(0.9, 1.0)
    axins.set_xticks([0.95])

    if ylim:
        axins.set_ylim(ylim)

    # Draw connecting lines between the inset and the main plot
    mark_inset(ax, axins, loc1=4, loc2=3, fc="none", ec="gray")
    save_dir = Path("results") / crosscoder
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"decoder_norm_diff_{crosscoder}.pdf", bbox_inches="tight")
    print(f"Saved to {save_dir / f'decoder_norm_diff_{crosscoder}.pdf'}")
    plt.show()


thres_error = 0.2
thres_reconstruction = 0.5
crosscoder = "gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04"
plot_decoder_norm_diff(crosscoder, thres_error, thres_reconstruction, ylim=(0, 100))
crosscoder = "gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss"
plot_decoder_norm_diff(crosscoder, thres_error, thres_reconstruction, no_legend=True)

# %%



# %%
df_cc = load_latent_df("gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04")
df_cc = df_cc[df_cc["tag"] == "IT only"]
df_topk = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss")
df_topk = df_topk.sort_values(by="dec_norm_diff", ascending=True)
df_topk = df_topk.iloc[:len(df_cc)]

# %%
# Create line plot showing number of latents vs threshold
plt.figure(figsize=(7, 2.3))

green = "limegreen"


thresholds = np.linspace(0, 1, 100)
cc_latents = []
for t in thresholds:
    # Count latents meeting criteria at each threshold
    count = np.sum((df_cc["beta_ratio_error"].abs() < t) & 
                   (df_cc["beta_ratio_reconstruction"].abs() < t) &
                   (df_cc["dead"] == False))
    cc_latents.append(count)

plt.plot(thresholds, cc_latents, label="L1")

# Calculate for base model
cc_latents = []
for t in thresholds:
    count = np.sum((df_topk["beta_ratio_error"].abs() < t) &
                   (df_topk["beta_ratio_reconstruction"].abs() < t))
    cc_latents.append(count)

plt.plot(thresholds, cc_latents, label="BatchTopK", color=green)
plt.xlabel(r"$\leftarrow$ more chat-specific")
plt.ylabel("\# latents \nbelow \nthreshold", rotation=0, labelpad=40, y=0.2)
plt.legend()
plt.tight_layout()
# y log
plt.yscale("log")
plt.savefig("results/latents_vs_threshold_poster.svg", bbox_inches="tight")
plt.show()
# %%
# Create line plot showing number of latents vs threshold
def plot_latents_vs_threshold(
    df,
    label_map=None,
    color_map=None,
    linestyle_map=None,
    save_path="results/latents_vs_threshold.pdf",
    columns_to_threshold=None,
):
    """
    Plots the number of latents below threshold for each DataFrame in the df dictionary.
    Args:
        df: dict mapping names to DataFrames.
        label_map: optional dict mapping names to plot labels.
        color_map: optional dict mapping names to colors.
        linestyle_map: optional dict mapping names to linestyles.
        save_path: path to save the figure.
        columns_to_threshold: list or tuple of column names to threshold (default: ["beta_ratio_error", "beta_ratio_reconstruction"])
    """
    if columns_to_threshold is None:
        columns_to_threshold = ["beta_ratio_error", "beta_ratio_reconstruction"]

    plt.figure(figsize=(4, 3))
    thresholds = np.linspace(0, 1, 100)
    default_colors = ["black", "limegreen", "C0", "C1", "C2", "C3", "C4", "C5"]
    default_linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, (name, df_i) in enumerate(df.items()):
        latents = []
        for t in thresholds:
            # Build mask for all columns in columns_to_threshold
            mask = np.ones(len(df_i), dtype=bool)
            for col in columns_to_threshold:
                if col in df_i.columns:
                    mask &= (df_i[col].abs() < t)
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame '{name}'")
            latents.append(np.sum(mask))
        label = label_map[name] if label_map and name in label_map else name
        color = color_map[name] if color_map and name in color_map else default_colors[i % len(default_colors)]
        linestyle = linestyle_map[name] if linestyle_map and name in linestyle_map else default_linestyles[i % len(default_linestyles)]
        plt.plot(thresholds, latents, label=label, color=color, linestyle=linestyle)
    plt.xlabel(r"Threshold $\pi$")
    plt.ylabel("Count")
    plt.legend(fontsize=16, loc=(0.28, 0.04))
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
# %%
df_dict_gemma = {
    "BatchTopK": df_topk,
    "L1": df_cc,
}
plot_latents_vs_threshold(df_dict_gemma, save_path="results/latents_vs_threshold_gemma.pdf")
df_dict_llama = {
    "BatchTopK": load_latent_df("Meta-Llama-3.1-8B-L16-k222-lr1e-04-local-shuffling-Crosscoder"),
    "L1": load_latent_df("Meta-Llama-3.1-8B-L16-mu2.0e-02-lr1e-04-local-shuffling-CCLoss"),
}
plot_latents_vs_threshold(df_dict_llama, save_path="results/latents_vs_threshold_llama.pdf")
plot_latents_vs_threshold(df_dict_llama, columns_to_threshold=["beta_ratio_activation"], save_path="results/latents_vs_threshold_llama_activation.pdf")
# %%
df_dict_llama_2 = {
    "BatchTopK": load_latent_df("Llama-3.2-1B-L8-k100-lr1e-04-local-shuffling-Crosscoder"),
    "L1": load_latent_df("Llama-3.2-1B-L8-mu3.6e-02-lr1e-04-local-shuffling-CrosscoderLoss"),
}
plot_latents_vs_threshold(df_dict_llama_2, save_path="results/latents_vs_threshold_llama_1b.pdf")
# %%
import scipy.stats
# Create scatter plot comparing sum of ranks vs decoder norm difference
plt.figure(figsize=(6, 2.3))

# Calculate ranks for both metrics
df_cc["error_rank"] = df_cc["beta_ratio_error"].rank()
df_cc["recon_rank"] = df_cc["beta_ratio_reconstruction"].rank()
df_cc["rank_sum"] = df_cc["error_rank"] + df_cc["recon_rank"]

plt.scatter(df_cc["dec_norm_diff"], df_cc["rank_sum"], 
           alpha=0.5, label="L1")


# Same for base model
df_topk["error_rank"] = df_topk["beta_ratio_error"].rank()
df_topk["recon_rank"] = df_topk["beta_ratio_reconstruction"].rank()
df_topk["rank_sum"] = df_topk["error_rank"] + df_topk["recon_rank"]

# Compute correlation coefficients for CC
cc_error = df_cc[["dec_norm_diff", "beta_ratio_error"]].dropna()
cc_recon = df_cc[["dec_norm_diff", "beta_ratio_reconstruction"]].dropna()

corr_error, p_error = scipy.stats.pearsonr(cc_error["dec_norm_diff"], cc_error["beta_ratio_error"])
corr_recon, p_recon = scipy.stats.pearsonr(cc_recon["dec_norm_diff"], cc_recon["beta_ratio_reconstruction"])

print(f"CC error correlation: {corr_error:.2f} (p={p_error:.2e})")
print(f"CC reconstruction correlation: {corr_recon:.2f} (p={p_recon:.2e})")

# Compute correlation coefficients for TopK
topk_error = df_topk[["dec_norm_diff", "beta_ratio_error"]].dropna()
topk_recon = df_topk[["dec_norm_diff", "beta_ratio_reconstruction"]].dropna()

corr_error, p_error = scipy.stats.pearsonr(topk_error["dec_norm_diff"], topk_error["beta_ratio_error"])
corr_recon, p_recon = scipy.stats.pearsonr(topk_recon["dec_norm_diff"], topk_recon["beta_ratio_reconstruction"])

print(f"TopK error correlation: {corr_error:.2f} (p={p_error:.2e})")
print(f"TopK reconstruction correlation: {corr_recon:.2f} (p={p_recon:.2e})")

plt.scatter(df_topk["dec_norm_diff"], df_topk["rank_sum"],
           alpha=0.5, label="BatchTopK", color=green)

plt.xlabel("Relative Decoder Norm")
plt.ylabel("Sum of Ranks")
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "rank_sum_vs_norm_.pdf", bbox_inches="tight")
plt.show()


# %%
cosims = df["dec_cos_sim"]
plt.figure(figsize=(6, 3.9))
# Define masks for different regions
mask_center = (values >= 0.4) & (values < 0.6)
mask_left = (values >= 0.9) & (values <= 1.0)
mask_right = (values >= 0.0) & (values < 0.1)
mask_other = ~(mask_center | mask_left | mask_right)

data = [cosims[mask_left], cosims[mask_right], cosims[mask_other], cosims[mask_center]]
labels = ["Chat only", "Base only", "Other", "Shared"]
colors = ["C0", green, "darkgray", "C1"]

# Create weights for each group to normalize them independently
weights = []
for group in data:
    weights.append(np.ones_like(group) / len(group))

plt.hist(data, bins=20, label=labels, color=colors, histtype="bar", weights=weights)
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.tight_layout()
plt.legend()
plt.savefig(Path("results") / "decoder_cos_sim.pdf", bbox_inches="tight")
plt.show()

# %%
"""
========================
L1 Gradients
========================
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Create coordinate grid
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)
arrow_factor = 4

# Calculate the two functions
f1 = np.sqrt(X**2) + np.sqrt(Y**2)  # sqrt(x^2) + sqrt(y^2)
f2 = np.sqrt(X**2 + Y**2)  # sqrt(x^2 + y^2)


# Calculate gradients for -f1 and -f2
def gradient_f1(x, y):
    return -x / np.sqrt(x**2), -y / np.sqrt(y**2)


def gradient_f2(x, y):
    return -x / np.sqrt(x**2 + y**2), -y / np.sqrt(x**2 + y**2)


# Create figure with two subplots
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.03])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Plot first function with gradient field
im1 = ax1.imshow(f1, extent=[0, 100, 0, 100], origin="lower", cmap="RdBu_r")
im2 = ax2.imshow(
    f2,
    extent=[0, 100, 0, 100],
    origin="lower",
    cmap="RdBu_r",
    vmin=min(f1.min(), f2.min()),
    vmax=max(f1.max(), f2.max()),
)

# Add colorbar on the last subplot
plt.colorbar(im1, cax=ax3, label="Function value")

# Add gradient arrows for first function
arrow_points = np.linspace(5, 95, 15)
XX, YY = np.meshgrid(arrow_points, arrow_points)
U1, V1 = gradient_f1(XX, YY)

ax1.quiver(
    XX,
    YY,
    arrow_factor * U1,
    arrow_factor * V1,
    color="black",
    width=0.004,
    headwidth=3,
    headlength=4,
    scale_units="xy",
    scale=1,
)
ax1.set_title("$\sqrt{x^2} + \sqrt{y^2}$ with $-\\nabla f$ arrows", pad=20)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

# Add gradient arrows for second function
U2, V2 = gradient_f2(XX, YY)
ax2.quiver(
    XX,
    YY,
    arrow_factor * U2,
    arrow_factor * V2,
    color="black",
    width=0.004,
    headwidth=3,
    headlength=4,
    scale_units="xy",
    scale=1,
)
ax2.set_title("$\sqrt{x^2 + y^2}$ with $-\\nabla f$ arrows", pad=20)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

plt.tight_layout()
plt.savefig(Path("results") / "decoder_gradient_v2.pdf", bbox_inches="tight")
plt.show()

# %%
"""
========================
Scatter of beta ratio error and reconstruction
========================
"""
# fig = px.scatter(scatter_df, x="beta_ratio_error", y="beta_ratio_reconstruction", color="tag", text=scatter_df.index, hover_data=["lmsys_ctrl_%", "freq"], opacity=0.5)
# fig.update_traces(marker=dict(size=10))
# # Add blue outline for points with lmsys_ctrl_% > 0.5
# fig.update_traces(marker=dict(
#     line=dict(
#         color=np.where((scatter_df['lmsys_ctrl_%'].fillna(0) > 0.5), 'darkgreen', 'rgba(0,0,0,0)'),
#         width=2
#     )
# ))
# fig.update_xaxes(range=[-0.1, 1.1])
# fig.update_yaxes(range=[-0.1, 1.1])

# fig.show()
# Split the data
scatter_df = df[
    df["beta_ratio_error"].notna() & df["beta_ratio_reconstruction"].notna()
]
df_high = scatter_df[scatter_df["lmsys_ctrl_%"].fillna(0) > 0.5]
df_low = scatter_df[scatter_df["lmsys_ctrl_%"].fillna(0) <= 0.5]

# Create the figure with the "low" values (no outline)
fig = px.scatter(
    df_low,
    x="beta_ratio_error",
    y="beta_ratio_reconstruction",
    color="tag",
    hover_data=["lmsys_ctrl_%", "freq", df_low.index],
    opacity=0.5,
)

scat = px.scatter(
    df_high,
    x="beta_ratio_error",
    y="beta_ratio_reconstruction",
    color="tag",
    # text=df_high.index,
    hover_data=["lmsys_ctrl_%", "freq", df_high.index],
    opacity=0.9,
)
# Add a trace for the high values (with outline)
fig.add_trace(scat.data[0])
fig.add_trace(scat.data[1])
fig.update_traces(
    marker=dict(
        size=10,
    )
)
# Update only the newly added trace to have the darkgreen outline
fig.data[-1].marker.line = dict(color="darkgreen", width=2)
fig.data[-2].marker.line = dict(color="darkgreen", width=2)
fig.update_xaxes(range=[-0.1, 1.1])
fig.update_yaxes(range=[-0.1, 1.1])
fig.show()


# %%
"""
========================
KL plots
========================
"""
import plotly.graph_objects as go
import numpy as np
from display_metrics.shared import (
    load_metrics,
    build_complete_dataframe,
)
import plotly.io as pio

pio.kaleido.scope.mathjax = None

data_l1 = load_metrics(
    "../results/interv_effects/all_l1_crosscoder_dogfish-bat-bison-octopus.json"
)
data_batchtopk = load_metrics(
    "../results/interv_effects/1741127183_ultrachat-gemma-batchtopk-CC_cornflower-caterpillar_result.json"
)

# Constants for setup strings
VANILLA_BASE2CHAT = "vanilla base2chat"
PATCH_BASE_ERROR = "patch base error cchat"
PATCH_CHAT_ERROR = "patch chat error cchat"


# CrossCoder Both ratios group (50%)
def patch_best_latent_scaling(use_additive_steering):
    add_str = " add" if use_additive_steering else ""
    return f"patch all{add_str} rank sum pareto 50pct cchat"


def patch_worst_latent_scaling(use_additive_steering):
    add_str = " add" if use_additive_steering else ""
    return f"patch all{add_str} rank sum antipareto 50pct cchat"


# RND (Norm difference) group
def patch_best_norm_diff(use_additive_steering):
    add_str = " add" if use_additive_steering else ""
    return f"patch all{add_str} dec norm diff pareto 50pct cchat"


def patch_worst_norm_diff(use_additive_steering):
    add_str = " add" if use_additive_steering else ""
    return f"patch all{add_str} dec norm diff antipareto 50pct cchat"

def plot_kl_both_models(
    data_l1,
    data_batchtopk,
    metric_type="kl-instruct",
    category="k_first",
    text_mode="above",
    output_file=None,
    show_legend=True,
    use_additive_steering=False,
):
    """
    Plot KL divergence for both L1 and BatchTopK models.

    Parameters:
    -----------
    data_l1 : dict
        Metrics data for L1 model
    data_batchtopk : dict
        Metrics data for BatchTopK model
    metric_type : str
        Metric type to plot (default: "kl-instruct")
    category : str
        Category to plot (default: "k_first", alternative: "all")
    text_mode : str
        Text display mode: "in" (inside bars), "above" (above error bars), or "none" (no text)
    output_file : str, optional
        Path to save the plot (default: None, no saving)
    show_legend : bool
        Whether to display the legend (default: True)
    """
    categories = [category]  # Use the specified category

    # Define setups based on the groups from the drawing
    requested_setups = [
        # Vanilla group
        VANILLA_BASE2CHAT,  # Base->Chat
        PATCH_BASE_ERROR,  # Base error + chat reconstruction
        # Chat error
        PATCH_CHAT_ERROR,  # Chat error + chat reconstruction
        # CrossCoder Both ratios group (50%)
        patch_best_latent_scaling(use_additive_steering),  # Best latents
        patch_worst_latent_scaling(use_additive_steering),  # Worst latents
        # RND (Norm difference) group
        patch_best_norm_diff(use_additive_steering),  # Best latents
        patch_worst_norm_diff(use_additive_steering),  # Worst latents
        # Ctrl group
        # "patch base ctrl cchat",  # Control tokens replacement
    ]

    # Function to process data for a specific model
    def process_model_data(data, model_name):
        # Build dataframe for selected setups across categories
        df = build_complete_dataframe(data)
        if df is None:
            print(f"No valid data available for any setup in {model_name}")
            return None

        # Filter for requested metric and categories
        df = df.loc[:, (categories, metric_type)]

        # Find which setups are available in the data
        valid_setups = [s for s in requested_setups if s in df.index]
        missing_setups = [s for s in requested_setups if s not in df.index]

        if missing_setups:
            print(f"Warning: Missing data for setups in {model_name}:", missing_setups)

        if not valid_setups:
            print(f"No data available for any of the requested setups in {model_name}")
            return None

        df = df.loc[valid_setups]
        df = df.dropna(axis=1, how="all")
        return df

    # Process data for both models
    df_l1 = process_model_data(data_l1, "L1")
    df_batchtopk = process_model_data(data_batchtopk, "BatchTopK")

    # Check if we have valid data for at least one model
    if df_l1 is None and df_batchtopk is None:
        print("No valid data available for any model")
        return None
    else:
        # Define colors for different categories
        COLOR_LOWEST_50PCT = "#59cd90"  # Dark green for best latents
        COLOR_HIGHEST_50PCT = "#ee6352"  # Light red for worst latents
        COLOR_ALL = "#7eb2dd"  # Light blue for "All" category
        COLOR_NONE = "#445e93"  # Dark blue for "None" category
        COLOR_CHAT_ERROR = "#9370db"  # Purple for chat error

        # Define hierarchical labels
        group_labels = ["<b>None</b>", "<b>L1</b>", "<b>BatchTopK</b>"]

        fig = go.Figure()

        # Calculate positions for grouped bars
        n_groups = len(group_labels)
        # Move groups closer together - make None much closer
        group_positions = np.array(
            [0, 0.6, 1.71]
        )  # Further reduced spacing between groups
        group_width = 0.8

        # Define positions for each bar
        # For None: just one bar
        # For L1: All, Chat Error, Latent Scaling (Best+Worst), ∆<sub>norm</sub> (Best+Worst)
        # For BatchTopK: All, Chat Error, Latent Scaling (Best+Worst), ∆<sub>norm</sub> (Best+Worst)

        # Define method positions within each group (further increased spacing between methods)
        method_positions = {
            "None": 0,
            "All": -0.32,  # Further increased spacing between subgroups
            "Chat Error": -0.19,  # Between All and Latent Scaling
            "Latent Scaling": 0.1,
            "∆<sub>norm</sub>": 0.45,  # Further increased spacing between subgroups
        }

        # Define bar width for all bars (1.5x larger)
        bar_width = group_width / 10 * 1.5

        # Define bar positions with exact positioning to avoid overlap
        bar_positions = {
            # None baseline
            VANILLA_BASE2CHAT: (group_positions[0], 0),
            # L1 group
            "L1_" + PATCH_BASE_ERROR: (group_positions[1], method_positions["All"]),
            # Chat Error
            "L1_"
            + PATCH_CHAT_ERROR: (
                group_positions[1],
                method_positions["Chat Error"],
            ),
            # Latent Scaling pair
            "L1_"
            + patch_best_latent_scaling(use_additive_steering): (
                group_positions[1],
                method_positions["Latent Scaling"] - bar_width / 2,
            ),
            "L1_"
            + patch_worst_latent_scaling(use_additive_steering): (
                group_positions[1],
                method_positions["Latent Scaling"] + bar_width / 2,
            ),
            # ∆<sub>norm</sub> pair
            "L1_"
            + patch_best_norm_diff(use_additive_steering): (
                group_positions[1],
                method_positions["∆<sub>norm</sub>"] - bar_width / 2,
            ),
            "L1_"
            + patch_worst_norm_diff(use_additive_steering): (
                group_positions[1],
                method_positions["∆<sub>norm</sub>"] + bar_width / 2,
            ),
            # BatchTopK group
            "BatchTopK_"
            + PATCH_BASE_ERROR: (
                group_positions[2],
                method_positions["All"],
            ),
            # Chat Error
            "BatchTopK_"
            + PATCH_CHAT_ERROR: (
                group_positions[2],
                method_positions["Chat Error"],
            ),
            # Latent Scaling pair
            "BatchTopK_"
            + patch_best_latent_scaling(use_additive_steering): (
                group_positions[2],
                method_positions["Latent Scaling"] - bar_width / 2,
            ),
            "BatchTopK_"
            + patch_worst_latent_scaling(use_additive_steering): (
                group_positions[2],
                method_positions["Latent Scaling"] + bar_width / 2,
            ),
            # ∆<sub>norm</sub> pair
            "BatchTopK_"
            + patch_best_norm_diff(use_additive_steering): (
                group_positions[2],
                method_positions["∆<sub>norm</sub>"] - bar_width / 2,
            ),
            "BatchTopK_"
            + patch_worst_norm_diff(use_additive_steering): (
                group_positions[2],
                method_positions["∆<sub>norm</sub>"] + bar_width / 2,
            ),
        }

        # Function to add text annotation based on mode
        def add_text_annotation(x, y, error_val, text_value):
            if text_mode == "none":
                return  # No text annotation

            if text_mode == "in":
                # Inside bar
                fig.add_annotation(
                    x=x,
                    y=y / 2,
                    text=f"{text_value:.3f}",
                    showarrow=False,
                    textangle=270,  # Vertical text
                    font=dict(size=10, color="white"),
                )
            elif text_mode == "above":
                # Above error bar with more space
                fig.add_annotation(
                    x=x,
                    y=y + error_val + 0.05,  # Increased gap above error bar
                    text=f"{text_value:.2f}",
                    showarrow=False,
                    textangle=0,  # Horizontal text
                    font=dict(size=10, color="black"),
                )

        # Function to calculate error value
        def calculate_error(df, setup, category, metric_type):
            return (
                1.96
                * (
                    df.loc[setup, (category, metric_type)]["variance"]
                    / df.loc[setup, (category, metric_type)]["n"]
                )
                ** 0.5
            )

        # Add None baseline bar
        if df_l1 is not None and VANILLA_BASE2CHAT in df_l1.index:
            group_pos, offset = bar_positions[VANILLA_BASE2CHAT]
            mean_val = df_l1.loc[VANILLA_BASE2CHAT, (category, metric_type)]["mean"]
            error_val = calculate_error(df_l1, VANILLA_BASE2CHAT, category, metric_type)

            fig.add_trace(
                go.Bar(
                    x=[group_pos + offset],
                    y=[mean_val],
                    width=bar_width,
                    marker_color=COLOR_NONE,
                    showlegend=False,  # Remove from legend
                    name="None",
                    error_y=dict(
                        type="data",
                        array=[error_val],
                        visible=True,
                        thickness=1.5,
                        width=6,
                    ),
                )
            )

            # Add text annotation
            add_text_annotation(group_pos + offset, mean_val, error_val, mean_val)

        # Add "All" bars for each crosscoder
        # L1 All
        if df_l1 is not None and PATCH_BASE_ERROR in df_l1.index:
            group_pos, offset = bar_positions["L1_" + PATCH_BASE_ERROR]
            mean_val = df_l1.loc[PATCH_BASE_ERROR, (category, metric_type)]["mean"]
            error_val = calculate_error(df_l1, PATCH_BASE_ERROR, category, metric_type)

            fig.add_trace(
                go.Bar(
                    x=[group_pos + offset],
                    y=[mean_val],
                    width=bar_width,
                    marker_color=COLOR_ALL,
                    showlegend=False,  # Remove from legend
                    name="All",
                    error_y=dict(
                        type="data",
                        array=[error_val],
                        visible=True,
                        thickness=1.5,
                        width=6,
                    ),
                )
            )

            # Add text annotation
            add_text_annotation(group_pos + offset, mean_val, error_val, mean_val)

        # BatchTopK All
        if df_batchtopk is not None and PATCH_BASE_ERROR in df_batchtopk.index:
            group_pos, offset = bar_positions["BatchTopK_" + PATCH_BASE_ERROR]
            mean_val = df_batchtopk.loc[PATCH_BASE_ERROR, (category, metric_type)][
                "mean"
            ]
            error_val = calculate_error(
                df_batchtopk, PATCH_BASE_ERROR, category, metric_type
            )

            fig.add_trace(
                go.Bar(
                    x=[group_pos + offset],
                    y=[mean_val],
                    width=bar_width,
                    marker_color=COLOR_ALL,
                    showlegend=False,  # Already in legend
                    name="All",
                    error_y=dict(
                        type="data",
                        array=[error_val],
                        visible=True,
                        thickness=1.5,
                        width=6,
                    ),
                )
            )

            # Add text annotation
            add_text_annotation(group_pos + offset, mean_val, error_val, mean_val)

        # Add Chat Error bar for L1
        if df_l1 is not None and PATCH_CHAT_ERROR in df_l1.index:
            group_pos, offset = bar_positions["L1_" + PATCH_CHAT_ERROR]
            mean_val = df_l1.loc[PATCH_CHAT_ERROR, (category, metric_type)]["mean"]
            error_val = calculate_error(df_l1, PATCH_CHAT_ERROR, category, metric_type)

            fig.add_trace(
                go.Bar(
                    x=[group_pos + offset],
                    y=[mean_val],
                    width=bar_width,
                    marker_color=COLOR_CHAT_ERROR,
                    showlegend=False,
                    name="ε<sub>chat</sub>",
                    error_y=dict(
                        type="data",
                        array=[error_val],
                        visible=True,
                        thickness=1.5,
                        width=6,
                    ),
                )
            )

            # Add text annotation
            add_text_annotation(group_pos + offset, mean_val, error_val, mean_val)

        # Add Chat Error bar for BatchTopK
        if df_batchtopk is not None and PATCH_CHAT_ERROR in df_batchtopk.index:
            group_pos, offset = bar_positions["BatchTopK_" + PATCH_CHAT_ERROR]
            mean_val = df_batchtopk.loc[PATCH_CHAT_ERROR, (category, metric_type)][
                "mean"
            ]
            error_val = calculate_error(
                df_batchtopk, PATCH_CHAT_ERROR, category, metric_type
            )

            fig.add_trace(
                go.Bar(
                    x=[group_pos + offset],
                    y=[mean_val],
                    width=bar_width,
                    marker_color=COLOR_CHAT_ERROR,
                    showlegend=False,  # Already in legend
                    name="ε<sub>chat</sub>",
                    error_y=dict(
                        type="data",
                        array=[error_val],
                        visible=True,
                        thickness=1.5,
                        width=6,
                    ),
                )
            )

            # Add text annotation
            add_text_annotation(group_pos + offset, mean_val, error_val, mean_val)

        # Add L1 model bars for Best/Worst pairs
        if df_l1 is not None:
            for method in ["Latent Scaling", "∆<sub>norm</sub>"]:
                best_setup = None
                worst_setup = None

                if method == "Latent Scaling":
                    best_setup = patch_best_latent_scaling(use_additive_steering)
                    worst_setup = patch_worst_latent_scaling(use_additive_steering)
                else:  # ∆<sub>norm</sub>
                    best_setup = patch_best_norm_diff(use_additive_steering)
                    worst_setup = patch_worst_norm_diff(use_additive_steering)

                # Add Best bar
                if best_setup in df_l1.index:
                    group_pos, offset = bar_positions[f"L1_{best_setup}"]
                    mean_val = df_l1.loc[best_setup, (category, metric_type)]["mean"]
                    error_val = calculate_error(
                        df_l1, best_setup, category, metric_type
                    )

                    # Only add to legend if this is the first occurrence of Best and legend is enabled
                    add_to_legend = show_legend and "Lowest 50%" not in [
                        t.name for t in fig.data
                    ]

                    fig.add_trace(
                        go.Bar(
                            x=[group_pos + offset],
                            y=[mean_val],
                            width=bar_width,
                            marker_color=COLOR_LOWEST_50PCT,
                            showlegend=add_to_legend,
                            name="Lowest 50%",
                            error_y=dict(
                                type="data",
                                array=[error_val],
                                visible=True,
                                thickness=1.5,
                                width=6,
                            ),
                        )
                    )

                    # Add text annotation
                    add_text_annotation(
                        group_pos + offset, mean_val, error_val, mean_val
                    )

                # Add Worst bar
                if worst_setup in df_l1.index:
                    group_pos, offset = bar_positions[f"L1_{worst_setup}"]
                    mean_val = df_l1.loc[worst_setup, (category, metric_type)]["mean"]
                    error_val = calculate_error(
                        df_l1, worst_setup, category, metric_type
                    )

                    # Only add to legend if this is the first occurrence of Worst and legend is enabled
                    add_to_legend = show_legend and "Highest 50%" not in [
                        t.name for t in fig.data
                    ]

                    fig.add_trace(
                        go.Bar(
                            x=[group_pos + offset],
                            y=[mean_val],
                            width=bar_width,
                            marker_color=COLOR_HIGHEST_50PCT,
                            showlegend=add_to_legend,
                            name="Highest 50%",
                            error_y=dict(
                                type="data",
                                array=[error_val],
                                visible=True,
                                thickness=1.5,
                                width=6,
                            ),
                        )
                    )

                    # Add text annotation
                    add_text_annotation(
                        group_pos + offset, mean_val, error_val, mean_val
                    )

        # Add BatchTopK model bars for Best/Worst pairs
        if df_batchtopk is not None:
            for method in ["Latent Scaling", "∆<sub>norm</sub>"]:
                best_setup = None
                worst_setup = None

                if method == "Latent Scaling":
                    best_setup = patch_best_latent_scaling(use_additive_steering)
                    worst_setup = patch_worst_latent_scaling(use_additive_steering)
                else:  # ∆<sub>norm</sub>
                    best_setup = patch_best_norm_diff(use_additive_steering)
                    worst_setup = patch_worst_norm_diff(use_additive_steering)

                # Add Best bar
                if best_setup in df_batchtopk.index:
                    group_pos, offset = bar_positions[f"BatchTopK_{best_setup}"]
                    mean_val = df_batchtopk.loc[best_setup, (category, metric_type)][
                        "mean"
                    ]
                    error_val = calculate_error(
                        df_batchtopk, best_setup, category, metric_type
                    )

                    fig.add_trace(
                        go.Bar(
                            x=[group_pos + offset],
                            y=[mean_val],
                            width=bar_width,
                            marker_color=COLOR_LOWEST_50PCT,
                            showlegend=False,  # Already in legend
                            name="Lowest 50%",
                            error_y=dict(
                                type="data",
                                array=[error_val],
                                visible=True,
                                thickness=1.5,
                                width=6,
                            ),
                        )
                    )

                    # Add text annotation
                    add_text_annotation(
                        group_pos + offset, mean_val, error_val, mean_val
                    )

                # Add Worst bar
                if worst_setup in df_batchtopk.index:
                    group_pos, offset = bar_positions[f"BatchTopK_{worst_setup}"]
                    mean_val = df_batchtopk.loc[worst_setup, (category, metric_type)][
                        "mean"
                    ]
                    error_val = calculate_error(
                        df_batchtopk, worst_setup, category, metric_type
                    )

                    fig.add_trace(
                        go.Bar(
                            x=[group_pos + offset],
                            y=[mean_val],
                            width=bar_width,
                            marker_color=COLOR_HIGHEST_50PCT,
                            showlegend=False,  # Already in legend
                            name="Highest 50%",
                            error_y=dict(
                                type="data",
                                array=[error_val],
                                visible=True,
                                thickness=1.5,
                                width=6,
                            ),
                        )
                    )

                    # Add text annotation
                    add_text_annotation(
                        group_pos + offset, mean_val, error_val, mean_val
                    )

        # Add method labels below each group
        method_labels = {
            1: [
                "All",
                "ε<sub>chat</sub>",
                "Latent Scaling",
                "∆<sub>norm</sub>",
            ],  # L1 group
            2: [
                "All",
                "ε<sub>chat</sub>",
                "Latent Scaling",
                "∆<sub>norm</sub>",
            ],  # BatchTopK group
        }

        for group_idx, labels in method_labels.items():
            for i, label in enumerate(labels):
                position = method_positions[
                    label.replace("ε<sub>chat</sub>", "Chat Error")
                ]
                fig.add_annotation(
                    x=group_positions[group_idx] + position,
                    y=-0.01,
                    text=label,
                    showarrow=False,
                    yshift=-10,
                    font=dict(size=12),
                )

        # Update layout
        fig.update_layout(
            title=None,  # Remove title
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),  # Reduce margins
            xaxis=dict(
                ticktext=group_labels,
                tickvals=group_positions,
                title="",
                tickfont=dict(size=16),
                showgrid=False,
                showline=False,
                linewidth=1,
                linecolor="black",
            ),
            yaxis=dict(
                title="KL divergence",
                titlefont=dict(size=16),
                tickfont=dict(size=16),
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgrey",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="grey",
                showline=False,
                linewidth=1,
                linecolor="black",
                # Add explicit tick values and make the last one bold
                tickmode="array",
                tickvals=(
                    [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    if category == "k_first"
                    else [0, 0.05, 0.1, 0.15, 0.2, 0.25]
                ),
                ticktext=(
                    ["0.0", "0.2", "0.4", "0.6", "0.8", "<b>1.0</b>"]
                    if category == "k_first"
                    else ["0.0", "0.05", "0.1", "0.15", "0.2", "<b>0.25</b>"]
                ),
            ),
            bargap=0,
            plot_bgcolor="white",
            bargroupgap=0.05,
            # showlegend=show_legend,  # Control legend visibility
            legend=dict(
                orientation="h", yanchor="bottom", y=0.95, xanchor="right", x=1
            ),
        )

        # Adjust x-axis range to match group positioning
        fig.update_xaxes(range=[-0.1, group_positions[-1] + 0.6])

        fig.update_layout(
            height=300,
            width=600,  # Slightly wider to accommodate both models
            margin=dict(t=30, b=0, l=0, r=0),  # Adjust top margin for legend
        )

        # Make the topmost y-axis tick bold
        fig.update_yaxes(tickfont=dict(size=16))

        fig.show()

        # Save the figure if output_file is provided
        if output_file:
            fig.write_image(output_file)

        return fig


# %%
fig = plot_kl_both_models(
    data_l1,
    data_batchtopk,
    text_mode="in",
    show_legend=True,
    output_file="results/first_k_kl_both_models.pdf",
    use_additive_steering=True,
)

plot_kl_both_models(
    data_l1,
    data_batchtopk,
    text_mode="in",
    show_legend=True,
    output_file="results/first_k_kl_both_models_replace.pdf",
    use_additive_steering=False,
)

fig = plot_kl_both_models(
    data_l1,
    data_batchtopk,
    category="all",
    text_mode="in",
    show_legend=False,
    output_file="results/all_kl_both_models.pdf",
    use_additive_steering=True,
)

fig = plot_kl_both_models(
    data_l1,
    data_batchtopk,
    category="all",
    text_mode="in",
    show_legend=False,
    output_file="results/all_kl_both_models_replace.pdf",
    use_additive_steering=False,
)


# fig = plot_kl_both_models(data_l1, data_batchtopk, category="post_k_first", text_mode="in", show_legend=False,
#                          output_file="results/post_k_first_kl_both_models.pdf")

# No text labels and no legend
# fig = plot_kl_both_models(data_l1, data_batchtopk, text_mode="none", show_legend=False,
#  output_file="results/first_k_kl_instruct_both_models_no_text_no_legend.pdf")
