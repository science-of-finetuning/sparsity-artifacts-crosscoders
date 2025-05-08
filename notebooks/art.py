# %%
%load_ext autoreload
%autoreload 2
import sys

sys.path.append("..")
from tools.cc_utils import load_latent_df
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib as mpl
df = load_latent_df()
df = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss")
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 18})
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools.utils import load_json
from tools.cc_utils import load_latent_df

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

    exclusivity = (only_A_high + only_B_high) / (A_high + B_high - buckets[:, 3:, 3:].sum(axis=(1, 2)) + 1e-10)

    # Create the histogram with improved styling
    plt.figure(figsize=(8, 4))
    plt.hist(exclusivity, bins=15, color='C3', alpha=0.8, edgecolor='black')

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')

    # Customize axis labels with LaTeX formatting
    plt.xlabel(r"Twin Activation Divergence")#, fontsize=12)
    plt.ylabel(r"Pair Count")#, fontsize=12)

    # Add title if desired
    # plt.title("Distribution of Twin Activation Divergence", pad=10)

    # Customize ticks
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 30, 5))

    # Optional: Add mean line
    mean_exclusivity = np.mean(exclusivity)
    plt.axvline(mean_exclusivity, color='darkred', linestyle='--', alpha=0.5, 
                label=f'Mean: {mean_exclusivity:.2f}')
    plt.legend()
    if stitle:
        plt.title(title)
        

    # Adjust layout
    plt.tight_layout()
    plt.savefig(Path("results") / f"twin_activation_divergence_{title}.pdf", bbox_inches="tight")
    plt.show()
val_results = load_json("../results/twin_stats/validation_twins-l13_crosscoder_stats_all.json")
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
dec_df = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss") #pd.read_csv("/workspace/julian/repositories/representation-structure-comparison/results/eval_crosscoder/gemma-2-2b-L13-mu5.2e-02-lr1e-04-2x100M-local-shuffling-SAELoss/data/feature_df.csv")
if "dead" not in dec_df.columns:
    print("no dead column")
    dec_df["dead"] = False
green = "limegreen"
dec_ratios = dec_df["dec_norm_diff"][dec_df["dead"] == False]
ratio_error_values = dec_df["beta_ratio_error"][dec_df["dead"] == False]
ratio_reconstruction_values = dec_df["beta_ratio_reconstruction"][dec_df["dead"] == False]
values = 1 - dec_ratios
plt.figure(figsize=(6, 4.))
hist, bins, _ = plt.hist(values, bins=100, color="lightgray", label="Other", log=True)

# Color specific regions
mask_center = (bins[:-1] >= 0.4) & (bins[:-1] < 0.6)
mask_left = (bins[:-1] >= 0.9) & (bins[:-1] <= 1.0)
mask_right = (bins[:-1] >= 0.0) & (bins[:-1] < 0.1)

plt.hist(values, bins=bins, color="lightgray", log=True)  # Base gray histogram
plt.hist(
    values[((values >= 0.4) & (values < 0.6))], bins=bins, color="C1", label="Shared", log=True
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
plt.yticks([1, 10, 100, 1000, 4000], 
           ["$10^0$", "$10^1$", "$10^2$", "$10^3$", r"$4\times10^3$"])

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
plt.figure(figsize=(4, 3))

green = "limegreen"


thresholds = np.linspace(0, 1, 100)
cc_latents = []
for t in thresholds:
    # Count latents meeting criteria at each threshold
    count = np.sum((df_cc["beta_ratio_error"].abs() < t) & 
                   (df_cc["beta_ratio_reconstruction"].abs() < t) &
                   (df_cc["dead"] == False))
    cc_latents.append(count)

plt.plot(thresholds, cc_latents, label="L1", color="black")

# Calculate for base model
cc_latents = []
for t in thresholds:
    count = np.sum((df_topk["beta_ratio_error"].abs() < t) &
                   (df_topk["beta_ratio_reconstruction"].abs() < t))
    cc_latents.append(count)

plt.plot(thresholds, cc_latents, label="BatchTopK", color="black", linestyle="--")
plt.xlabel(r"Threshold $\pi$")
plt.ylabel("Count")
plt.legend(fontsize=16, loc=(0.28, 0.04))
plt.tight_layout()
# y log
plt.yscale("log")
plt.savefig("results/latents_vs_threshold.pdf", bbox_inches="tight")
plt.show()


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
min(cosims)


# %%
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
    return -x/np.sqrt(x**2), -y/np.sqrt(y**2)


def gradient_f2(x, y):
    return -x/np.sqrt(x**2 + y**2), -y/np.sqrt(x**2 + y**2)


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

# %%
from tools.cc_utils import CCLatent

CCLatent(57045)
# %%
df[df["tag"] == "Shared"]["lmsys_ctrl_%"].max()
# %%
[d["legendgroup"] for d in fig.data]
# %%
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
algo = KMeans

# filter for -0.1, 1.1
gmm_df = scatter_df[
    (scatter_df["beta_ratio_error"] >= -0.1)
    & (scatter_df["beta_ratio_error"] <= 1.1)
    & (scatter_df["beta_ratio_reconstruction"] >= -0.1)
    & (scatter_df["beta_ratio_reconstruction"] <= 1.1)
]
kwargs = dict(n_components=2) if algo == GaussianMixture else dict(n_clusters=2)
gmm = algo(**kwargs, random_state=42)
gmm.fit(gmm_df[["beta_ratio_error", "beta_ratio_reconstruction"]].values)
# %%
import plotly.express as px
clusters = gmm.predict(scatter_df[["beta_ratio_error", "beta_ratio_reconstruction"]].values).shape
# %%
# identify the Shared cluster
predictions = gmm.predict(gmm_df[["beta_ratio_error", "beta_ratio_reconstruction"]].values)
cluster_shared_counts = [
    (gmm_df["tag"] == "Shared")[predictions == i].sum() for i in range(2)
]
minority_cluster = np.argmin(cluster_shared_counts)

# Get indices of shared features in the minority cluster
shared_indices = gmm_df[
    (predictions == minority_cluster) & (gmm_df["tag"] == "Shared")
].index.tolist()

print(f"Cluster {minority_cluster} has {cluster_shared_counts[minority_cluster]} shared features")
print("Indices of shared features in minority cluster:", shared_indices)
fig = px.scatter(
    gmm_df,
    x="beta_ratio_error",
    y="beta_ratio_reconstruction",
    color=gmm.predict(gmm_df[["beta_ratio_error", "beta_ratio_reconstruction"]].values).astype(str),
    hover_data=["lmsys_ctrl_%", "freq", gmm_df.index],
    opacity=0.5,
)
shared_beta_ratios = scatter_df.loc[shared_indices, ["beta_ratio_error", "beta_ratio_reconstruction", "lmsys_ctrl_%", "freq"]]
fig2 = px.scatter(
    shared_beta_ratios,
    x="beta_ratio_error",
    y="beta_ratio_reconstruction",
    title="Scatter of Beta Ratios for Shared Indices",
    hover_data=["lmsys_ctrl_%", "freq"],
    opacity=1,
)
fig2.update_traces(
    marker=dict(color="green", size=10)
)
fig.add_trace(fig2.data[0])

fig.update_traces(
    marker=dict(
        size=10,
    )
)
fig.update_xaxes(range=[-0.1, 1.1])
fig.update_yaxes(range=[-0.1, 1.1])
fig.show()


# %%
import json
with open("results/shared_not_shared_indices.json", "w") as f:
    json.dump(shared_indices, f)

# %%
# Plotting the scatter of beta ratios for the shared indices


# %%
