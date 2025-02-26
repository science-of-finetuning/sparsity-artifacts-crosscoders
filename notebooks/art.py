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
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 20})
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools.utils import load_json
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

dec_df = pd.read_csv("/workspace/clement/repos/representation-structure-comparison/results/eval_crosscoder/gemma-2-2b-L13-mu5.2e-02-lr1e-04-local-shuffling-SAEloss_model_final.pt/data/feature_df.csv")
if "dead" not in dec_df.columns:
    print("no dead column")
    dec_df["dead"] = False
green = "limegreen"
dec_ratios = dec_df["dec_norm_diff"][dec_df["dead"] == False]
values = 1 - dec_ratios
plt.figure(figsize=(6, 4.))
hist, bins, _ = plt.hist(values, bins=100, color="lightgray", label="Other")
# Color specific regions
mask_center = (bins[:-1] >= 0.4) & (bins[:-1] < 0.6)
mask_left = (bins[:-1] >= 0.9) & (bins[:-1] <= 1.0)
mask_right = (bins[:-1] >= 0.0) & (bins[:-1] < 0.1)

plt.hist(values, bins=bins, color="lightgray")  # Base gray histogram
plt.hist(
    values[((values >= 0.4) & (values < 0.6))], bins=bins, color="C1", label="Shared"
)
plt.hist(values[((values >= 0.9))], bins=bins, color="C0", label="Chat-only")
plt.hist(values[(values <= 0.1)], bins=bins, color=green, label="Base-only")

# Remove the original xticks call
# plt.xticks([0, 0.5, 1]), #["0\n(Base only)", "0.5\n(Shared)", "1\n(Chat only)"])
plt.xticks([0, 0.1, 0.4, 0.5, 0.6, 0.9, 1])
plt.yticks([0, 2000, 4000], ["$0$", r"$2\text{k}$", r"$4\text{k}$"])
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
plt.legend(loc="upper left")

plt.tight_layout()
plt.savefig(Path("results") / "decoder_norm_diff.pdf", bbox_inches="tight")

plt.show()
# %%
cosims = df["dec_cos_sim"][df["dead"] == False]
plt.figure(figsize=(6, 3.9))
# Define masks for different regions
mask_center = (values >= 0.4) & (values < 0.6)
mask_left = (values >= 0.9) & (values <= 1.0)
mask_right = (values >= 0.0) & (values < 0.1)
mask_other = ~(mask_center | mask_left | mask_right)

data = [cosims[mask_left], cosims[mask_right], cosims[mask_other], cosims[mask_center]]
labels = ["Chat only", "Base only", "Other", "Shared"]
colors = ["C0", green, "lightgray", "C1"]

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
