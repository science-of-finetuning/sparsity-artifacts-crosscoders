# %%
import sys

sys.path.append("..")
from tools.utils import get_available_models, load_dictionary_model, chunked_max_cosim
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import torch as th

th.set_grad_enabled(False)
get_available_models()

# %%
model1 = load_dictionary_model("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss").to(
    "cuda"
)
model2 = load_dictionary_model(
    "gemma-2-2b-L13-k100-lr1e-04-local-shuffling-SAELoss"
).to("cuda")


base_cosim = cosine_similarity(
    model1.decoder.weight[0], model2.decoder.weight[0], dim=1
)
chat_cosim = cosine_similarity(
    model1.decoder.weight[1], model2.decoder.weight[1], dim=1
)
print(base_cosim.shape, chat_cosim.shape)


def cosim_hist(cosim_base, cosim_chat):
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Flatten the cosine similarity matrices to 1D arrays for histograms
    base_cosim_flat = cosim_base.cpu().detach().numpy().flatten()
    chat_cosim_flat = cosim_chat.cpu().detach().numpy().flatten()

    # Plot histograms
    ax1.hist(base_cosim_flat, bins=50, alpha=0.7, color="blue")
    ax1.set_title("Base Decoder Cosine Similarity")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Frequency")

    ax2.hist(chat_cosim_flat, bins=50, alpha=0.7, color="green")
    ax2.set_title("Chat Decoder Cosine Similarity")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


cosim_hist(base_cosim, chat_cosim)
# %%
max_cosim_base = chunked_max_cosim(
    model1.decoder.weight[0], model2.decoder.weight[0], chunk_size=8
)
max_cosim_chat = chunked_max_cosim(
    model1.decoder.weight[1], model2.decoder.weight[1], chunk_size=8
)
cosim_hist(max_cosim_base, max_cosim_chat)
# %%
from tools.cc_utils import load_latent_df
import matplotlib.pyplot as plt
import numpy as np

df1 = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss")
df2 = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-SAELoss")
df3 = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-Decoupled")
co_idx1 = set(df1.query("tag == 'Chat only'").index)
co_idx2 = set(df2.query("tag == 'Chat only'").index)
co_idx3 = set(df3.query("tag == 'Chat only'").index)

# Prepare pairwise intersection counts for heatmap
labels = ["CCLoss", "SAELoss", "Decoupled"]
sets = [co_idx1, co_idx2, co_idx3]
n = len(sets)
heatmap = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        if i == j:
            heatmap[i, j] = len(sets[i])
        else:
            heatmap[i, j] = len(sets[i] & sets[j])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(heatmap, cmap="Blues")

# Show all ticks and label them
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Rotate the tick labels and set alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(n):
    for j in range(n):
        ax.text(j, i, heatmap[i, j], ha="center", va="center", color="black")

ax.set_title("Chat-only Latent Overlap Heatmap")
fig.tight_layout()
plt.savefig("results/num_chat_only_overlap.png", dpi=300)
plt.show()

# Optionally, print the triple overlap as well
print(f"num chat only in all: {len(co_idx1 & co_idx2 & co_idx3)}")
# %%
