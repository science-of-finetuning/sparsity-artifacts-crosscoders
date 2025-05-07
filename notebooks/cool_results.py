# %%
import sys

sys.path.append("..")
from tools.utils import get_available_models, load_dictionary_model, chunked_max_cosim
from torch.nn.functional import cosine_similarity
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
# %%
import matplotlib.pyplot as plt
import numpy as np


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

df1 = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss")
df2 = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-SAELoss")
co_idx1 = set(df1.query("tag == 'Chat only'").index)
co_idx2 = set(df2.query("tag == 'Chat only'").index)

print(f"num chat only in CCLoss: {len(co_idx1)}")
print(f"num chat only in SAELoss: {len(co_idx2)}")
print(f"num chat only in both: {len(co_idx1 & co_idx2)}")
# %%
