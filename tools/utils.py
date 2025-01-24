# %%
from huggingface_hub import hf_hub_download
import torch as th
import json
from dictionary_learning import CrossCoder


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
    state_dict = th.load(weights_path, map_location=cfg["device"])

    crosscoder = CrossCoder(
        activation_dim=cfg["d_in"],
        dict_size=cfg["dict_size"],
        num_layers=2,
        num_decoder_layers=2,
    )

    crosscoder.encoder.weight = th.nn.Parameter(state_dict["W_enc"].permute(1, 0, 2))
    crosscoder.encoder.bias = th.nn.Parameter(state_dict["b_enc"])
    crosscoder.decoder.weight = th.nn.Parameter(state_dict["W_dec"].permute(1, 0, 2))
    crosscoder.decoder.bias = th.nn.Parameter(state_dict["b_dec"])

    return crosscoder
