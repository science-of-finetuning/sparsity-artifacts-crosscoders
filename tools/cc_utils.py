import json
import warnings
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd
import numpy as np
from pandas.io.formats.printing import pprint_thing
import torch as th
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, hf_api

from dictionary_learning import CrossCoder
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from nnterp import load_model
from tiny_dashboard import OfflineFeatureCentricDashboard
from tiny_dashboard.dashboard_implementations import CrosscoderOnlineFeatureDashboard

dfs = defaultdict(lambda: None)
df_hf_repo = {
    "l13_crosscoder": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
    "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
    "connor": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-ckissane",
    "ckkissane/crosscoder-gemma-2-2b-model-diff": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-ckissane",
}


def load_latent_df(crosscoder=None):
    """Load the latent_df for the given crosscoder."""
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    if crosscoder in df_hf_repo:
        df_path = hf_hub_download(
            repo_id=df_hf_repo[crosscoder],
            filename="feature_df.csv",
            repo_type="dataset",
        )
    else:
        df_path = Path(crosscoder)
        if not df_path.exists():
            raise ValueError(f"Unknown crosscoder: {crosscoder}")
    df = pd.read_csv(df_path, index_col=0)
    return df


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
        confirm: if True, ask the user to confirm the push
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
        missing_columns = original_columns - new_columns
        added_columns = new_columns - original_columns
        shared_columns = original_columns & new_columns
        duplicated_columns = df.columns.duplicated()
        if duplicated_columns.any():
            raise ValueError(
                f"Duplicated columns in uploaded df: {df.columns[duplicated_columns]}"
            )
        if len(missing_columns) > 0:
            real_missing_columns = missing_columns - allow_remove_columns
            if len(real_missing_columns) > 0 and not force:
                raise ValueError(
                    f"Missing columns in uploaded df: {missing_columns}\n"
                    "If you want to upload the df anyway, set allow_remove_columns=your_removed_columns"
                    " or force=True"
                )
            elif len(missing_columns) > 0 and len(real_missing_columns) == 0:
                print(f"Removed columns in uploaded df: {missing_columns}")
            else:
                warnings.warn(
                    f"Missing columns in uploaded df: {missing_columns}\n"
                    "Force=True -> Upload df anyway"
                )

        if len(added_columns) > 0 and not force:
            print(f"Added columns in uploaded df: {added_columns}")

        for column in shared_columns:
            if original_df[column].dtype != df[column].dtype:
                warnings.warn(
                    f"Column {column} has different dtype in original and new df"
                )
            # diff the columns
            if "float" in str(original_df[column].dtype):
                equal = np.allclose(
                    original_df[column].values, df[column].values, equal_nan=True
                )
            else:
                equal = original_df[column].equals(df[column])
            if not equal:
                print(f"Column {column} has different values in original and new df:")
                if "float" in str(original_df[column].dtype):
                    diff_ratio = (
                        ~np.isclose(
                            original_df[column].values,
                            df[column].values,
                            equal_nan=True,
                        )
                    ).mean() * 100
                else:
                    diff_ratio = (original_df[column] != df[column]).mean() * 100
                print(f"% of different values: {diff_ratio:.2f}%")

                print(f"Original: {pprint_thing(original_df[column].values)}")
                print(f"New     : {pprint_thing(df[column].values)}")
                print("=" * 20 + "\n", flush=True)
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


def _latent_df(crosscoder=None):
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    if dfs[crosscoder] is None:
        dfs[crosscoder] = load_latent_df(crosscoder)
    return dfs[crosscoder]


def base_only_latent_indices(crosscoder=None):
    """Return the indices of the base only latents of the given crosscoder."""
    df = _latent_df(crosscoder)
    # filter for tag = Base only
    return th.tensor(df[df["tag"] == "Base only"].index.tolist())


def chat_only_latent_indices(crosscoder=None):
    """Return the indices of the chat only latents of the given crosscoder."""
    df = _latent_df(crosscoder)
    # filter for tag = Chat only
    return th.tensor(
        df[(df["tag"] == "Chat only") | (df["tag"] == "IT only")].index.tolist()
    )


def shared_latent_indices(crosscoder=None):
    """Return the indices of the shared latents of the given crosscoder."""
    df = _latent_df(crosscoder)
    # filter for tag = Shared
    return th.tensor(df[df["tag"] == "Shared"].index.tolist())


class CCLatent:  # pylint: disable=E1101
    """
    A class for a latent in a crosscoder.

    Args:
        id_: the index of the latent
        crosscoder: the crosscoder to use
    """

    def __init__(self, id_: int, crosscoder=None):
        self.id = id_
        self.row = _latent_df(crosscoder).loc[id_]
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
    if crosscoder == "l13_crosscoder" or crosscoder == "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04":
        return CrossCoder.from_pretrained(
            "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04", from_hub=True
        )
    elif crosscoder == "connor" or crosscoder == "ckkissane/crosscoder-gemma-2-2b-model-diff":
        return load_connor_crosscoder()
    else:
        path = Path(crosscoder)
        if path.exists():
            return CrossCoder.from_pretrained(path)
        else:
            raise ValueError(f"Unknown crosscoder: {crosscoder}")

def load_dictionary_model(model_name: str):
    if "SAE-" in model_name:
        return BatchTopKSAE.from_pretrained(model_name)
    else:
        return load_crosscoder(model_name)

crosscoders = defaultdict(lambda: None)


def _crosscoder(crosscoder=None):
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    if crosscoders[crosscoder] is None:
        crosscoders[crosscoder] = load_crosscoder(crosscoder)
    return crosscoders[crosscoder]


def online_dashboard(
    crosscoder=None,
    max_acts=None,
    crosscoder_device="auto",
    base_device="auto",
    chat_device="auto",
    torch_dtype=th.bfloat16,
):
    """
    Instantiate an online dashboard for crosscoder latent analysis.

    Args:
        crosscoder: the crosscoder to use
        max_acts: a dictionary of max activations for each latent. If None, will be loaded from the latent_df of the crosscoder.
    """
    coder = _crosscoder(crosscoder)
    if crosscoder_device == "auto":
        crosscoder_device = "cuda:0" if th.cuda.is_available() else "cpu"
    coder = coder.to(crosscoder_device)
    if max_acts is None:
        df = _latent_df(crosscoder)
        max_acts_cols = ["max_act", "lmsys_max_act"]
        for col in max_acts_cols:
            if col in df.columns:
                max_acts = df[col].dropna().to_dict()
                break
    base_model = load_model(
        "google/gemma-2-2b",
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        device_map=base_device,
    )
    chat_model = load_model(
        "google/gemma-2-2b-it",
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        device_map=chat_device,
    )
    return CrosscoderOnlineFeatureDashboard(
        base_model,
        chat_model,
        coder,
        13,
        max_acts=max_acts,
        crosscoder_device=crosscoder_device,
    )


def load_max_activating_examples(
    crosscoder=None, act_type: Literal["chat", "base", "both"] = "chat"
):
    """
    Load the max activating examples for the given crosscoder and act_type.

    Args:
        crosscoder: the crosscoder to use
        act_type: the type of examples to load
    """
    match act_type:
        case "chat":
            return th.load(
                hf_hub_download(
                    repo_id=df_hf_repo[crosscoder],
                    filename="chat_examples.pt",
                    repo_type="dataset",
                )
            )
        case "base":
            return th.load(
                hf_hub_download(
                    repo_id=df_hf_repo[crosscoder],
                    filename="base_examples.pt",
                    repo_type="dataset",
                )
            )
        case "both":
            return th.load(
                hf_hub_download(
                    repo_id=df_hf_repo[crosscoder],
                    filename="chat_base_examples.pt",
                    repo_type="dataset",
                )
            )


def offline_dashboard(
    crosscoder=None,
    act_type: Literal["chat", "base", "both"] = "chat",
    max_num_examples=50,
):
    """
    Instantiate an offline dashboard for crosscoder latent analysis.

    Args:
        crosscoder: the crosscoder to use
        act_type: the type of examples to load
        max_num_examples: the maximum number of examples to load
    """
    max_acts_examples = load_max_activating_examples(crosscoder, act_type)
    return OfflineFeatureCentricDashboard(
        max_acts_examples,
        tokenizer=AutoTokenizer.from_pretrained("google/gemma-2-2b-it"),
        max_examples=max_num_examples,
    )
