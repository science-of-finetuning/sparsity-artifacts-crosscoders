import json
import warnings
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
import sqlite3
import sys

import pandas as pd
import numpy as np
from pandas.io.formats.printing import pprint_thing
import torch as th
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, hf_api, repo_exists, file_exists
from huggingface_hub import HfApi
from loguru import logger

from dictionary_learning.dictionary import BatchTopKCrossCoder, CrossCoder
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from nnterp import load_model
from tiny_dashboard import OfflineFeatureCentricDashboard
from tiny_dashboard.dashboard_implementations import CrosscoderOnlineFeatureDashboard
from .configs import REPO_ROOT

sys.path.append(str(REPO_ROOT))

from tools.configs import HF_NAME

df_hf_repo_legacy = {
    "l13_crosscoder": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
    "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
    "connor": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-ckissane",
    "ckkissane/crosscoder-gemma-2-2b-model-diff": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-ckissane",
    # "gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
    "science-of-finetuning/diffing-stats-gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04": "science-of-finetuning/max-activating-examples-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
}


def stats_repo_id(crosscoder, author=HF_NAME):
    return f"{author}/diffing-stats-{crosscoder}"


def latent_df_exists(crosscoder_or_path, author=HF_NAME, df_name="feature_df"):
    if crosscoder_or_path in df_hf_repo_legacy:
        return file_exists(
            repo_id=df_hf_repo_legacy[crosscoder_or_path],
            filename=f"{df_name}.csv",
            repo_type="dataset",
        )
    elif Path(crosscoder_or_path).exists():
        return True
    else:
        return file_exists(
            repo_id=stats_repo_id(crosscoder_or_path),
            filename=f"{df_name}.csv",
            repo_type="dataset",
        )


def load_latent_df(crosscoder_or_path, author=HF_NAME, df_name="feature_df"):
    """Load the latent_df for the given crosscoder."""
    if crosscoder_or_path in df_hf_repo_legacy:
        # LEGACY SUPPORT
        df_path = hf_hub_download(
            repo_id=df_hf_repo_legacy[crosscoder_or_path],
            filename=f"{df_name}.csv",
            repo_type="dataset",
        )
    elif Path(crosscoder_or_path).exists():
        # Local model
        df_path = Path(crosscoder_or_path)
    else:
        repo_id = stats_repo_id(crosscoder_or_path, author=author)
        if not repo_exists(repo_id=repo_id, repo_type="dataset"):
            raise ValueError(
                f"Repository {repo_id} does not exist, can't load latent_df"
            )
        if not file_exists(
            repo_id=repo_id, filename=f"{df_name}.csv", repo_type="dataset"
        ):
            raise ValueError(
                f"File {df_name}.csv does not exist in repository {repo_id}, can't load latent_df"
            )
        df_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{df_name}.csv",
            repo_type="dataset",
        )
    df = pd.read_csv(df_path, index_col=0)
    return df


def push_latent_df(
    df,
    crosscoder=None,
    force=False,
    allow_remove_columns=None,
    commit_message=None,
    confirm=True,
    create_repo_if_missing=False,
    author=HF_NAME,
    filename="feature_df",
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
        create_repo_if_missing: if True, create the repository if it doesn't exist
    """
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    if (not force or confirm) and latent_df_exists(
        crosscoder, df_name=filename, author=author
    ):
        original_df = load_latent_df(crosscoder, df_name=filename, author=author)
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
    if not repo_exists(repo_id=stats_repo_id(crosscoder), repo_type="dataset"):
        if not create_repo_if_missing:
            raise ValueError(
                f"Repository {stats_repo_id(crosscoder)} does not exist, can't push latent_df. P"
            )
        print("Will create a new repository.")

    if confirm:
        print(f"Commit message: {commit_message}")
        r = input("Would you like to push the df to the hub? y/(n)")
        if r != "y":
            raise ValueError("User cancelled")

    # Get the repository ID
    repo_id = (
        df_hf_repo_legacy.get(crosscoder) if hasattr(df_hf_repo_legacy, "get") else None
    )
    if repo_id is None:
        repo_id = stats_repo_id(crosscoder)

    with TemporaryDirectory() as tmpdir:
        df.to_csv(Path(tmpdir) / f"{filename}.csv")
        try:
            hf_api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=Path(tmpdir) / f"{filename}.csv",
                path_in_repo=f"{filename}.csv",
                repo_type="dataset",
                commit_message=commit_message,
            )
        except Exception as e:
            if not create_repo_if_missing:
                raise e

            print(f"Repository {repo_id} doesn't exist. Creating it...")

            # Create the repository
            hf_api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
            )

            # Try uploading again
            hf_api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=Path(tmpdir) / f"{filename}.csv",
                path_in_repo=f"{filename}.csv",
                repo_type="dataset",
                commit_message=commit_message or f"Initial upload for {crosscoder}",
            )
            print(f"Successfully created repository {repo_id} and uploaded data.")
    return repo_id


def model_path_to_name(model_path: Path):
    """Convert a model path to a name."""
    if str(model_path).endswith(".pt"):
        return model_path.parent.name
    else:
        return model_path.name


def push_dictionary_model(model_path: Path, author=HF_NAME):
    """Push a dictionary model to the Hugging Face Hub.

    Args:
        model_path: The path to the model to push
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)
    model_name = model_path_to_name(model_path)
    repo_id = f"{author}/{model_name}"
    model_dir = model_path.parent
    config_path = model_dir / "config.json"

    model = load_dictionary_model(model_path)
    # Upload files to the hub
    try:
        model.push_to_hub(repo_id)

        # Upload config
        hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=config_path,
            path_in_repo="trainer_config.json",
            repo_type="model",
            commit_message=f"Upload {model_name} dictionary model",
        )

        print(f"Successfully uploaded model to {repo_id}")
    except Exception as e:
        print(f"Error uploading model to hub: {e}")

        # Try creating the repository
        try:
            print(f"Repository {repo_id} doesn't exist. Creating it...")
            hf_api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
            )

            # Try uploading again
            model.push_to_hub(repo_id)

            hf_api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=config_path,
                path_in_repo="trainer_config.json",
                repo_type="model",
                commit_message=f"Initial upload of {model_name} dictionary model",
            )

            print(f"Successfully created repository {repo_id} and uploaded model.")
        except Exception as e2:
            print(f"Failed to create repository and upload model: {e2}")
            raise e2
    return repo_id


def base_only_latent_indices(crosscoder=None):
    """Return the indices of the base only latents of the given crosscoder."""
    df = load_latent_df(crosscoder)
    # filter for tag = Base only
    return th.tensor(df[df["tag"] == "Base only"].index.tolist())


def chat_only_latent_indices(crosscoder=None):
    """Return the indices of the chat only latents of the given crosscoder."""
    df = load_latent_df(crosscoder)
    # filter for tag = Chat only
    return th.tensor(
        df[(df["tag"] == "Chat only") | (df["tag"] == "IT only")].index.tolist()
    )


def shared_latent_indices(crosscoder=None):
    """Return the indices of the shared latents of the given crosscoder."""
    df = load_latent_df(crosscoder)
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
        self.row = load_latent_df(crosscoder).loc[id_]
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
    if (
        crosscoder == "l13_crosscoder"
        or crosscoder == "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04"
    ):
        return CrossCoder.from_pretrained(
            "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04", from_hub=True
        )
    elif (
        crosscoder == "connor"
        or crosscoder == "ckkissane/crosscoder-gemma-2-2b-model-diff"
    ):
        return load_connor_crosscoder()
    elif "-k" in crosscoder:
        return BatchTopKCrossCoder.from_pretrained(crosscoder)
    else:
        path = Path(crosscoder)
        if path.exists():
            return CrossCoder.from_pretrained(path)
        else:
            raise ValueError(f"Unknown crosscoder: {crosscoder}")


def load_dictionary_model(
    model_name: str | Path, is_sae: bool | None = None, author=HF_NAME
):
    """Load a dictionary model from a local path or HuggingFace Hub.

    Args:
        model_name: Name or path of the model to load

    Returns:
        The loaded dictionary model
    """
    # Check if it's a HuggingFace Hub model
    if "/" not in str(model_name) or not Path(model_name).exists():
        # Legacy model
        if str(model_name) in df_hf_repo_legacy:
            model_name = df_hf_repo_legacy[str(model_name)]
        else:
            model_name = str(model_name)
        if "/" not in str(model_name):
            model_id = f"{author}/{str(model_name)}"
        else:
            model_id = model_name
        # Download config to determine model type
        if file_exists(model_id, "trainer_config.json", repo_type="model"):
            config_path = hf_hub_download(
                repo_id=model_id, filename="trainer_config.json"
            )
            with open(config_path, "r") as f:
                config = json.load(f)["trainer"]

            # Determine model class based on config
            if "dict_class" in config and config["dict_class"] in [
                "BatchTopKSAE",
                "CrossCoder",
                "BatchTopKCrossCoder",
            ]:
                return eval(
                    f"{config['dict_class']}.from_pretrained(model_id, from_hub=True)"
                )
            else:
                raise ValueError(f"Unknown model type: {config['dict_class']}")
        else:
            logger.info(
                f"No config found for {model_id}, relying on is_sae={is_sae} arg to determine model type"
            )
            # If no model_type in config, try to infer from other fields
            if is_sae:
                return BatchTopKSAE.from_pretrained(model_id, from_hub=True)
            else:
                return CrossCoder.from_pretrained(model_id, from_hub=True)
    else:
        # Local model
        model_path = Path(model_name)
        if not model_path.exists():
            raise ValueError(f"Local model {model_name} does not exist")

        # Load the config
        with open(model_path.parent / "config.json", "r") as f:
            config = json.load(f)["trainer"]

        # Determine model class based on config
        if "dict_class" in config and config["dict_class"] in [
            "BatchTopKSAE",
            "CrossCoder",
            "BatchTopKCrossCoder",
        ]:
            return eval(f"{config['dict_class']}.from_pretrained(model_path)")
        else:
            raise ValueError(f"Unknown model type: {config['dict_class']}")


crosscoders = defaultdict(lambda: None)


def _crosscoder(crosscoder=None):
    if crosscoder is None:
        crosscoder = "l13_crosscoder"
    if crosscoders[crosscoder] is None:
        crosscoders[crosscoder] = load_crosscoder(crosscoder)
    return crosscoders[crosscoder]


class SAEAsCrosscoder:
    def __init__(self, sae, is_sae_diff=False, model_idx=1):
        self.sae = sae
        self.is_sae_diff = is_sae_diff
        self.model_idx = model_idx

    def get_activations(self, x, select_features=None):
        assert x.shape[1:] == (2, self.sae.activation_dim)
        x_sae = x[:, self.model_idx]
        if self.is_sae_diff:
            x_sae = x_sae - x[:, 1 - self.model_idx]
        activations = self.sae.encode(x_sae)
        assert activations.shape == (x.shape[0], self.sae.dict_size)
        if select_features is not None:
            activations = activations[:, select_features]
        return activations


def online_dashboard(
    crosscoder,
    layer,
    max_acts=None,
    crosscoder_device="auto",
    base_device="auto",
    chat_device="auto",
    base_model="google/gemma-2-2b",
    chat_model="google/gemma-2-2b-it",
    torch_dtype=th.bfloat16,
    is_sae=False,
    is_sae_diff=False,
    sae_model: Literal["chat", "base"] = "chat",
):
    """
    Instantiate an online dashboard for crosscoder latent analysis.

    Args:
        crosscoder: the crosscoder to use
        max_acts: a dictionary of max activations for each latent. If None, will be loaded from the latent_df of the crosscoder.
    """
    if is_sae or is_sae_diff:
        if sae_model == "chat":
            sae_model_idx = 1
        elif sae_model == "base":
            sae_model_idx = 0
        else:
            raise ValueError(f"Invalid sae_model: {sae_model}")
    coder = load_dictionary_model(crosscoder, is_sae=is_sae or is_sae_diff)
    if crosscoder_device == "auto":
        crosscoder_device = "cuda:0" if th.cuda.is_available() else "cpu"
    coder = coder.to(crosscoder_device)
    if is_sae or is_sae_diff:
        coder = SAEAsCrosscoder(coder, is_sae_diff=is_sae_diff, model_idx=sae_model_idx)
    if max_acts is None:
        df = load_latent_df(crosscoder)
        max_acts_cols = ["max_act", "lmsys_max_act", "max_act_val"]
        for col in max_acts_cols:
            if col in df.columns:
                max_acts = df[col].dropna().to_dict()
                break
    base_model = load_model(
        base_model,
        torch_dtype=torch_dtype,
        attn_implementation="eager" if "gemma" in base_model else None,
        device_map=base_device,
    )
    chat_model = load_model(
        chat_model,
        torch_dtype=torch_dtype,
        attn_implementation="eager" if "gemma" in chat_model else None,
        device_map=chat_device,
    )
    return CrosscoderOnlineFeatureDashboard(
        base_model,
        chat_model,
        coder,
        layer,
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
                    repo_id=df_hf_repo_legacy[crosscoder],
                    filename="chat_examples.pt",
                    repo_type="dataset",
                )
            )
        case "base":
            return th.load(
                hf_hub_download(
                    repo_id=df_hf_repo_legacy[crosscoder],
                    filename="base_examples.pt",
                    repo_type="dataset",
                )
            )
        case "both":
            return th.load(
                hf_hub_download(
                    repo_id=df_hf_repo_legacy[crosscoder],
                    filename="chat_base_examples.pt",
                    repo_type="dataset",
                )
            )


def legacy_offline_dashboard(
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


class QuantileExamplesDB:
    """A persistent, read-only dictionary-like interface for quantile examples database."""

    def __init__(self, db_path, tokenizer, max_example_per_quantile=20):
        """Initialize the database connection.

        Args:
            db_path: Path to the SQLite database
        """
        # Use URI format with read-only mode for better concurrent access
        self.db_path = f"file:{db_path}?mode=ro"
        # Create connection with URI mode enabled
        self.conn = sqlite3.connect(self.db_path, uri=True)

        # Cache the feature indices for faster access
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT feature_idx FROM quantile_examples")
        self._feature_indices = frozenset(row[0] for row in cursor.fetchall())

        cursor.execute("SELECT COUNT(DISTINCT quantile_idx) FROM quantile_examples")
        self.num_quantiles = cursor.fetchone()[0]
        cursor.close()
        self.max_example_per_quantile = max_example_per_quantile
        self.tokenizer = tokenizer

    def __getitem__(self, feature_idx):
        """Get examples for a specific feature index.

        Returns:
            List of tuples (max_activation_value, token_ids, activation_values)
        """
        return self.get_examples(feature_idx)

    def get_examples(self, feature_idx, quantile=None):
        """Get examples for a specific feature index.

        Args:
            feature_idx: The feature index to get examples for
            quantile: The quantile to get examples for (default is all)

        Returns:
            List of tuples (max_activation_value, token_ids, activation_values)
        """
        if feature_idx not in self._feature_indices:
            raise KeyError(f"Feature index {feature_idx} not found in database")

        cursor = self.conn.cursor()
        if quantile is None:
            quantile = list(range(self.num_quantiles))[::-1]
        if isinstance(quantile, list):
            res = []
            for q in quantile:
                res.extend(self.get_examples(feature_idx, q))
            return res
        else:
            # If quantile is specified as an integer, filter by that quantile
            cursor.execute(
                """
                SELECT q.activation, q.sequence_idx, s.token_ids, a.positions, a.activation_values
                FROM quantile_examples q
                JOIN sequences s ON q.sequence_idx = s.sequence_idx
                JOIN activation_details a ON q.feature_idx = a.feature_idx AND q.sequence_idx = a.sequence_idx
                WHERE q.feature_idx = ? AND q.quantile_idx = ?
                ORDER BY q.activation DESC
                """,
                (feature_idx, int(quantile)),
            )

        results = []
        fetch = (
            cursor.fetchmany(self.max_example_per_quantile)
            if self.max_example_per_quantile is not None
            else cursor.fetchall()
        )
        for (
            activation,
            sequence_idx,
            token_ids_blob,
            positions_blob,
            values_blob,
        ) in fetch:
            token_ids = np.frombuffer(token_ids_blob, dtype=np.int32).tolist()
            positions = np.frombuffer(positions_blob, dtype=np.int32).tolist()
            values = np.frombuffer(values_blob, dtype=np.float32).tolist()

            # Initialize activation values with zeros
            activation_values = [0.0] * len(token_ids)

            # Fill in the non-zero activations
            for pos, val in zip(positions, values):
                activation_values[pos] = val

            results.append(
                (
                    activation,
                    self.tokenizer.convert_ids_to_tokens(token_ids),
                    activation_values,
                )
            )

        cursor.close()
        return results

    def keys(self):
        """Get all feature indices in the database."""
        return self._feature_indices

    def __iter__(self):
        """Iterate over feature indices."""
        return iter(self._feature_indices)

    def __len__(self):
        """Get the number of unique features."""
        return len(self._feature_indices)

    def __contains__(self, feature_idx):
        """Check if a feature index exists in the database."""
        return feature_idx in self._feature_indices


def offline_dashboard(
    crosscoder,
    max_example_per_quantile=20,
    tokenizer=None,
    db_path=None,
    filename="examples",
):
    """
    Returns an offline_dashboard showing activations from different quantile

    Args:
      crosscoder: The crosscoder to take the max activating examples
      max_example_per_quantile: the maximimum number of examples per quantile
    """
    if db_path is None:
        db_path = hf_hub_download(
            repo_id=stats_repo_id(crosscoder),
            repo_type="dataset",
            filename=f"{filename}.db",
        )
    else:
        db_path = db_path / crosscoder / f"{filename}.db"
    if tokenizer is None:
        assert (
            "gemma-2" in crosscoder
        ), "Tokenizer must be provided for non-gemma models"
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    activation_examples = QuantileExamplesDB(
        db_path, tokenizer, max_example_per_quantile=max_example_per_quantile
    )
    dashboard = OfflineFeatureCentricDashboard(
        activation_examples,
        tokenizer,
        max_examples=activation_examples.num_quantiles * max_example_per_quantile,
    )
    dashboard.display()
    return dashboard


def get_available_models(author=HF_NAME):
    """Fetch CrossCoder models from Hugging Face"""
    try:
        # Initialize the Hugging Face API
        api = HfApi()

        # Get models from the science-of-finetuning organization
        models = api.list_models(author=author)

        # Filter for CrossCoder models (you may need to adjust this filter)
        crosscoder_models = [model.id.split("/")[-1] for model in models]

        return crosscoder_models
    except Exception as e:
        # If there's an error fetching models, return just the dummy option
        print(f"Error fetching CrossCoder models: {e}")
        return []
