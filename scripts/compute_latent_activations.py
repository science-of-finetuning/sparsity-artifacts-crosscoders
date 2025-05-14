import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import torch as th
from tqdm.auto import trange
from argparse import ArgumentParser
from tools.utils import load_activation_dataset, load_dictionary_model
from transformers import AutoTokenizer
from CONFIG import HF_NAME


@th.no_grad()
def get_positive_activations(sequences, ranges, dataset, cc, latent_ids):
    """
    Extract positive activations and their indices from sequences.
    Also compute the maximum activation for each latent feature.

    Args:
        sequences: List of sequences
        ranges: List of (start_idx, end_idx) tuples for each sequence
        dataset: Dataset containing activations
        cc: Object with get_activations method
        latent_ids: Tensor of latent indices to extract

    Returns:
        Tuple of:
        - activations tensor: positive activation values
        - indices tensor: in (seq_idx, seq_pos, feature_pos) format
        - max_activations: maximum activation value for each latent feature
    """
    out_activations = []
    out_ids = []
    seq_ranges = [0]

    # Initialize tensors to track max activations for each latent
    max_activations = th.zeros(len(latent_ids), device="cuda")
    # print(f"Computing {len(latent_ids)} latent activations for {len(sequences)} sequences and ranges {ranges}")

    for seq_idx in trange(len(sequences)):
        activations = th.stack(
            [dataset[j].cuda() for j in range(ranges[seq_idx][0], ranges[seq_idx][1])]
        )
        feature_activations = cc.get_activations(activations)
        assert feature_activations.shape == (
            len(activations),
            len(latent_ids),
        ), f"Feature activations shape: {feature_activations.shape}, expected: {(len(activations), len(latent_ids))}"

        # Track maximum activations
        # For each latent feature, find the max activation in this sequence
        seq_max_values, seq_max_positions = feature_activations.max(dim=0)

        # Update global maximums where this sequence has a higher value
        update_mask = seq_max_values > max_activations
        max_activations[update_mask] = seq_max_values[update_mask]

        # Get indices where feature activations are positive
        pos_mask = feature_activations > 0
        pos_indices = th.nonzero(pos_mask, as_tuple=True)

        # Get the positive activation values
        pos_activations = feature_activations[pos_mask]

        # Create sequence indices tensor matching size of positive indices
        seq_idx_tensor = th.full_like(pos_indices[0], seq_idx)

        # Stack indices into (seq_idx, seq_pos, feature_pos) format
        pos_ids = th.stack([seq_idx_tensor, pos_indices[0], pos_indices[1]], dim=1)

        out_activations.append(pos_activations)
        out_ids.append(pos_ids)
        seq_ranges.append(seq_ranges[-1] + len(pos_ids))

    out_activations = th.cat(out_activations).cpu()
    out_ids = th.cat(out_ids).cpu()
    return out_activations, out_ids, seq_ranges, max_activations


def split_into_sequences(tokenizer, tokens):
    # Find indices of BOS tokens
    indices_of_bos = th.where(tokens == tokenizer.bos_token_id)[0]

    # Split tokens into sequences starting with BOS token
    sequences = []
    index_to_seq_pos = []  # List of (sequence_idx, idx_in_sequence) tuples
    ranges = []
    for i in trange(len(indices_of_bos)):
        start_idx = indices_of_bos[i]
        end_idx = indices_of_bos[i + 1] if i < len(indices_of_bos) - 1 else len(tokens)
        sequence = tokens[start_idx:end_idx]
        sequences.append(sequence)
        ranges.append((start_idx, end_idx))
        # Add mapping for each token in this sequence
        for j in range(len(sequence)):
            index_to_seq_pos.append((i, j))

    return sequences, index_to_seq_pos, ranges


def load_latent_activations(
    repo_id="science-of-finetuning/autointerp-data-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
):
    """
    Load the autointerp data from Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face Hub repository ID containing the data

    Returns:
        tuple: (activations, indices, sequences) tensors where:
            - activations: tensor of shape [n_total_activations] containing latent activations
            - indices: tensor of shape [n_total_activations, 3] containing (seq_idx, seq_pos, latent_idx)
            - sequences: tensor of shape [n_total_sequences, max_seq_len] containing the padded input sequences (right padded)
    """
    import torch
    from huggingface_hub import hf_hub_download

    # Download files from hub
    activations_path = hf_hub_download(
        repo_id=repo_id, filename="activations.pt", repo_type="dataset"
    )
    indices_path = hf_hub_download(
        repo_id=repo_id, filename="indices.pt", repo_type="dataset"
    )
    sequences_path = hf_hub_download(
        repo_id=repo_id, filename="sequences.pt", repo_type="dataset"
    )
    latent_ids_path = hf_hub_download(
        repo_id=repo_id, filename="latent_ids.pt", repo_type="dataset"
    )

    # Load tensors
    activations = torch.load(activations_path, weights_only=False)
    indices = torch.load(indices_path, weights_only=False)
    sequences = torch.load(sequences_path, weights_only=False)
    latent_ids = torch.load(latent_ids_path, weights_only=False)

    return activations, indices, sequences, latent_ids


def compute_latent_activations(
    dictionary_model: str,
    activation_store_dir: str = "/workspace/data/activations/",
    base_model: str = "google/gemma-2-2b",
    chat_model: str = "google/gemma-2-2b-it",
    layer: int = 13,
    latent_ids: th.Tensor | None = None,
    latent_activations_dir: str = "/workspace/data/latent_activations/",
    upload_to_hub: bool = False,
    split: str = "validation",
    load_from_disk: bool = False,
    lmsys_col: str = "",
) -> None:
    """
    Compute and save latent activations for a given dictionary model.

    This function processes activations from specified datasets (e.g., FineWeb and LMSYS),
    applies the provided dictionary model to compute latent activations, and saves the results
    to disk. Optionally, it can upload the computed activations to the Hugging Face Hub.

    Args:
        dictionary_model (str): Path or identifier for the dictionary (crosscoder) model to use.
        activation_store_dir (str, optional): Directory containing the raw activation datasets.
            Defaults to "/workspace/data/activations/".
        base_model (str, optional): Name or path of the base model (e.g., "google/gemma-2-2b").
            Defaults to "google/gemma-2-2b".
        chat_model (str, optional): Name or path of the chat/instruct model.
            Defaults to "google/gemma-2-2b-it".
        layer (int, optional): The layer index from which to extract activations.
            Defaults to 13.
        latent_ids (th.Tensor or None, optional): Tensor of latent indices to compute activations for.
            If None, uses all latents in the dictionary model.
        latent_activations_dir (str, optional): Directory to save computed latent activations.
            Defaults to "/workspace/data/latent_activations/".
        upload_to_hub (bool, optional): Whether to upload the computed activations to the Hugging Face Hub.
            Defaults to False.
        split (str, optional): Dataset split to use (e.g., "validation").
            Defaults to "validation".
        load_from_disk (bool, optional): If True, load precomputed activations from disk instead of recomputing.
            Defaults to False.

    Returns:
        None
    """
    out_dir = Path(latent_activations_dir) / f"{dictionary_model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the activation dataset
    if not load_from_disk:
        fineweb_cache, lmsys_cache = load_activation_dataset(
            activation_store_dir=activation_store_dir,
            base_model=base_model.split("/")[-1],
            instruct_model=chat_model.split("/")[-1],
            layer=layer,
            lmsys_split=split + f"-col{lmsys_col}" if lmsys_col else split,
            split=split,
        )
        tokens_fineweb = fineweb_cache.tokens[0]
        tokens_lmsys = lmsys_cache.tokens[0]

        # Load the dictionary model
        dictionary_model = load_dictionary_model(dictionary_model).to("cuda")
        if latent_ids is None:
            latent_ids = th.arange(dictionary_model.dict_size)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        seq_lmsys, idx_to_seq_pos_lmsys, ranges_lmsys = split_into_sequences(
            tokenizer, tokens_lmsys
        )
        seq_fineweb, idx_to_seq_pos_fineweb, ranges_fineweb = split_into_sequences(
            tokenizer, tokens_fineweb
        )

        (
            out_acts_fineweb,
            out_ids_fineweb,
            seq_ranges_fineweb,
            max_activations_fineweb,
        ) = get_positive_activations(
            seq_fineweb, ranges_fineweb, fineweb_cache, dictionary_model, latent_ids
        )
        out_acts_lmsys, out_ids_lmsys, seq_ranges_lmsys, max_activations_lmsys = (
            get_positive_activations(
                seq_lmsys, ranges_lmsys, lmsys_cache, dictionary_model, latent_ids
            )
        )

        out_acts = th.cat([out_acts_fineweb, out_acts_lmsys])
        # add offset to seq_idx in out_ids_lmsys
        out_ids_lmsys[:, 0] += len(seq_fineweb)
        out_ids = th.cat([out_ids_fineweb, out_ids_lmsys])

        seq_ranges_lmsys = [i + len(out_acts_fineweb) for i in seq_ranges_lmsys]
        seq_ranges = th.cat(
            [th.tensor(seq_ranges_fineweb[:-1]), th.tensor(seq_ranges_lmsys)]
        )

        # Combine max activations, taking the maximum between both datasets
        combined_max_activations = th.maximum(
            max_activations_fineweb, max_activations_lmsys
        )

        sequences_all = seq_fineweb + seq_lmsys

        # Find max length
        max_len = max(len(s) for s in sequences_all)
        seq_lengths = th.tensor([len(s) for s in sequences_all])
        # Pad each sequence to max length
        padded_seqs = [
            th.cat(
                [
                    s,
                    th.full(
                        (max_len - len(s),), tokenizer.pad_token_id, device=s.device
                    ),
                ]
            )
            for s in sequences_all
        ]
        # Convert to tensor and save
        padded_tensor = th.stack(padded_seqs)

        # Save tensors
        th.save(out_acts.cpu(), out_dir / "out_acts.pt")
        th.save(out_ids.cpu(), out_dir / "out_ids.pt")
        th.save(padded_tensor.cpu(), out_dir / "padded_sequences.pt")
        th.save(latent_ids.cpu(), out_dir / "latent_ids.pt")
        th.save(seq_ranges.cpu(), out_dir / "seq_ranges.pt")
        th.save(seq_lengths.cpu(), out_dir / "seq_lengths.pt")
        th.save(combined_max_activations.cpu(), out_dir / "max_activations.pt")

        # Print some stats about max activations
        print("Maximum activation statistics:")
        print(f"  Average: {combined_max_activations.mean().item():.4f}")
        print(f"  Maximum: {combined_max_activations.max().item():.4f}")
        print(f"  Minimum: {combined_max_activations.min().item():.4f}")

    if upload_to_hub:
        # Initialize Hugging Face API
        from huggingface_hub import HfApi

        api = HfApi()

        # Define repository ID for the dataset
        repo_id = f"{HF_NAME}/latent-activations-{dictionary_model}"
        # Check if repository exists, create it if it doesn't
        try:
            # Try to get repository info to check if it exists
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            print(f"Repository {repo_id} already exists")
        except Exception as e:
            # Repository doesn't exist, create it
            print(f"Repository {repo_id} doesn't exist (error: {e}), creating it...")
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
            print(f"Created repository {repo_id}")

        # Upload all tensors to HF Hub directly from saved files
        api.upload_file(
            path_or_fileobj=str(out_dir / "out_acts.pt"),
            path_in_repo="activations.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "out_ids.pt"),
            path_in_repo="indices.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "padded_sequences.pt"),
            path_in_repo="sequences.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "latent_ids.pt"),
            path_in_repo="latent_ids.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Upload max activations and indices
        api.upload_file(
            path_or_fileobj=str(out_dir / "max_activations.pt"),
            path_in_repo="max_activations.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"All files uploaded to Hugging Face Hub at {repo_id}")
    else:
        print("Skipping upload to Hugging Face Hub")

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute positive and maximum activations for latent features"
    )
    parser.add_argument(
        "--activation-store-dir", type=str, default="/workspace/data/activations/"
    )
    parser.add_argument(
        "--indices-root", type=str, default="/workspace/data/latent_indices/"
    )
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--dictionary-model", type=str, required=True)
    parser.add_argument("--target-set", type=str, nargs="+", default=[])
    parser.add_argument(
        "--latent-activations-dir",
        type=str,
        default="/workspace/data/latent_activations/",
    )
    parser.add_argument("--upload-to-hub", action="store_true")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--load-from-disk", action="store_true")
    args = parser.parse_args()

    compute_latent_activations(
        dictionary_model=args.dictionary_model,
        activation_store_dir=args.activation_store_dir,
        indices_root=args.indices_root,
        base_model=args.base_model,
        chat_model=args.chat_model,
        layer=args.layer,
        target_set=args.target_set,
        latent_activations_dir=args.latent_activations_dir,
        upload_to_hub=args.upload_to_hub,
        split=args.split,
        load_from_disk=args.load_from_disk,
    )

