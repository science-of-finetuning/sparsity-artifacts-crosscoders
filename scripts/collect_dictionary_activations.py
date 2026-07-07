import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import torch as th
from loguru import logger
from tqdm.auto import trange
from argparse import ArgumentParser
from tools.utils import load_activation_dataset, load_dictionary_model
from transformers import AutoTokenizer
from tools.cache_utils import DifferenceCache, LatentActivationCache
from huggingface_hub import repo_exists

from tools.configs import DATA_ROOT
from tools.configs import HF_NAME


@th.no_grad()
def get_positive_activations_incremental(tokens, boundaries, dataset, cc, latent_ids, checkpoint_every_n_seqs=5000, temp_dir=None, dataset_name=""):
    """
    Extract positive activations and save to temp directory.
    
    Args:
        tokens: All tokens
        boundaries: List of sequence boundary indices
        dataset: Dataset containing activations
        cc: Object with get_activations method
        latent_ids: Tensor of latent indices to extract
        checkpoint_every_n_seqs: Save checkpoint every N sequences (default 5000)
        temp_dir: Directory to save temporary results
        dataset_name: Name prefix for files (e.g., "fineweb" or "lmsys")

    Returns:
        Path to the saved results directory
    """
    out_activations = []
    out_ids = []
    seq_ranges = [0]
    sequences = []
    
    num_sequences = len(boundaries) - 1
    
    # Initialize tensors to track max activations for each latent
    max_activations = th.zeros(len(latent_ids), device="cuda")
    
    # Check for existing checkpoint
    start_seq_idx = 0
    checkpoint_file = None
    if temp_dir is not None:
        checkpoint_file = temp_dir / f"checkpoint_{dataset_name}.pt"
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = th.load(checkpoint_file, weights_only=True)
            out_activations = checkpoint["activations"]
            out_ids = checkpoint["ids"]
            seq_ranges = checkpoint["seq_ranges"]
            sequences = checkpoint["sequences"]
            max_activations = checkpoint["max_activations"].cuda()
            start_seq_idx = checkpoint["last_seq_idx"] + 1
            logger.info(f"Resuming from sequence {start_seq_idx}/{num_sequences}")

    for seq_idx in trange(start_seq_idx, num_sequences, desc=f"Processing {dataset_name} sequences"):
        start_idx = boundaries[seq_idx]
        end_idx = boundaries[seq_idx + 1]
        
        # Get the sequence tokens
        sequence = tokens[start_idx:end_idx]
        sequences.append(sequence)
        
        # Get activations for this sequence
        activations = th.stack(
            [dataset[j].cuda() for j in range(start_idx, end_idx)]
        )
        feature_activations = cc.get_activations(activations)
        assert feature_activations.shape == (
            len(activations),
            len(latent_ids),
        ), f"Feature activations shape: {feature_activations.shape}, expected: {(len(activations), len(latent_ids))}"

        # Track maximum activations
        seq_max_values, _ = feature_activations.max(dim=0)
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

        # Move to CPU immediately to free GPU memory
        out_activations.append(pos_activations.cpu())
        out_ids.append(pos_ids.cpu())
        seq_ranges.append(seq_ranges[-1] + len(pos_ids))
        
        # Clean up GPU memory
        del activations, feature_activations, pos_mask, pos_indices, pos_activations, seq_idx_tensor, pos_ids
        
        # Save checkpoint periodically
        if checkpoint_file is not None and (seq_idx + 1) % checkpoint_every_n_seqs == 0:
            logger.info(f"Saving checkpoint at sequence {seq_idx + 1}/{num_sequences}")
            checkpoint = {
                "activations": out_activations,
                "ids": out_ids,
                "seq_ranges": seq_ranges,
                "sequences": sequences,
                "max_activations": max_activations.cpu(),
                "last_seq_idx": seq_idx
            }
            th.save(checkpoint, checkpoint_file)
            th.cuda.empty_cache()

    # Final concatenation and save to temp
    logger.info(f"Finalizing {dataset_name} results...")
    out_activations = th.cat(out_activations) if out_activations else th.tensor([])
    out_ids = th.cat(out_ids) if out_ids else th.tensor([])
    
    # Save results to temp directory
    result_dir = temp_dir / dataset_name
    result_dir.mkdir(exist_ok=True)
    
    th.save(out_activations, result_dir / "out_acts.pt")
    th.save(out_ids, result_dir / "out_ids.pt") 
    th.save(sequences, result_dir / "sequences.pt")  # Keep original name since not padded yet
    th.save(th.tensor(seq_ranges), result_dir / "seq_ranges.pt")
    th.save(max_activations.cpu(), result_dir / "max_activations.pt")
    
    logger.info(f"Saved {dataset_name} results to {result_dir}")
    
    # Free memory but keep files
    del out_activations, out_ids, sequences
    th.cuda.empty_cache()
    
    return result_dir


def get_sequence_boundaries(tokenizer, tokens):
    """
    Get sequence boundaries without creating all sequences upfront.
    Returns indices of BOS tokens or fixed boundaries for efficient processing.
    """
    bos_mask = tokens == tokenizer.bos_token_id
    indices_of_bos = th.where(bos_mask)[0]
    
    if not bos_mask.any():
        # If no BOS tokens, create fixed-size sequence boundaries
        seq_len = 1024
        logger.warning(f"No BOS tokens found in data, using fixed-size sequences of {seq_len} tokens")
        boundaries = list(range(0, len(tokens), seq_len))
        boundaries.append(len(tokens))
        return boundaries, False
    else:
        # Return BOS token positions as boundaries
        boundaries = indices_of_bos.tolist()
        if len(tokens) not in boundaries:
            boundaries.append(len(tokens))
        return boundaries, True


def add_get_activations_sae(sae, model_idx, is_difference=False):
    """
    Add get_activations method to SAE model.

    Args:
        sae: The SAE model
        model_idx: Index of model to use (0 for base, 1 for chat) if not difference
        is_difference: If True, compute difference between chat and base activations
    """
    if is_difference:

        def get_activation(x: th.Tensor, select_features=None, **kwargs):
            # For difference SAEs, x should be the difference already computed by DifferenceCache
            # x shape: (batch_size, activation_dim)
            assert x.ndim == 2, f"Expected 2D tensor for difference SAE, got {x.ndim}D"
            f = sae.encode(x)
            if select_features is not None:
                f = f[:, select_features]
            return f

    else:

        def get_activation(x: th.Tensor, select_features=None, **kwargs):
            assert x.ndim == 3 and x.shape[1] == 2
            f = sae.encode(x[:, model_idx])
            if select_features is not None:
                f = f[:, select_features]
            return f

    sae.get_activations = get_activation
    return sae


def load_latent_activations(
    repo_id=f"{HF_NAME}/autointerp-data-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
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


def create_difference_cache(cache, sae_model_idx):
    if sae_model_idx == 0:
        return DifferenceCache(cache.activation_cache_1, cache.activation_cache_2)
    else:
        assert sae_model_idx == 1
        return DifferenceCache(cache.activation_cache_2, cache.activation_cache_1)


def collect_dictionary_activations(
    dictionary_model_name: str,
    activation_store_dir: str | Path = DATA_ROOT / "activations/",
    base_model: str = "google/gemma-2-2b",
    chat_model: str = "google/gemma-2-2b-it",
    layer: int = 13,
    latent_ids: th.Tensor | None = None,
    latent_activations_dir: str | Path = DATA_ROOT / "latent_activations/",
    upload_to_hub: bool = False,
    split: str = "validation",
    load_from_disk: bool = False,
    lmsys_col: str = "",
    is_sae: bool = False,
    is_difference_sae: bool = False,
    sae_model_idx: int | None = None,
    cache_suffix: str = "",
    checkpoint_every_n_seqs: int = 5000,
) -> None:
    """
    Compute and save latent activations for a given dictionary model.

    This function processes activations from specified datasets (e.g., FineWeb and LMSYS),
    applies the provided dictionary model to compute latent activations, and saves the results
    to disk. Optionally, it can upload the computed activations to the Hugging Face Hub.

    Args:
        dictionary_model (str): Path or identifier for the dictionary (crosscoder) model to use.
        activation_store_dir (str, optional): Directory containing the raw activation datasets.
            Defaults to $DATASTORE/activations/.
        base_model (str, optional): Name or path of the base model (e.g., "google/gemma-2-2b").
            Defaults to "google/gemma-2-2b".
        chat_model (str, optional): Name or path of the chat/instruct model.
            Defaults to "google/gemma-2-2b-it".
        layer (int, optional): The layer index from which to extract activations.
            Defaults to 13.
        latent_ids (th.Tensor or None, optional): Tensor of latent indices to compute activations for.
            If None, uses all latents in the dictionary model.
        latent_activations_dir (str, optional): Directory to save computed latent activations.
            Defaults to $DATASTORE/latent_activations/.
        upload_to_hub (bool, optional): Whether to upload the computed activations to the Hugging Face Hub.
            Defaults to False.
        split (str, optional): Dataset split to use (e.g., "validation").
            Defaults to "validation".
        load_from_disk (bool, optional): If True, load precomputed activations from disk instead of recomputing.
            Defaults to False.
        is_sae (bool, optional): Whether the model is an SAE rather than a crosscoder.
            Defaults to False.
        is_difference_sae (bool, optional): Whether the SAE is trained on activation differences.
            Defaults to False.
        checkpoint_every_n_seqs (int, optional): Save checkpoint and clear GPU memory every N sequences.
            Helps prevent OOM errors on large datasets. Defaults to 5000.

    Returns:
        None
    """
    is_sae = is_sae or is_difference_sae
    if is_sae and sae_model_idx is None:
        raise ValueError(
            "sae_model_idx must be provided if is_sae is True. This is the index of the model activations to use for the SAE."
        )

    # Handle case where dictionary_model_name is a file path
    if "/" in dictionary_model_name and dictionary_model_name.endswith(".pt"):
        # Extract meaningful name from path like /path/to/model-name/model_final.pt
        model_dir_name = Path(dictionary_model_name).parent.name
        out_dir = Path(latent_activations_dir) / model_dir_name
    else:
        out_dir = Path(latent_activations_dir) / dictionary_model_name
    if cache_suffix:
        out_dir = out_dir / cache_suffix
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

        # For difference SAEs, convert to DifferenceCache
        if is_difference_sae:
            fineweb_cache = create_difference_cache(fineweb_cache, sae_model_idx)
            lmsys_cache = create_difference_cache(lmsys_cache, sae_model_idx)
            tokens_fineweb = fineweb_cache.tokens[0]  # Both should have same tokens
            tokens_lmsys = lmsys_cache.tokens[0]
        else:
            tokens_fineweb = fineweb_cache.tokens[0]
            tokens_lmsys = lmsys_cache.tokens[0]

        # Load the dictionary model
        dictionary_model = load_dictionary_model(
            dictionary_model_name, is_sae=is_sae
        ).to("cuda")
        if is_sae:
            dictionary_model = add_get_activations_sae(
                dictionary_model,
                model_idx=sae_model_idx,
                is_difference=is_difference_sae,
            )
        if latent_ids is None:
            latent_ids = th.arange(dictionary_model.dict_size)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Create temp directory
        temp_dir = out_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Get sequence boundaries without creating all sequences upfront
        boundaries_fineweb, has_bos_fineweb = get_sequence_boundaries(tokenizer, tokens_fineweb)
        boundaries_lmsys, has_bos_lmsys = get_sequence_boundaries(tokenizer, tokens_lmsys)

        print(
            f"Collecting activations for {len(boundaries_fineweb)-1} fineweb sequences and {len(boundaries_lmsys)-1} lmsys sequences"
        )

        # Process fineweb and save to temp
        fineweb_result_dir = get_positive_activations_incremental(
            tokens_fineweb, boundaries_fineweb, fineweb_cache, dictionary_model, latent_ids,
            checkpoint_every_n_seqs=checkpoint_every_n_seqs,
            temp_dir=temp_dir,
            dataset_name="fineweb"
        )
        
        # Process lmsys and save to temp
        lmsys_result_dir = get_positive_activations_incremental(
            tokens_lmsys, boundaries_lmsys, lmsys_cache, dictionary_model, latent_ids,
            checkpoint_every_n_seqs=checkpoint_every_n_seqs,
            temp_dir=temp_dir,
            dataset_name="lmsys"
        )
        
        # Load results from temp
        logger.info("Loading results from temp...")
        out_acts_fineweb = th.load(fineweb_result_dir / "out_acts.pt", weights_only=True)
        out_ids_fineweb = th.load(fineweb_result_dir / "out_ids.pt", weights_only=True)
        seq_fineweb = th.load(fineweb_result_dir / "sequences.pt", weights_only=True)
        seq_ranges_fineweb = th.load(fineweb_result_dir / "seq_ranges.pt", weights_only=True).tolist()
        max_activations_fineweb = th.load(fineweb_result_dir / "max_activations.pt", weights_only=True)
        
        out_acts_lmsys = th.load(lmsys_result_dir / "out_acts.pt", weights_only=True)
        out_ids_lmsys = th.load(lmsys_result_dir / "out_ids.pt", weights_only=True)
        seq_lmsys = th.load(lmsys_result_dir / "sequences.pt", weights_only=True)
        seq_ranges_lmsys = th.load(lmsys_result_dir / "seq_ranges.pt", weights_only=True).tolist()
        max_activations_lmsys = th.load(lmsys_result_dir / "max_activations.pt", weights_only=True)

        # Combine datasets for the merged output
        out_acts = th.cat([out_acts_fineweb, out_acts_lmsys])
        # add offset to seq_idx in out_ids_lmsys
        out_ids_lmsys_combined = out_ids_lmsys.clone()
        out_ids_lmsys_combined[:, 0] += len(seq_fineweb)
        out_ids = th.cat([out_ids_fineweb, out_ids_lmsys_combined])

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

        # Save combined tensors
        print("Saving combined dataset results...")
        th.save(out_acts.cpu(), out_dir / "out_acts.pt")
        th.save(out_ids.cpu(), out_dir / "out_ids.pt")
        th.save(padded_tensor.cpu(), out_dir / "padded_sequences.pt")
        th.save(latent_ids.cpu(), out_dir / "latent_ids.pt")
        th.save(seq_ranges.cpu(), out_dir / "seq_ranges.pt")
        th.save(seq_lengths.cpu(), out_dir / "seq_lengths.pt")
        th.save(combined_max_activations.cpu(), out_dir / "max_activations.pt")
        print(f"  Saved combined results to {out_dir}")

        # Print some stats about max activations
        print("\nMaximum activation statistics (combined):")
        print(f"  Average: {combined_max_activations.mean().item():.4f}")
        print(f"  Maximum: {combined_max_activations.max().item():.4f}")
        print(f"  Minimum: {combined_max_activations.min().item():.4f}")
        
        # Keep temp directory - it has the individual dataset results
        logger.info(f"Individual dataset results are in: {temp_dir}")

    if upload_to_hub:
        # Initialize Hugging Face API
        from huggingface_hub import HfApi

        api = HfApi()

        # Define repository ID for the dataset
        repo_id = f"{HF_NAME}/latent-activations-{dictionary_model_name}"
        # Check if repository exists, create it if it doesn't
        if repo_exists(repo_id=repo_id, repo_type="dataset"):
            print(f"Repository {repo_id} already exists")
        else:
            # Repository doesn't exist, create it
            print(f"Repository {repo_id}, creating it...")
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
            print(f"Created repository {repo_id}")

        # Upload all tensors to HF Hub directly from saved files
        def hf_path(name: str):
            if cache_suffix:
                return f"{cache_suffix}/{name}"
            else:
                return name

        api.upload_file(
            path_or_fileobj=str(out_dir / "out_acts.pt"),
            path_in_repo=hf_path("activations.pt"),
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "out_ids.pt"),
            path_in_repo=hf_path("indices.pt"),
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "padded_sequences.pt"),
            path_in_repo=hf_path("sequences.pt"),
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "latent_ids.pt"),
            path_in_repo=hf_path("latent_ids.pt"),
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Upload max activations and indices
        api.upload_file(
            path_or_fileobj=str(out_dir / "max_activations.pt"),
            path_in_repo=hf_path("max_activations.pt"),
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"All files uploaded to Hugging Face Hub at {repo_id}")
    else:
        print("Skipping upload to Hugging Face Hub")
    return LatentActivationCache(out_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute positive and maximum activations for latent features"
    )
    parser.add_argument(
        "--activation-store-dir", type=str, default=DATA_ROOT / "activations/"
    )
    parser.add_argument(
        "--indices-root", type=str, default=DATA_ROOT / "latent_indices/"
    )
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("dictionary_model", type=str)
    parser.add_argument("--target-set", type=str, nargs="+", default=[])
    parser.add_argument(
        "--latent-activations-dir",
        type=str,
        default=DATA_ROOT / "latent_activations/",
    )
    parser.add_argument("--upload-to-hub", action="store_true")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument(
        "--load-from-disk",
        action="store_true",
        help="Load precomputed activations from disk instead of recomputing. Useful if you forgot to upload to hub in previous run.",
    )
    parser.add_argument(
        "--is-sae", action="store_true"
    )  # TODO: allow base sae by changing the model_idx in add_get_activations_sae
    parser.add_argument("--is-difference-sae", action="store_true")
    parser.add_argument("--sae-model-idx", type=int, default=None)
    parser.add_argument("--cache-suffix", type=str, default="")
    args = parser.parse_args()
    if args.is_sae or args.is_difference_sae:
        if args.sae_model_idx is None:
            raise ValueError(
                "sae_model_idx must be provided if is_sae or is_difference_sae is True. This is the index of the model activations to use for the SAE. 0 for base, 1 for chat."
            )
    indices_root = Path(args.indices_root)
    if len(args.target_set) == 0:
        latent_ids = None
    else:
        indices = []
        for target_set in args.target_set:
            indices.append(
                th.load(indices_root / f"{target_set}.pt", weights_only=True)
            )
        latent_ids = th.cat(indices)

    # Set default checkpoint interval - can be modified here if needed
    checkpoint_every_n_seqs = 5000
    
    collect_dictionary_activations(
        dictionary_model_name=args.dictionary_model,
        activation_store_dir=args.activation_store_dir,
        base_model=args.base_model,
        chat_model=args.chat_model,
        layer=args.layer,
        latent_ids=latent_ids,
        latent_activations_dir=args.latent_activations_dir,
        upload_to_hub=args.upload_to_hub,
        split=args.split,
        load_from_disk=args.load_from_disk,
        is_sae=args.is_sae,
        is_difference_sae=args.is_difference_sae,
        sae_model_idx=args.sae_model_idx,
        cache_suffix=args.cache_suffix,
        checkpoint_every_n_seqs=checkpoint_every_n_seqs,
    )
