"""
Analyze a given dictionary as shown in the paper.

1. Compute dictionary latent activations on validation set
2. Run eval_crosscoder notebook if dictionary is a crosscoder
"""

from run_notebook import run_notebook
import time
import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch as th
from transformers import AutoTokenizer

from tools.utils import dict_to_args, auto_device, load_hf_model
from tools.tokenization_utils import patch_tokenizer
from tools.cc_utils import (
    load_latent_df,
    push_latent_df,
    load_dictionary_model,
    push_dictionary_model,
    latent_df_exists,
)
from scripts import (
    collect_dictionary_activations,
    collect_activating_examples,
    compute_latent_stats,
    compute_scalers,
    kl_experiment,
    compute_latents_template_stats,
)
from tools.configs import MODEL_CONFIGS
from scripts.eval_betas import (
    make_beta_df,
    make_betas_plots,
    plot_beta_ratios_template_perc,
)
from tools.cache_utils import LatentActivationCache
from loguru import logger

from tools.configs import HF_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dictionary", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--no-upload", action="store_false", dest="upload_to_hub")
    parser.add_argument("--lmsys-col", type=str, default=None)
    parser.add_argument(
        "--kl-dataset",
        type=str,
        default="science-of-finetuning/lmsys-chat-1m-chat-formatted",
        help="Dataset to use for KL experiment",
    )
    parser.add_argument("--kl-dataset-col", type=str, default="conversation")
    parser.add_argument("--kl-dataset-split", type=str, default="validation")
    parser.add_argument(
        "--num-effective-chat-only-latents",
        type=int,
        default=3176,
        help="Amount of latents to consider as chat-only",
    )
    parser.add_argument(
        "--chat-model-idx",
        type=int,
        default=1,
        help="Index of the chat model in the stacked activation cache",
    )
    parser.add_argument("--skip-notebook", action="store_true")
    parser.add_argument("--skip-recon-scalars", action="store_true")
    parser.add_argument("--skip-error-scalars", action="store_true")
    parser.add_argument("--batch-size-kl", type=int, default=6)
    parser.add_argument("--skip-token-level-replacement", action="store_true")
    parser.add_argument("--num-samples-betas", type=int, default=50_000_000)
    parser.add_argument("--compute-latent-stats", action="store_true")
    parser.add_argument("--run-kl-experiment", action="store_true")
    parser.add_argument(
        "--is-difference-sae",
        action="store_true",
        help="Specify if the SAE is trained on activation differences",
    )
    parser.add_argument(
        "--is-sae", action="store_true", help="Specify if the model is a SAE"
    )
    parser.add_argument(
        "--sae-model",
        type=str,
        default=None,
        choices=["base", "chat"],
        help="Specify if the SAE is trained on base or chat activations",
    )
    args = parser.parse_args()
    print(args)
    is_sae = args.is_sae or args.is_difference_sae
    if is_sae and args.sae_model is None:
        raise ValueError("SAE model must be specified for SAEs. Got None.")

    if args.upload_to_hub:
        # Check if dictionary is a local path that exists
        dictionary_path = Path(args.dictionary)
        if dictionary_path.exists():
            logger.info(
                f"Found local dictionary at {dictionary_path}. Uploading first..."
            )
            if dictionary_path.is_file() and dictionary_path.suffix == ".pt":
                # Upload dictionary model
                repo_id = push_dictionary_model(dictionary_path)
                # Update args.dictionary to use the model name
                args.dictionary = repo_id.split("/")[-1]
                print(f"Using dictionary name: {args.dictionary}")
            else:
                raise ValueError(
                    f"Dictionary path must be a .pt file, got {dictionary_path}"
                )

    if args.chat_model_idx != 1:
        c = input(
            f"Chat model idx set to {args.chat_model_idx} != 1. Some of the analysis pipeline will not work as expected (e.g. kl experiment). Continue? y/(n)"
        )
        if c != "y":
            exit()

    latent_activations_dir = args.data_dir / "latent_activations"
    activation_store_dir = args.data_dir / "activations"
    dictionary = None
    if args.tokenizer is None:
        args.tokenizer = args.chat_model
    if not args.skip_notebook and not is_sae:
        run_notebook(
            notebook="eval_crosscoder",
            crosscoder=args.dictionary,
            extra_args=dict_to_args(upload=args.upload_to_hub, overwrite=True),
        )
    elif is_sae:
        dictionary = load_dictionary_model(args.dictionary, is_sae=True).to(
            auto_device()
        )
        df = pd.DataFrame(index=range(dictionary.dict_size))
        dec_norms = dictionary.decoder.weight.data.norm(dim=0).cpu().numpy()
        assert dec_norms.shape == (dictionary.dict_size,)
        df["dec_norm"] = dec_norms
        enc_norms = dictionary.encoder.weight.data.norm(dim=1).cpu().numpy()
        assert enc_norms.shape == (dictionary.dict_size,)
        df["enc_norm"] = enc_norms
        if not latent_df_exists(args.dictionary):
            push_latent_df(
                df,
                crosscoder=args.dictionary,
                confirm=False,
                create_repo_if_missing=True,
            )

    scaler_lmsys_split = (
        "train" if args.lmsys_col is None else f"train-col{args.lmsys_col}"
    )

    if is_sae:
        logger.info("Running pipeline for SAE - only computing activation scalers")
        if not args.skip_recon_scalars:
            compute_scalers(
                dictionary_model=args.dictionary,
                layer=args.layer,
                activation_store_dir=activation_store_dir,
                results_dir=args.results_dir,
                base_model=args.base_model,
                chat_model=args.chat_model,
                lmsys_split=scaler_lmsys_split,
                target_model_idx=1,
                chat_activation=True,
                base_activation=True,
                chat_activation_no_bias=False,
                base_activation_no_bias=False,
                num_samples=args.num_samples_betas,
                is_difference_sae=args.is_difference_sae,
                sae_model=args.sae_model,
            )
        # Skip error scalars entirely for difference SAEs
        effective_chat_latents_indices = None
        shared_baseline_indices = None
        if not args.skip_error_scalars:
            logger.info("Skipping error scalars computation for difference SAEs")
    else:
        # Original pipeline for dictionaries
        if not args.skip_recon_scalars:
            compute_scalers(
                dictionary_model=args.dictionary,
                layer=args.layer,
                activation_store_dir=activation_store_dir,
                results_dir=args.results_dir,
                base_model=args.base_model,
                chat_model=args.chat_model,
                lmsys_split=scaler_lmsys_split,
                target_model_idx=1,
                chat_activation=True,
                base_activation=True,
                chat_reconstruction=True,
                base_reconstruction=True,
                chat_activation_no_bias=True,
                base_activation_no_bias=True,
                chat_error=False,
                base_error=False,
                num_samples=args.num_samples_betas,
            )

        df = load_latent_df(args.dictionary)
        if args.num_effective_chat_only_latents == -1:
            effective_chat_latents_indices = df.query(
                "tag == 'Chat only'"
            ).index.tolist()
        else:
            effective_chat_latents_indices = (
                df.sort_values(by="dec_norm_diff", ascending=True)
                .head(args.num_effective_chat_only_latents)
                .index.tolist()
            )
        shared_baseline_indices = (
            df[df["tag"] == "Shared"]
            .sample(n=len(effective_chat_latents_indices), random_state=42)
            .index.tolist()
        )
        if not args.skip_error_scalars:
            compute_scalers(
                dictionary_model=args.dictionary,
                layer=args.layer,
                activation_store_dir=activation_store_dir,
                results_dir=args.results_dir,
                base_model=args.base_model,
                chat_model=args.chat_model,
                target_model_idx=1,
                chat_error=True,
                base_error=True,
                latent_indices=effective_chat_latents_indices,
                latent_indices_name="effective_chat_only_latents",
                lmsys_split=scaler_lmsys_split,
                num_samples=args.num_samples_betas,
            )
            compute_scalers(
                dictionary_model=args.dictionary,
                layer=args.layer,
                activation_store_dir=activation_store_dir,
                results_dir=args.results_dir,
                base_model=args.base_model,
                chat_model=args.chat_model,
                target_model_idx=1,
                chat_error=True,
                base_error=True,
                latent_indices=shared_baseline_indices,
                latent_indices_name="shared_baseline_latents",
                lmsys_split=scaler_lmsys_split,
                num_samples=args.num_samples_betas,
            )

    df = make_beta_df(
        args.dictionary,
        args.results_dir,
        effective_chat_latents_indices,
        shared_baseline_indices,
        num_samples=args.num_samples_betas,
    )
    uploaded_latent_df = load_latent_df(args.dictionary)
    for col in set(uploaded_latent_df.columns) - set(df.columns):
        df[col] = uploaded_latent_df[col]
    push_latent_df(
        df,
        crosscoder=args.dictionary,
        confirm=False,
        commit_message="Added betas columns to df",
    )
    if not is_sae:
        chat_only_indices = df[df["tag"] == "Chat only"].index.tolist()
        make_betas_plots(
            df,
            chat_only_indices,
            shared_baseline_indices,
            args.results_dir / "closed_form_scalars" / args.dictionary,
        )

    if args.compute_latent_stats:
        compute_dict_acts = True
        if (latent_activations_dir / args.dictionary).exists():
            compute_dict_acts = False
            logger.info(
                f"Found latent activations for {args.dictionary} in {latent_activations_dir / args.dictionary}. Skipping collection."
            )
            try:
                latent_activation_cache = LatentActivationCache(
                    latent_activations_dir / args.dictionary,
                    expand=False,
                    use_sparse_tensor=False,
                )
            except Exception as e:
                logger.error(
                    f"Error loading latent activation cache for {args.dictionary}: {e}. Recomputing the dictionary activations."
                )
                compute_dict_acts = True
        if compute_dict_acts:
            if is_sae:
                if args.sae_model == "base":
                    sae_model_idx = 0
                else:
                    sae_model_idx = 1
            else:
                sae_model_idx = None
            latent_activation_cache = collect_dictionary_activations(
                dictionary_model_name=args.dictionary,
                latent_activations_dir=latent_activations_dir,
                base_model=args.base_model,
                chat_model=args.chat_model,
                layer=args.layer,
                upload_to_hub=args.upload_to_hub,
                split="validation",
                lmsys_col=args.lmsys_col,
                is_sae=is_sae,
                is_difference_sae=args.is_difference_sae,
                sae_model_idx=sae_model_idx,
            )
            latent_activation_cache.expand = False
            latent_activation_cache.use_sparse_tensor = False
        latent_activation_cache.to(auto_device())

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        quantile_examples_path = Path("results/quantile_examples") / args.dictionary
        if (
            not (quantile_examples_path / "examples.pt").exists()
            or not (quantile_examples_path / "examples.db").exists()
        ):
            collect_activating_examples(
                crosscoder=args.dictionary,
                bos_token_id=tokenizer.bos_token_id,
                latent_activation_cache=latent_activation_cache,
                n=100,
                min_threshold=1e-4,
                quantiles=[0.25, 0.5, 0.75, 0.95, 1.0],
                save_path=Path("results/quantile_examples"),
                test=args.test,
                only_upload=False,
            )
        compute_latent_stats(
            crosscoder=args.dictionary,
            latent_activation_cache=latent_activation_cache,
            layer=args.layer,
            confirm=False,
        )

        tokenizer = patch_tokenizer(
            AutoTokenizer.from_pretrained(args.chat_model), args.chat_model
        )
        compute_latents_template_stats(
            tokenizer=tokenizer,
            crosscoder=args.dictionary,
            latent_activation_cache=latent_activation_cache,
            max_activations=latent_activation_cache.max_activations,
            save_path=args.results_dir / "latents_template_stats",
            test=args.test,
        )

        df = pd.read_csv(
            args.results_dir / "latents_template_stats" / "latent_stats_global.csv"
        )
        # Skip beta analysis for difference SAEs as they don't have chat-only latents
        if not is_sae:
            plot_beta_ratios_template_perc(
                df.query("tag == 'Chat only'"),
                df[df["lmsys_ctrl_%"] > 0.5].query("tag == 'Chat only'"),
                args.results_dir / args.dictionary,
            )

    # KL experiment for dictionary and difference SAE
    if args.run_kl_experiment:
        if dictionary is None:
            dictionary = load_dictionary_model(
                args.dictionary, is_sae=args.is_sae or args.is_difference_sae
            ).to(auto_device())
        base_model = load_hf_model(args.base_model, torch_dtype=th.bfloat16)
        chat_model = load_hf_model(args.chat_model, torch_dtype=th.bfloat16)
        if args.base_model not in MODEL_CONFIGS or args.chat_model not in MODEL_CONFIGS:
            raise ValueError(
                "Both base and chat models must be in MODEL_CONFIGS. Ensure that both models are in MODEL_CONFIGS."
            )
        if not args.skip_token_level_replacement:
            token_level_replacement = MODEL_CONFIGS[args.base_model][
                "token_level_replacement"
            ]
            logger.info(f"Using token level replacement: {token_level_replacement}")
        else:
            logger.info(
                f"Skipping token level replacement for {args.base_model} as --skip-token-level-replacement flag is set"
            )
            token_level_replacement = None
        ignore_first_n_tokens = MODEL_CONFIGS[args.base_model][
            "ignore_first_n_tokens_per_sample"
        ]
        if (
            ignore_first_n_tokens
            != MODEL_CONFIGS[args.chat_model]["ignore_first_n_tokens_per_sample"]
        ):
            raise ValueError(
                f"Weird, ignore_first_n_tokens_per_sample for {args.base_model} and {args.chat_model} are different. If it's expected, you need to adapt the code to handle this."
            )
        logger.info(
            f"Using ignore_first_n_tokens: {ignore_first_n_tokens} for {args.base_model} and {args.chat_model}"
        )

        kl_experiment(
            dictionary=dictionary,
            base_model=base_model,
            chat_model=chat_model,
            tokenizer_name=args.chat_model,
            dictionary_name=args.dictionary,
            dataset_name=args.kl_dataset,
            split=args.kl_dataset_split,
            latent_df=df,
            chat_only_indices=effective_chat_latents_indices,
            layer_to_stop=args.layer,
            max_seq_len=1024,
            dataset_col=args.kl_dataset_col,
            batch_size=args.batch_size_kl,
            test=args.test,
            token_level_replacement=token_level_replacement,
            ignore_first_n_tokens=ignore_first_n_tokens,
            is_difference_sae=args.is_difference_sae,
            num_sae_latents=args.num_effective_chat_only_latents // 2,
            sae_model=args.sae_model,
        )
