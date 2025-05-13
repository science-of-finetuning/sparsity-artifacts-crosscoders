"""
Analyze a given crosscoder as shown in the paper.

1. Compute crosscoder latent activations on validation set
2. Run eval_crosscoder notebook
"""

from run_notebook import run_notebook
import time
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from transformers import AutoTokenizer

from tools.utils import dict_to_args, auto_device, load_hf_model
from tools.tokenization_utils import patch_tokenizer
from tools.cc_utils import load_latent_df, push_latent_df, load_dictionary_model
from scripts import (
    compute_latent_activations,
    collect_activating_examples,
    compute_latent_stats,
    compute_scalers,
    kl_experiment,
    compute_latents_template_stats,
)
from scripts.eval_betas import (
    load_betas_results,
    add_possible_cols,
    plot_error_vs_reconstruction,
    plot_ratio_histogram,
    plot_beta_distribution_histograms,
    plot_correlation_with_frequency,
    plot_rank_distributions,
    plot_beta_ratios_template_perc,
)
from tools.cache_utils import LatentActivationCache
from loguru import logger


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def frequency_plot(df: pd.DataFrame):
    # Get unique tags for grouping
    tags = df["tag"].unique()

    # Create figure
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(111)

    # Colors matching the other plot
    colors = {
        "Chat only": "C0",
        "Base only": "C1",
        "Shared": "C2",
        "Other": "darkgray",
    }

    # Apply log transformation to frequency data
    all_freqs = np.concatenate(
        [
            np.log10(
                df[(df["tag"] == tag) & (df["lmsys_freq"] > 1e-8)]["lmsys_freq"] + 1e-10
            )
            for tag in tags
        ]
    )

    # Determine bin edges in log space
    bins = np.linspace(min(all_freqs), max(all_freqs), 30)
    bin_width = bins[1] - bins[0]

    # Calculate bar width and offsets
    n_tags = len(tags)
    single_bar_width = bin_width / (n_tags)  # Add 1 for spacing
    offsets = np.linspace(
        -bin_width / 2 + single_bar_width / 2,
        bin_width / 2 - single_bar_width / 2,
        n_tags,
    )

    # Plot histogram for each tag
    for tag, offset in zip(tags, offsets):
        tag_data = df[df["tag"] == tag]
        # Apply log transformation to the data
        log_freqs = np.log10(
            tag_data["lmsys_freq"] + 1e-10
        )  # Add small constant to avoid log(0)
        counts, _ = np.histogram(log_freqs, bins=bins)
        normalized_counts = counts / counts.sum()
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.bar(
            bin_centers + offset,
            normalized_counts,
            width=single_bar_width,
            alpha=1.0,
            label=tag.replace("Chat only", "Chat-only").replace(
                "Base only", "Base-only"
            ),
            color=colors[tag],
        )

    # Styling
    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({"font.size": 20})

    ax.grid(True, alpha=0.15)

    # Use more human-readable tick values at nice round numbers
    log_ticks = np.array([-10, -8, -6, -4, -2])  # Powers of 10 for cleaner values
    log_ticks = log_ticks[
        np.logical_and(log_ticks >= min(all_freqs), log_ticks <= max(all_freqs))
    ]
    if len(log_ticks) < 3:  # Ensure we have enough ticks
        log_ticks = np.linspace(min(all_freqs), max(all_freqs), 5)
        log_ticks = np.round(log_ticks)  # Round to integers for cleaner display

    ax.set_xticks(log_ticks)
    ax.set_xticklabels(
        [f"$10^{{{int(x)}}}$" for x in log_ticks]
    )  # Use LaTeX for cleaner display

    ax.set_xlabel("Latent Frequency (log scale)")
    ax.set_ylabel("Density")

    # Move legend below plot
    ax.legend(fontsize=16, loc="upper left")

    plt.savefig("latent_frequency_histogram.pdf", bbox_inches="tight")
    plt.show()


def make_beta_df(
    crosscoder: str,
    data_dir: Path,
    results_dir: Path,
    chat_specific_indices: list[int],
    shared_indices: list[int],
):
    betas_dir = data_dir / "results" / "closed_form_scalars"
    cc_name = crosscoder.replace("/", "_")
    plots_dir = results_dir / "closed_form_scalars" / cc_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "normal": {
            model: {
                target: (f"{model}_{target}", "")
                for target in [
                    "error",
                    "reconstruction",
                    "activation",
                    "activation_no_bias",
                ]
            }
            for model in ["base", "chat"]
        },
    }

    df = load_latent_df(crosscoder)
    all_betas, count_active = load_betas_results(
        betas_dir / cc_name / "all_latents", configs
    )
    chat_error_betas, count_active_chat = load_betas_results(
        betas_dir / cc_name / "effective_chat_only_latents", configs
    )
    shared_error_betas, count_active_shared = load_betas_results(
        betas_dir / cc_name / "shared_baseline_latents", configs
    )

    df = add_possible_cols(df, df.index.tolist(), all_betas)
    df = add_possible_cols(df, chat_specific_indices, chat_error_betas)
    df = add_possible_cols(df, shared_indices, shared_error_betas)
    df_path = results_dir / cc_name / "latent_df.csv"
    df_path.parent.mkdir(exist_ok=True)
    if df_path.exists():
        logger.info(
            f"Updating local latent df at {df_path}, old df will be backed up to {df_path}.{int(time.time())}"
        )
        df_path.rename(df_path.with_suffix(f".{int(time.time())}.csv"))
    df.to_csv(df_path)
    logger.info(f"Saved latent df with betas to {df_path}")
    return df


def make_betas_plots(
    df: pd.DataFrame,
    chat_specific_indices: list[int],
    shared_indices: list[int],
    plots_dir: Path,
):
    target_df = df.iloc[chat_specific_indices]
    baseline_df = df.iloc[shared_indices]
    plot_error_vs_reconstruction(target_df, baseline_df, plots_dir, variant="standard")
    plot_error_vs_reconstruction(
        target_df, baseline_df, plots_dir, variant="custom_color"
    )
    plot_error_vs_reconstruction(target_df, baseline_df, plots_dir, variant="poster")

    plot_ratio_histogram(target_df, baseline_df, plots_dir, ratio_type="error")
    plot_ratio_histogram(target_df, baseline_df, plots_dir, ratio_type="reconstruction")

    plot_beta_distribution_histograms(target_df, plots_dir)
    plot_correlation_with_frequency(df, plots_dir)
    plot_rank_distributions(target_df, plots_dir)
    beta_ratio_error = df["beta_ratio_error"]
    beta_ratio_reconstruction = df["beta_ratio_reconstruction"]
    mask = ~np.isnan(beta_ratio_error) & ~np.isnan(beta_ratio_reconstruction)
    beta_ratio_error_clean = beta_ratio_error[mask]
    beta_ratio_reconstruction_clean = beta_ratio_reconstruction[mask]
    error_ranks = beta_ratio_error_clean.rank()
    reconstruction_ranks = beta_ratio_reconstruction_clean.rank()
    print(
        f"Beta ratio error ranks range: {error_ranks.min():.0f} to {error_ranks.max():.0f}"
    )
    print(
        f"Beta ratio reconstruction ranks range: {reconstruction_ranks.min():.0f} to {reconstruction_ranks.max():.0f}"
    )


# python crosscoder_analysis_pipeline.py gemma-2-2b-L13-k100-lr1e-04-local-shuffling-Decoupled --layer 13 --data-dir /workspace/data/ --results-dir /workspace/data/results
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("crosscoder", type=str)
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
    parser.add_argument("--lmsys-col", type=str, default="")
    parser.add_argument("--kl-dataset", type=str, default="science-of-finetuning/ultrachat_200k_gemma-2-2b-it-generated", help="Dataset to use for KL experiment")
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
    args = parser.parse_args()
    if args.chat_model_idx != 1:
        c = input(
            f"Chat model idx set to {args.chat_model_idx} != 1. Some of the analysis pipeline will not work as expected (e.g. kl experiment). Continue? y/(n)"
        )
        if c != "y":
            exit()

    latent_activations_dir = args.data_dir / "latent_activations"
    activation_store_dir = args.data_dir / "activations"
    if args.tokenizer is None:
        args.tokenizer = args.chat_model
    # run_notebook(
    #     notebook="eval_crosscoder",
    #     crosscoder=args.crosscoder,
    #     extra_args=dict_to_args(upload=args.upload_to_hub, overwrite=True),
    # )
    scaler_lmsys_split = (
        "train" if args.lmsys_col is None else f"train-col{args.lmsys_col}"
    )
    # compute_scalers(
    #     dictionary_model=args.crosscoder,
    #     layer=args.layer,
    #     activation_store_dir=activation_store_dir,
    #     results_dir=args.results_dir,
    #     base_model=args.base_model,
    #     chat_model=args.chat_model,
    #     lmsys_split=scaler_lmsys_split,
    #     target_model_idx=1,
    #     chat_activation=True,
    #     base_activation=True,
    #     chat_reconstruction=True,
    #     base_reconstruction=True,
    #     chat_activation_no_bias=True,
    #     base_activation_no_bias=True,
    #     chat_error=False,
    #     base_error=False,
    # )
    df = load_latent_df(args.crosscoder)
    if args.num_effective_chat_only_latents == -1:
        effective_chat_latents_indices = df.query("tag == 'Chat only'").index.tolist()
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
    # compute_scalers(
    #     dictionary_model=args.crosscoder,
    #     layer=args.layer,
    #     activation_store_dir=activation_store_dir,
    #     results_dir=args.results_dir,
    #     base_model=args.base_model,
    #     chat_model=args.chat_model,
    #     target_model_idx=1,
    #     chat_error=True,
    #     # base_error=True,
    #     latent_indices=effective_chat_latents_indices,
    #     latent_indices_name="effective_chat_only_latents",
    #     lmsys_split=scaler_lmsys_split,
    # )
    # compute_scalers(
    #     dictionary_model=args.crosscoder,
    #     layer=args.layer,
    #     activation_store_dir=activation_store_dir,
    #     results_dir=args.results_dir,
    #     base_model=args.base_model,
    #     chat_model=args.chat_model,
    #     target_model_idx=1,
    #     chat_error=True,
    #     # base_error=True,
    #     latent_indices=shared_baseline_indices,
    #     latent_indices_name="shared_baseline_latents",
    #     lmsys_split=scaler_lmsys_split,
    # )
    # df = make_beta_df(
    #     args.crosscoder,
    #     args.data_dir,
    #     args.results_dir,
    #     effective_chat_latents_indices,
    #     shared_baseline_indices,
    # )
    # chat_only_indices = df[df["tag"] == "Chat only"].index.tolist()
    # if args.upload_to_hub:
    #     push_latent_df(
    #         df,
    #         crosscoder=args.crosscoder,
    #         confirm=False,
    #         commit_message="Added betas columns to df",
    #     )
    # make_betas_plots(
    #     df,
    #     chat_only_indices,
    #     shared_baseline_indices,
    #     args.results_dir / "closed_form_scalars" / args.crosscoder,
    # )
    # compute_latent_activations(
    #     dictionary_model=args.crosscoder,
    #     latent_activations_dir=latent_activations_dir,
    #     base_model=args.base_model,
    #     chat_model=args.chat_model,
    #     layer=args.layer,
    #     upload_to_hub=args.upload_to_hub,
    #     split="validation",
    #     lmsys_col=args.lmsys_col,
    # )
    # latent_activation_cache = LatentActivationCache(
    #     latent_activations_dir / args.crosscoder, expand=False, use_sparse_tensor=False
    # )
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # collect_activating_examples(
    #     crosscoder=args.crosscoder,
    #     bos_token_id=tokenizer.bos_token_id,
    #     latent_activation_cache=latent_activation_cache,
    #     n=100,
    #     min_threshold=1e-4,
    #     quantiles=[0.25, 0.5, 0.75, 0.95, 1.0],
    #     save_path=Path("results/quantile_examples"),
    #     test=args.test,
    #     only_upload=True,
    # )
    # compute_latent_stats(
    #     crosscoder=args.crosscoder,
    #     latent_activation_cache=latent_activation_cache,
    #     layer=args.layer,
    #     confirm=False,
    # )

    tokenizer = patch_tokenizer(
        AutoTokenizer.from_pretrained(args.chat_model), args.chat_model
    )
    # compute_latents_template_stats(
    #     tokenizer=tokenizer,
    #     crosscoder=args.crosscoder,
    #     latent_activation_cache=latent_activation_cache,
    #     max_activations=latent_activation_cache.max_activations,
    #     save_path=args.results_dir / "latents_template_stats",
    #     test=args.test,
    # )
    # df = pd.read_csv(
    #     args.results_dir / "latents_template_stats" / "latent_stats_global.csv"
    # )
    # plot_beta_ratios_template_perc(
    #     df.query("tag == 'Chat only'"),
    #     df[df["lmsys_ctrl_%"] > 0.5].query("tag == 'Chat only'"),
    #     args.results_dir / args.crosscoder,
    # )
    dictionary = load_dictionary_model(args.crosscoder).to(auto_device())
    base_model = load_hf_model(args.base_model)
    chat_model = load_hf_model(args.chat_model)
    kl_experiment(
        dictionary=dictionary,
        base_model=base_model,
        chat_model=chat_model,
        tokenizer=tokenizer,
        # dataset_name="science-of-finetuning/ultrachat_200k_gemma-2-2b-it-generated",
        dataset_name="science-of-finetuning/lmsys-chat-1m-chat-formatted",
        split="train",
        latent_df=df,
        chat_only_indices=effective_chat_latents_indices,
        layer_to_stop=args.layer,
        max_seq_len=1024,
        dataset_col=args.lmsys_col
    )
