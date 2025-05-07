# %%
import sys

sys.path.append("..")
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path
from tools.utils import load_latent_df

# Download files from hub
latent_ids_path = hf_hub_download(
    repo_id="science-of-finetuning/autointerp-data-gemma-2-2b-l13-mu4.1e-02-lr1e-04",
    filename="latent_ids.pt",
    repo_type="dataset",
)

# Load tensors
latent_ids = torch.load(latent_ids_path, weights_only=False)

df = load_latent_df()


# %%
def calculate_accuracy(stats):
    """Calculate accuracy from confusion matrix stats"""
    tp = stats["TP"]
    tn = stats["TN"]
    fp = stats["FP"]
    fn = stats["FN"]

    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0


def calculate_total_accuracy(inner_dict):
    acc = 0
    for quantile, stats in inner_dict.items():
        acc += calculate_accuracy(stats)
    return acc / len(inner_dict)


def load_and_process_data(filenames):
    """
    Load and process multiple data files, calculating accuracy statistics for each.

    Args:
        filenames: List of filename strings without extension (e.g., ['scores_data_0_1025'])

    Returns:
        DataFrame containing merged accuracy statistics from all files
    """
    dataframes = []

    for filename in filenames:
        # Load detection data
        detection_path = f"detection/{filename}.json"
        detection = json.load(open(detection_path))

        # Process detection data
        detection_results = []
        for feature, inner_dict in detection.items():
            detection_results.append(
                {
                    "feature": feature,
                    "accuracy": calculate_total_accuracy(inner_dict),
                    "filename": filename,
                }
            )
        detection_df = pd.DataFrame(detection_results)

        try:
            # Load fuzzing data
            fuzzing_path = f"fuzzing/{filename}.json"
            fuzzing = json.load(open(fuzzing_path))
            # Process fuzzing data
            fuzzing_results = []
            for feature, inner_dict in fuzzing.items():
                fuzzing_results.append(
                    {
                        "feature": feature,
                        "accuracy": calculate_total_accuracy(inner_dict),
                        "filename": filename,
                    }
                )

            # Convert to DataFrames
            fuzzing_df = pd.DataFrame(fuzzing_results)

            # Merge the dataframes
            file_df = pd.merge(
                detection_df,
                fuzzing_df,
                on=["feature", "filename"],
                suffixes=("_detection", "_fuzzing"),
            )
        except FileNotFoundError:
            print("No fuzzing data found for", filename)
            file_df = detection_df.copy()
            file_df["accuracy_detection"] = file_df["accuracy"]
            file_df = file_df.drop(columns=["accuracy"])
        print(len(file_df))
        dataframes.append(file_df)

    # Stack all dataframes
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        return merged_df
    else:
        return pd.DataFrame()


# %%
# Example usage
filenames = Path("detection").glob("*.json")
filenames = [f.stem for f in filenames]
merged_df = load_and_process_data(filenames)

# %%
import numpy as np

df["autointerp_acc_fuzzing"] = np.nan
df["autointerp_acc_detection"] = np.nan

df.loc[latent_ids[merged_df["feature"].astype(int)], "autointerp_acc_fuzzing"] = (
    merged_df["accuracy_fuzzing"].tolist()
)
df.loc[latent_ids[merged_df["feature"].astype(int)], "autointerp_acc_detection"] = (
    merged_df["accuracy_detection"].tolist()
)


chat_only_df = df[df["tag"] == "IT only"]
shared_df = df.query("tag == 'Shared' and autointerp_acc_detection.notna()")
print(len(chat_only_df))
# Compute ranks for beta_ratio_error and beta_ratio_reconstruction
# Lower values get lower ranks (rank 1 is the smallest value)
chat_only_df["beta_ratio_error_rank"] = chat_only_df["beta_ratio_error"].rank()
chat_only_df["beta_ratio_reconstruction_rank"] = chat_only_df[
    "beta_ratio_reconstruction"
].rank()

shared_df["beta_ratio_error_rank"] = shared_df["beta_ratio_error"].rank()
shared_df["beta_ratio_reconstruction_rank"] = shared_df[
    "beta_ratio_reconstruction"
].rank()

# Print some statistics about the ranks
print(
    f"Beta ratio error rank range: {chat_only_df['beta_ratio_error_rank'].min()} to {chat_only_df['beta_ratio_error_rank'].max()}"
)
print(
    f"Beta ratio reconstruction rank range: {chat_only_df['beta_ratio_reconstruction_rank'].min()} to {chat_only_df['beta_ratio_reconstruction_rank'].max()}"
)


# %%
# Create scatter plot with color based on beta_ratio_error
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    chat_only_df["autointerp_acc_detection"],
    chat_only_df["autointerp_acc_fuzzing"],
    alpha=0.6,
    c=chat_only_df["beta_ratio_error_rank"]
    + chat_only_df["beta_ratio_reconstruction_rank"],
    cmap="viridis",
)

# Add diagonal line
min_val = min(
    chat_only_df["autointerp_acc_detection"].min(),
    chat_only_df["autointerp_acc_fuzzing"].min(),
)
max_val = max(
    chat_only_df["autointerp_acc_detection"].max(),
    chat_only_df["autointerp_acc_fuzzing"].max(),
)
plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

# Add labels and title
plt.xlabel("Detection Accuracy")
plt.ylabel("Fuzzing Accuracy")
plt.title("Detection vs Fuzzing Accuracy by Feature")

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Beta Ratio Error")

# Add grid
plt.grid(True, alpha=0.3)

# Make plot square
plt.axis("square")

# Set axis limits with a small margin
margin = 0.05
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

plt.show()


# %%


# Function to compare top and bottom percentiles of features based on beta ratios
def compare_percentiles(df, percentile=10):
    """
    Compare autointerp scores between top and bottom percentiles of features.

    Args:
        df: DataFrame containing the data
        percentile: Percentage of top/bottom features to compare (default: 10%)

    Returns:
        Tuple of DataFrames (bottom_percentile_df, top_percentile_df)
    """
    # Filter out rows with NaN values in relevant columns
    valid_df = df.dropna(
        subset=[
            "beta_ratio_error_rank",
            "beta_ratio_reconstruction_rank",
            "autointerp_acc_detection",
            "autointerp_acc_fuzzing",
        ]
    )

    if len(valid_df) < len(df):
        print(f"Removed {len(df) - len(valid_df)} rows with NaN values")

    # Calculate the sum of ranks
    sum_ranks = (
        valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"]
    )

    # Calculate percentile thresholds
    bottom_threshold = np.percentile(sum_ranks, percentile)
    top_threshold = np.percentile(sum_ranks, 100 - percentile)
    print(
        f"Bottom threshold: {bottom_threshold:.2f}, Top threshold: {top_threshold:.2f}"
    )

    # Split the dataframe into two groups: top and bottom percentiles
    bottom_percentile_df = valid_df[sum_ranks <= bottom_threshold]
    top_percentile_df = valid_df[sum_ranks >= top_threshold]
    print(
        f"Bottom percentile size: {len(bottom_percentile_df)}, Top percentile size: {len(top_percentile_df)}"
    )

    # Print autointerp scores for each group
    print(
        f"\nAutointerp scores for features in bottom {percentile}% (better beta ratios):"
    )
    print(
        f"Detection accuracy: {bottom_percentile_df['autointerp_acc_detection'].mean():.4f} ± {bottom_percentile_df['autointerp_acc_detection'].std():.4f}"
    )
    print(
        f"Fuzzing accuracy: {bottom_percentile_df['autointerp_acc_fuzzing'].mean():.4f} ± {bottom_percentile_df['autointerp_acc_fuzzing'].std():.4f}"
    )

    print(f"\nAutointerp scores for features in top {percentile}% (worse beta ratios):")
    print(
        f"Detection accuracy: {top_percentile_df['autointerp_acc_detection'].mean():.4f} ± {top_percentile_df['autointerp_acc_detection'].std():.4f}"
    )
    print(
        f"Fuzzing accuracy: {top_percentile_df['autointerp_acc_fuzzing'].mean():.4f} ± {top_percentile_df['autointerp_acc_fuzzing'].std():.4f}"
    )

    # Perform t-test to check if the differences are statistically significant
    from scipy import stats

    detection_ttest = stats.ttest_ind(
        bottom_percentile_df["autointerp_acc_detection"],
        top_percentile_df["autointerp_acc_detection"],
        equal_var=False,
    )
    fuzzing_ttest = stats.ttest_ind(
        bottom_percentile_df["autointerp_acc_fuzzing"],
        top_percentile_df["autointerp_acc_fuzzing"],
        equal_var=False,
    )

    print("\nStatistical significance:")
    print(f"Detection accuracy difference p-value: {detection_ttest.pvalue:.4f}")
    print(f"Fuzzing accuracy difference p-value: {fuzzing_ttest.pvalue:.4f}")

    return bottom_percentile_df, top_percentile_df


# Example usage: compare top and bottom 10%
bottom_10pct, top_10pct = compare_percentiles(chat_only_df, percentile=5)

# You can also try other percentiles
# compare_percentiles(df, percentile=25)  # Compare top/bottom 25%

# %%

valid_df = chat_only_df.dropna(
    subset=[
        "beta_ratio_error_rank",
        "beta_ratio_reconstruction_rank",
        "autointerp_acc_detection",
        "autointerp_acc_fuzzing",
    ]
)

# Create a scatter plot of rank sum vs detection accuracy
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"],
    valid_df["autointerp_acc_detection"],
    alpha=0.7,
    c=valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"],
    cmap="viridis",
)

# Add a color bar to show the rank sum scale
cbar = plt.colorbar(scatter)
cbar.set_label("Sum of Beta Ratio Ranks")

# Add a trend line
z = np.polyfit(
    valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"],
    valid_df["autointerp_acc_detection"],
    1,
)
p = np.poly1d(z)
plt.plot(
    sorted(
        valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"]
    ),
    p(
        sorted(
            valid_df["beta_ratio_error_rank"]
            + valid_df["beta_ratio_reconstruction_rank"]
        )
    ),
    "r--",
    alpha=0.8,
)

# Calculate correlation coefficient
corr, p_value = stats.pearsonr(
    valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"],
    valid_df["autointerp_acc_detection"],
)
# Calculate R-squared value
r_squared = corr**2
print(f"R-squared: {r_squared:.4f}")


# Add labels and title
plt.xlabel("Sum of Beta Ratio Ranks")
plt.ylabel("Detection Accuracy")
plt.title(
    f"Detection Accuracy vs. Sum of Beta Ratio Ranks\nCorrelation: {corr:.3f} (p={p_value:.4f})"
)

# Add grid for better readability
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()

# %%
# Create a new figure for the percentile plot
plt.figure(figsize=(10, 6))

# Define the percentile range
percentile_range = 10

# Calculate the sum of ranks for easier handling
valid_df["rank_sum"] = (
    valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"]
)

# Calculate ranks for dec_norm_diff and freq
valid_df["dec_norm_diff_rank"] = valid_df["dec_norm_diff"].rank(ascending=True)
valid_df["freq_rank"] = valid_df["freq"].rank(ascending=True)

# Create percentile bins based on the defined range
percentiles = np.percentile(valid_df["rank_sum"], np.arange(0, 101, percentile_range))
bin_labels = np.arange(0, 100, percentile_range)
valid_df["percentile_bin"] = pd.cut(
    valid_df["rank_sum"], bins=percentiles, labels=bin_labels
)

# Group by percentile bin and calculate statistics
grouped = valid_df.groupby("percentile_bin")["autointerp_acc_detection"].agg(
    ["mean", "std", "count"]
)
grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])  # Standard error
grouped["ci_95"] = 1.96 * grouped["se"]  # 95% confidence interval

# Plot the means for beta ratio ranks
plt.errorbar(
    grouped.index.astype(float),
    grouped["mean"],
    yerr=grouped["ci_95"],
    fmt="o-",
    capsize=5,
    ecolor="gray",
    markersize=8,
    linewidth=2,
    color="blue",
    label="Beta Ratio Ranks",
)

# Create percentile bins for dec_norm_diff using the same range
dec_percentiles = np.percentile(
    valid_df["dec_norm_diff_rank"], np.arange(0, 101, percentile_range)
)
valid_df["dec_percentile_bin"] = pd.cut(
    valid_df["dec_norm_diff_rank"], bins=dec_percentiles, labels=bin_labels
)

# Group by dec_norm_diff percentile bin and calculate statistics
dec_grouped = valid_df.groupby("dec_percentile_bin")["autointerp_acc_detection"].agg(
    ["mean", "std", "count"]
)
dec_grouped["se"] = dec_grouped["std"] / np.sqrt(dec_grouped["count"])  # Standard error
dec_grouped["ci_95"] = 1.96 * dec_grouped["se"]  # 95% confidence interval

# Plot the means for dec_norm_diff ranks
plt.errorbar(
    dec_grouped.index.astype(float),
    dec_grouped["mean"],
    yerr=dec_grouped["ci_95"],
    fmt="s--",
    capsize=5,
    ecolor="darkred",
    markersize=8,
    linewidth=2,
    color="red",
    label="Dec Norm Diff Ranks",
)

# Create percentile bins for freq using the same range
freq_percentiles = np.percentile(
    valid_df["freq_rank"], np.arange(0, 101, percentile_range)
)
valid_df["freq_percentile_bin"] = pd.cut(
    valid_df["freq_rank"], bins=freq_percentiles, labels=bin_labels
)

# Group by freq percentile bin and calculate statistics
freq_grouped = valid_df.groupby("freq_percentile_bin")["autointerp_acc_detection"].agg(
    ["mean", "std", "count"]
)
freq_grouped["se"] = freq_grouped["std"] / np.sqrt(
    freq_grouped["count"]
)  # Standard error
freq_grouped["ci_95"] = 1.96 * freq_grouped["se"]  # 95% confidence interval

# Plot the means for freq ranks
plt.errorbar(
    freq_grouped.index.astype(float),
    freq_grouped["mean"],
    yerr=freq_grouped["ci_95"],
    fmt="^-.",
    capsize=5,
    ecolor="darkgreen",
    markersize=8,
    linewidth=2,
    color="green",
    label="Frequency Ranks",
)

# Add labels and title
plt.xlabel("Percentile Bins")
plt.ylabel("Average Detection Accuracy")
plt.legend()

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Set x-axis ticks to match percentile bins
x_ticks = np.arange(0, 100, percentile_range)
x_tick_labels = [f"{i}-{i+percentile_range}" for i in x_ticks]
plt.xticks(x_ticks, x_tick_labels)

plt.tight_layout()
plt.show()

# %%
# Create a new figure for the cumulative percentile plot
plt.figure(figsize=(10, 6))

# Calculate the sum of ranks for easier handling
valid_df["rank_sum"] = (
    valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"]
)

# Calculate ranks for dec_norm_diff and frequency
valid_df["dec_norm_diff_rank"] = valid_df["dec_norm_diff"].rank(ascending=True)
valid_df["dec_norm_diff_sum"] = valid_df["dec_norm_diff_rank"]
valid_df["freq_rank"] = valid_df["freq"].rank(ascending=True)

# Create cumulative percentile bins (0-10, 0-20, ..., 0-100)
cumulative_percentiles = []
cumulative_detection_means = []
cumulative_detection_ci = []
cumulative_detection_means_dec = []
cumulative_detection_ci_dec = []
cumulative_detection_means_freq = []
cumulative_detection_ci_freq = []

for i in range(1, 11):
    # For beta ratio ranks
    threshold = np.percentile(valid_df["rank_sum"], i * 10)
    subset = valid_df[valid_df["rank_sum"] <= threshold]

    mean_acc = subset["autointerp_acc_detection"].mean()
    std_acc = subset["autointerp_acc_detection"].std()
    count = len(subset)
    se = std_acc / np.sqrt(count)
    ci_95 = 1.96 * se

    cumulative_percentiles.append(i * 10)
    cumulative_detection_means.append(mean_acc)
    cumulative_detection_ci.append(ci_95)

    # For dec_norm_diff ranks
    threshold_dec = np.percentile(valid_df["dec_norm_diff_sum"], i * 10)
    subset_dec = valid_df[valid_df["dec_norm_diff_sum"] <= threshold_dec]

    mean_acc_dec = subset_dec["autointerp_acc_detection"].mean()
    std_acc_dec = subset_dec["autointerp_acc_detection"].std()
    count_dec = len(subset_dec)
    se_dec = std_acc_dec / np.sqrt(count_dec)
    ci_95_dec = 1.96 * se_dec

    cumulative_detection_means_dec.append(mean_acc_dec)
    cumulative_detection_ci_dec.append(ci_95_dec)

    # For frequency ranks
    threshold_freq = np.percentile(valid_df["freq_rank"], i * 10)
    subset_freq = valid_df[valid_df["freq_rank"] <= threshold_freq]

    mean_acc_freq = subset_freq["autointerp_acc_detection"].mean()
    std_acc_freq = subset_freq["autointerp_acc_detection"].std()
    count_freq = len(subset_freq)
    se_freq = std_acc_freq / np.sqrt(count_freq)
    ci_95_freq = 1.96 * se_freq

    cumulative_detection_means_freq.append(mean_acc_freq)
    cumulative_detection_ci_freq.append(ci_95_freq)

# Plot the means with confidence intervals for beta ratio ranks
plt.errorbar(
    cumulative_percentiles,
    cumulative_detection_means,
    yerr=cumulative_detection_ci,
    fmt="o-",
    capsize=5,
    ecolor="gray",
    markersize=8,
    linewidth=2,
    color="green",
    label="Beta Ratio Ranks",
)

# Plot the means with confidence intervals for dec_norm_diff ranks
plt.errorbar(
    cumulative_percentiles,
    cumulative_detection_means_dec,
    yerr=cumulative_detection_ci_dec,
    fmt="s-",
    capsize=5,
    ecolor="gray",
    markersize=8,
    linewidth=2,
    color="purple",
    label="Dec Norm Diff Ranks",
)

# Plot the means with confidence intervals for frequency ranks
plt.errorbar(
    cumulative_percentiles,
    cumulative_detection_means_freq,
    yerr=cumulative_detection_ci_freq,
    fmt="^-.",
    capsize=5,
    ecolor="gray",
    markersize=8,
    linewidth=2,
    color="orange",
    label="Frequency Ranks",
)

# Add labels and title
plt.xlabel("Cumulative Percentile of Ranks")
plt.ylabel("Average Detection Accuracy")
plt.title(
    "Detection Accuracy by Cumulative Percentile of Ranks\nwith 95% Confidence Intervals"
)
plt.legend()

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Set x-axis ticks to match cumulative percentile bins
plt.xticks(np.arange(10, 101, 20))

plt.tight_layout()
plt.show()

# %%
valid_df.plot.scatter(
    x="rank_sum", y="dec_norm_diff_rank", c="autointerp_acc_detection", cmap="viridis"
)
# %%
valid_df.plot.scatter(x="rank_sum", y="freq", c="autointerp_acc_detection", logy=True)
# %%
# Create a 2D histogram plot
plt.figure(figsize=(10, 8))
hist = plt.hist2d(valid_df["rank_sum"], valid_df["freq_rank"], bins=30, cmap="viridis")
plt.colorbar(label="Count")
plt.xlabel("Rank Sum")
plt.ylabel("Frequency Rank")
plt.title("2D Histogram of Rank Sum vs Frequency Rank")
plt.tight_layout()
plt.show()
# %%

plt.figure(figsize=(10, 8))
hist = plt.hist2d(valid_df["rank_sum"], valid_df["freq_rank"], bins=30, cmap="viridis")
plt.colorbar(label="Count")
plt.xlabel("Rank Sum")
plt.ylabel("Frequency Rank")
plt.title("2D Histogram of Rank Sum vs Frequency Rank")
plt.tight_layout()
plt.show()

# %%

valid_df.plot.scatter(x="rank_sum", y="freq", c="autointerp_acc_detection", logy=True)
# %%
valid_df.plot.scatter(x="rank_sum", y="freq", c="autointerp_acc_detection", logy=True)

# %%
plt.figure(figsize=(10, 8))
tmpdf = valid_df.query("0 < beta_ratio_reconstruction <= 2")
hist = plt.hist2d(
    tmpdf["beta_ratio_reconstruction"],
    np.log10(tmpdf["freq"]),
    bins=30,
    cmap="viridis",
)
plt.colorbar(label="Count")
plt.xlabel("Beta Ratio Error")
plt.ylabel("Frequency")
plt.title("2D Histogram of Beta Ratio Error vs Frequency")
plt.tight_layout()
plt.show()
# %%


def plot_percentile_comparison(
    valid_df,
    percentile_range=10,
    metric="autointerp_acc_detection",
    figsize=(10, 6),
    include_metrics=None,
):
    """
    Create a percentile comparison plot showing how a metric varies across different percentile bins
    of various ranking metrics.

    Args:
        valid_df: DataFrame containing the data with rank columns
        percentile_range: Size of each percentile bin (default: 10)
        metric: The metric to plot on y-axis (default: 'autointerp_acc_detection')
        figsize: Figure size as (width, height) tuple (default: (10, 6))
        include_metrics: List of metrics to include in the plot. Options are:
                        'beta_ratio', 'dec_norm_diff', 'freq'
                        If None, includes all metrics.

    Returns:
        matplotlib figure object
    """
    # Create a new figure
    fig = plt.figure(figsize=figsize)
    font_size = 24
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )
    plt.rcParams["text.usetex"] = True
    # Determine which metrics to include
    if include_metrics is None:
        include_metrics = ["beta_ratio", "dec_norm_diff", "freq"]

    # Ensure rank_sum is calculated
    if "rank_sum" not in valid_df.columns and "beta_ratio" in include_metrics:
        valid_df = valid_df.copy()
        valid_df["rank_sum"] = (
            valid_df["beta_ratio_error_rank"]
            + valid_df["beta_ratio_reconstruction_rank"]
        )

    # Calculate ranks for metrics if they don't exist
    if (
        "dec_norm_diff_rank" not in valid_df.columns
        and "dec_norm_diff" in include_metrics
    ):
        valid_df = valid_df.copy()
        valid_df["dec_norm_diff_rank"] = valid_df["dec_norm_diff"].rank(ascending=True)

    if "freq_rank" not in valid_df.columns and "freq" in include_metrics:
        valid_df = valid_df.copy()
        valid_df["freq_rank"] = valid_df["freq"].rank(ascending=True)

    # Define colors and styles for each metric
    metric_styles = {
        "beta_ratio": {
            "color": "blue",
            "marker": "o",
            "linestyle": "-",
            "label": "Beta Ratio Ranks",
            "rank_col": "rank_sum",
        },
        "dec_norm_diff": {
            "color": "red",
            "marker": "s",
            "linestyle": "--",
            "label": "Dec Norm Diff Ranks",
            "rank_col": "dec_norm_diff_rank",
        },
        "freq": {
            "color": "green",
            "marker": "^",
            "linestyle": "-.",
            "label": "Frequency Ranks",
            "rank_col": "freq_rank",
        },
    }

    # Create bin labels
    bin_labels = np.arange(0, 100, percentile_range)

    # Plot each included metric
    for metric_name in include_metrics:
        style = metric_styles[metric_name]
        rank_col = style["rank_col"]

        # Create percentile bins
        percentiles = np.percentile(
            valid_df[rank_col], np.arange(0, 101, percentile_range)
        )
        bin_col = f"{metric_name}_percentile_bin"
        valid_df[bin_col] = pd.cut(
            valid_df[rank_col], bins=percentiles, labels=bin_labels
        )

        # Group by percentile bin and calculate statistics
        grouped = valid_df.groupby(bin_col)[metric].agg(["mean", "std", "count"])
        grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])  # Standard error
        grouped["ci_95"] = 1.96 * grouped["se"]  # 95% confidence interval

        # Plot the means with confidence intervals
        plt.errorbar(
            grouped.index.astype(float),
            grouped["mean"],
            yerr=grouped["ci_95"],
            fmt=f'{style["marker"]}{style["linestyle"]}',
            capsize=5,
            ecolor="gray",
            markersize=8,
            linewidth=2,
            color=style["color"],
            label=style["label"],
        )

    # Add labels and title
    plt.xlabel("Percentile Bins")
    plt.ylabel(f"Detection Accuracy")
    # plt.title(f'{metric} by Percentile Bins with 95% Confidence Intervals')
    # plt.legend()

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to match percentile bins
    x_ticks = np.arange(0, 100, percentile_range)
    x_tick_labels = [f"${i}$-${i+percentile_range}$" for i in x_ticks]
    plt.xticks(x_ticks, x_tick_labels)
    # Increase font size for better readability
    # Increase font size for better readability

    plt.tight_layout()
    return fig


# Example usage:
# plot_percentile_comparison(valid_df, percentile_range=10)
# plot_percentile_comparison(valid_df, percentile_range=20, metric='autointerp_acc_fuzzing')
# plot_percentile_comparison(valid_df, include_metrics=['beta_ratio', 'freq'])

fig = plot_percentile_comparison(
    valid_df, percentile_range=20, include_metrics=["beta_ratio"], figsize=(8, 4)
)
fig.savefig("beta_ratio_percentile_comparison.pdf", bbox_inches="tight")
# plot_percentile_violin(valid_df, percentile_range=20, include_metrics=["beta_ratio"])

# %%
# %%

tmpdf = valid_df.query("0 < beta_ratio_reconstruction <= 1")
plt.figure(figsize=(10, 8))
hist = plt.hist2d(
    tmpdf["beta_ratio_reconstruction"],
    np.log10(tmpdf["freq"]),
    bins=30,
    cmap="viridis",
)
plt.colorbar(label="Count")
plt.xlabel("Beta Ratio Error")
plt.ylabel("Frequency")
plt.title("2D Histogram of Beta Ratio Error vs Frequency")
plt.tight_layout()
plt.show()
# %%

from scipy import stats

# Calculate Spearman correlation between frequency and rank_sum
spearman_corr, p_value = stats.spearmanr(
    valid_df["beta_ratio_reconstruction"], valid_df["freq"]
)
print(spearman_corr, p_value)
spearman_corr, p_value = stats.spearmanr(valid_df["beta_ratio_error"], valid_df["freq"])
print(spearman_corr, p_value)
# %%
valid_df.plot.scatter(
    x="beta_ratio_reconstruction",
    y="beta_ratio_error",
    c=np.log10(valid_df["freq"]),
    colorbar=True,
    cmap="viridis",
    alpha=0.5,
)
# %%
