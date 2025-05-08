# %%
import sys


sys.path.append("..")
sys.path.append("../autointerp")
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path
from tools.utils import load_latent_df
from scipy import stats
import numpy as np

df_topk = load_latent_df("gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss")
df_cc = load_latent_df("gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04")

# %%
# Calculate Pearson correlation between beta_ratio_error and freq in df_cc
# First, handle NaN values by dropping them
cc_clean_error = df_cc.dropna(subset=['beta_ratio_error', 'freq'])
cc_clean_error = cc_clean_error.query("tag == 'IT only'")

# Calculate correlation and p-value
corr_error, p_error = stats.pearsonr(cc_clean_error['beta_ratio_error'], cc_clean_error['freq'])

print(f"Pearson correlation between beta_ratio_error and freq: {corr_error:.4f}")
print(f"p-value: {p_error:.6f}")

# Calculate Pearson correlation between beta_ratio_reconstruction and freq
# First, handle NaN values by dropping them
cc_clean_recon = df_cc.dropna(subset=['beta_ratio_reconstruction', 'freq'])
cc_clean_recon = cc_clean_recon.query("tag == 'IT only'")

# Calculate correlation and p-value
corr_recon, p_recon = stats.pearsonr(cc_clean_recon['beta_ratio_reconstruction'], cc_clean_recon['freq'])

print(f"Pearson correlation between beta_ratio_reconstruction and freq: {corr_recon:.4f}")
print(f"p-value: {p_recon:.6f}")


# %%
def calculate_accuracy(stats):
    """Calculate accuracy from confusion matrix stats"""
    tp = stats["TP"]
    tn = stats["TN"]
    fp = stats["FP"]
    fn = stats["FN"]

    total =  tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0


def calculate_total_accuracy(inner_dict, skip_quantiles=[]):
    acc = 0
    num_quantiles = 0
    for quantile, stats in inner_dict.items():
        if quantile in skip_quantiles:
            continue
        acc += calculate_accuracy(stats)
        num_quantiles += 1
    return acc / num_quantiles


def load_and_process_data(filenames, quantile=None, use_random=False):
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
        detection_path = f"{filename}.json"
        detection = json.load(open(detection_path))


        # Process detection data
        detection_results = []
        for feature, inner_dict in detection.items():
            result = {
                "feature": feature,
                "total_accuracy_detection": calculate_total_accuracy(inner_dict, skip_quantiles=["-1", "1"] if use_random else ["0", "1"]),
                "filename": filename,
            }
            for quantile, stats in inner_dict.items():
                if quantile == "0" or quantile == "-1":
                    continue
                # result[f"accuracy_detection_{quantile}"] = calculate_accuracy(stats)
                loc_quantiles_to_skip = ["1", "2", "3"]
                loc_quantiles_to_skip.remove(quantile)
                loc_quantiles_to_skip = loc_quantiles_to_skip + (["-1"] if use_random else ["0"]) + ["0", "-1"]
                result[f"accuracy_detection_{quantile}"] = calculate_total_accuracy(inner_dict, skip_quantiles=loc_quantiles_to_skip)

            result["total_accuracy_detection_no_zero"] = calculate_total_accuracy(
                inner_dict, skip_quantiles=["0", "-1"]
            )

            detection_results.append(result)
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
                        "accuracy": (
                            calculate_total_accuracy(inner_dict)
                            if quantile is None
                            else calculate_accuracy(inner_dict[quantile])
                        ),
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
                detection_df,
                fuzzing_df,
                on=["feature", "filename"],
                suffixes=("_detection", "_fuzzing"),
            )
        except FileNotFoundError:
            print("No fuzzing data found for", filename)
            file_df = detection_df.copy()

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
def load_data_for_model(model_path, use_random=True):
    """
    Load data for a specific model with either random or similar examples.
    
    Args:
        model_path (str): Path to the model data
        use_random (bool): Whether to use random examples (True) or similar examples (False)
    
    Returns:
        pd.DataFrame: Processed data for the model
    """
    return load_and_process_data([model_path], use_random=use_random)

def update_dataframe_with_metrics(target_df, source_df, prefix):
    """
    Update target dataframe with metrics from source dataframe.
    
    Args:
        target_df (pd.DataFrame): Target dataframe to update
        source_df (pd.DataFrame): Source dataframe with metrics
        prefix (str): Prefix to add to column names
    """
    for col in source_df.columns:
        if "accuracy_detection" in col:
            target_df.loc[source_df["feature"].astype(int), f"{prefix}_{col}"] = (
                source_df[col].tolist()
            )

# Load data for CrossCoder model
cc_model_path = "../autointerp/data/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04/llama_scores"
merged_df_cc_random = load_data_for_model(cc_model_path, use_random=True)
merged_df_cc_similar = load_data_for_model(cc_model_path, use_random=False)

# Load data for TopK model
topk_model_path_64 = "../autointerp/data/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss/llama_scores_64"
topk_model_path_128 = "../autointerp/data/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss/llama_scores_128"
topk_model_path_256 = "../autointerp/data/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss/llama_scores_256"
merged_df_topk_random_64 = load_data_for_model(topk_model_path_64, use_random=True)
merged_df_topk_similar_64 = load_data_for_model(topk_model_path_64, use_random=False)
merged_df_topk_random_128 = load_data_for_model(topk_model_path_128, use_random=True)
merged_df_topk_similar_128 = load_data_for_model(topk_model_path_128, use_random=False)
merged_df_topk_random_256 = load_data_for_model(topk_model_path_256, use_random=True)
merged_df_topk_similar_256 = load_data_for_model(topk_model_path_256, use_random=False)

# Update dataframes with metrics
df_topk_64 = df_topk.copy()
df_topk_128 = df_topk.copy()
df_topk_256 = df_topk.copy()
update_dataframe_with_metrics(df_cc, merged_df_cc_random, "autointerp_random")
update_dataframe_with_metrics(df_cc, merged_df_cc_similar, "autointerp_similar")
update_dataframe_with_metrics(df_topk_64, merged_df_topk_random_64, "autointerp_random")
update_dataframe_with_metrics(df_topk_64, merged_df_topk_similar_64, "autointerp_similar")
update_dataframe_with_metrics(df_topk_128, merged_df_topk_random_128, "autointerp_random")
update_dataframe_with_metrics(df_topk_128, merged_df_topk_similar_128, "autointerp_similar")
update_dataframe_with_metrics(df_topk_256, merged_df_topk_random_256, "autointerp_random")
update_dataframe_with_metrics(df_topk_256, merged_df_topk_similar_256, "autointerp_similar")


def plot_random_vs_similar_comparison(df, title_prefix="", sample_type="random"):
    versions = [
        "total_accuracy_detection",
        "total_accuracy_detection_no_zero", 
        "accuracy_detection_1",
        "accuracy_detection_2",
        "accuracy_detection_3"
    ]
    ticks = ["Total", "No Zero", "Q1", "Q2", "Q3"]
    
    random_means = []
    random_errors = []
    similar_means = [] 
    similar_errors = []

    for version in versions:
        # Random metrics
        random_col = f"autointerp_random_{version}"
        random_mean = df[random_col].mean()
        random_std = df[random_col].std()
        random_se = random_std / np.sqrt(len(df[random_col].dropna()))
        random_ci = 1.96 * random_se
        random_means.append(random_mean)
        random_errors.append(random_ci)

        # Similar metrics
        similar_col = f"autointerp_similar_{version}"
        similar_mean = df[similar_col].mean()
        similar_std = df[similar_col].std()
        similar_se = similar_std / np.sqrt(len(df[similar_col].dropna()))
        similar_ci = 1.96 * similar_se
        similar_means.append(similar_mean)
        similar_errors.append(similar_ci)

    # Create grouped bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(versions))
    width = 0.35

    plt.bar(x - width/2, random_means, width, yerr=random_errors, capsize=5, label='Random', color='lightblue')
    plt.bar(x + width/2, similar_means, width, yerr=similar_errors, capsize=5, label='Similar', color='lightgreen')

    print(x, ticks)
    plt.xticks(x, ticks, fontsize=12)
    
    plt.ylabel("Mean Detection Accuracy", fontsize=12)
    plt.title(f"{title_prefix}Random vs Similar Detection Accuracy with 95% CI", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    # Add value labels
    for i, (random_mean, similar_mean) in enumerate(zip(random_means, similar_means)):
        plt.text(i - width/2, random_mean + random_errors[i], f"{random_mean:.3f}", ha="center", va="bottom")
        plt.text(i + width/2, similar_mean + similar_errors[i], f"{similar_mean:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

# Plot comparisons for both dataframes
plot_random_vs_similar_comparison(df_cc, "CrossCoder: ")
plot_random_vs_similar_comparison(df_topk_64, "TopK: 64")
plot_random_vs_similar_comparison(df_topk_128, "TopK: 128")
plot_random_vs_similar_comparison(df_topk_256, "TopK: 256")

# %% 
def plot_topk_comparison(df_topk_64, df_topk_128, df_topk_256, metric='autointerp_similar_total_accuracy_detection', title='TopK Comparison'):
    """
    Creates a bar plot comparing the specified metric across different TopK values (64, 128, 256).
    
    Args:
        df_topk_64: DataFrame for TopK 64
        df_topk_128: DataFrame for TopK 128
        df_topk_256: DataFrame for TopK 256
        metric: The metric to compare (default: 'autointerp_similar_total_accuracy_detection')
        title: Title for the plot (default: 'TopK Comparison')
    """
    # Calculate means and confidence intervals
    means = []
    errors = []
    
    # TopK 64
    mean_64 = df_topk_64[metric].mean()
    std_64 = df_topk_64[metric].std()
    se_64 = std_64 / np.sqrt(len(df_topk_64[metric].dropna()))
    ci_64 = 1.96 * se_64
    means.append(mean_64)
    errors.append(ci_64)
    
    # TopK 128
    mean_128 = df_topk_128[metric].mean()
    std_128 = df_topk_128[metric].std()
    se_128 = std_128 / np.sqrt(len(df_topk_128[metric].dropna()))
    ci_128 = 1.96 * se_128
    means.append(mean_128)
    errors.append(ci_128)
    
    # TopK 256
    mean_256 = df_topk_256[metric].mean()
    std_256 = df_topk_256[metric].std()
    se_256 = std_256 / np.sqrt(len(df_topk_256[metric].dropna()))
    ci_256 = 1.96 * se_256
    means.append(mean_256)
    errors.append(ci_256)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    x = np.arange(3)
    labels = ['TopK: 64', 'TopK: 128', 'TopK: 256']
    
    bars = plt.bar(x, means, yerr=errors, capsize=5, color='skyblue', width=0.6)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + errors[i],
                f'{means[i]:.3f}', ha='center', va='bottom')
    
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.title(f'{title}', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Compare TopK values for similar detection accuracy
plot_topk_comparison(df_topk_64, df_topk_128, df_topk_256, 
                    metric='autointerp_similar_total_accuracy_detection',
                    title='Comparison of Detection Accuracy Across TopK Values (Similar)')

# Compare TopK values for random detection accuracy
plot_topk_comparison(df_topk_64, df_topk_128, df_topk_256, 
                    metric='autointerp_random_total_accuracy_detection',
                    title='Comparison of Detection Accuracy Across TopK Values (Random)')

# %%
df_topk = df_topk_64
focus_cc = df_cc[df_cc["tag"] == "IT only"]
focus_topk = df_topk.sort_values(by="dec_norm_diff", ascending=True).head(len(focus_cc))
# focus_topk = df_topk[df_topk["tag"] == "Chat only"]

focus_cc["beta_ratio_error_rank"] = focus_cc["beta_ratio_error"].rank(ascending=True)
focus_cc["beta_ratio_reconstruction_rank"] = focus_cc["beta_ratio_reconstruction"].rank(
    ascending=True
)
focus_topk["beta_ratio_error_rank"] = focus_topk["beta_ratio_error"].rank(
    ascending=True
)
focus_topk["beta_ratio_reconstruction_rank"] = focus_topk[
    "beta_ratio_reconstruction"
].rank(ascending=True)

focus_cc["rank_sum"] = focus_cc["beta_ratio_error_rank"] + focus_cc["beta_ratio_reconstruction_rank"]
focus_topk["rank_sum"] = focus_topk["beta_ratio_error_rank"] + focus_topk["beta_ratio_reconstruction_rank"]

# %%
sample_type = "random"
focus_cc.sort_values(by="beta_ratio_error", ascending=True, inplace=True)
focus_cc[["beta_ratio_error", "beta_ratio_reconstruction", "beta_ratio_error_rank", "beta_ratio_reconstruction_rank", "rank_sum", f"autointerp_{sample_type}_total_accuracy_detection"]].head(10)

# %%
def plot_rank_sum_vs_detection_accuracy(df, sample_type="similar", figsize=(10, 6)):
    """
    Creates a scatter plot showing the relationship between rank sum and detection accuracy.
    
    Args:
        df: DataFrame containing the data to plot
        sample_type: Type of sample to use for accuracy data (default: "similar")
        figsize: Figure size as (width, height) tuple (default: (10, 6))
    
    Returns:
        Tuple of (correlation coefficient, p-value, r-squared)
    """
    plt.figure(figsize=figsize)
    
    # Calculate rank sum
    rank_sum = df["beta_ratio_error_rank"] + df["beta_ratio_reconstruction_rank"]
    
    # Check for and handle NaN values in the data
    if df[f"autointerp_{sample_type}_total_accuracy_detection"].isna().any() or rank_sum.isna().any():
        print(f"Warning: Found {df[f'autointerp_{sample_type}_total_accuracy_detection'].isna().sum()} NaN values in accuracy data")
        print(f"Warning: Found {rank_sum.isna().sum()} NaN values in rank sum data")
        
        # Filter out rows with NaN values for the correlation calculation and plotting
        valid_mask = ~(df[f"autointerp_{sample_type}_total_accuracy_detection"].isna() | rank_sum.isna())
        if valid_mask.sum() == 0:
            print("Error: No valid data points after removing NaNs")
            return None, None, None
            
        # Use only valid data for subsequent operations
        rank_sum = rank_sum[valid_mask]
        df = df[valid_mask].copy()
    # Create scatter plot
    scatter = plt.scatter(
        rank_sum,
        df[f"autointerp_{sample_type}_total_accuracy_detection"],
        alpha=0.7,
        c=rank_sum,
        cmap="viridis",
    )

    # Add a color bar to show the rank sum scale
    cbar = plt.colorbar(scatter)
    cbar.set_label("Sum of Beta Ratio Ranks")

    # Calculate correlation coefficient
    corr, p_value = stats.pearsonr(
        rank_sum,
        df[f"autointerp_{sample_type}_total_accuracy_detection"],
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
    
    return corr, p_value, r_squared

# Example usage
sample_type = "similar"
plot_rank_sum_vs_detection_accuracy(focus_cc, sample_type)
plot_rank_sum_vs_detection_accuracy(focus_topk, sample_type)


# %%
def plot_percentile_accuracy(data_cols, percentile_range=10, figsize=(10,6), sample_types=["random", "similar"]):
    """
    Plot average detection accuracy across percentile bins for multiple dataframes and columns.
    
    Args:
        data_cols: List of (dataframe, column_name, display_name) tuples to plot
        percentile_range: Size of percentile bins (default 10)
        figsize: Figure size as (width, height) tuple (default (10,6))
        sample_types: List of sample types to plot (default ["random", "similar"])
    """
    # Create a new figure for the percentile plot
    plt.figure(figsize=figsize)
    
    # Define colors for each data column
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot for each dataframe, column, and sample type
    for i, (df, col, name) in enumerate(data_cols):
        color = colors[i % len(colors)]
        
        for j, sample_type in enumerate(sample_types):
            valid_df = df.copy()
            
            # Calculate ranks for the column
            valid_df[f"{col}_rank"] = valid_df[col].rank(ascending=True)
            
            # Create percentile bins based on the defined range
            percentiles = np.percentile(
                valid_df[f"{col}_rank"].dropna(), np.arange(0, 101, percentile_range)
            )
            
            bin_labels = np.arange(0, 100, percentile_range)
            valid_df["percentile_bin"] = pd.cut(
                valid_df[f"{col}_rank"], bins=percentiles, labels=bin_labels
            )
            
            # Group by percentile bin and calculate statistics
            grouped = valid_df.groupby("percentile_bin")[f"autointerp_{sample_type}_total_accuracy_detection"].agg(
                ["mean", "std", "count"]
            )
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])  # Standard error
            grouped["ci_95"] = 1.96 * grouped["se"]  # 95% confidence interval
            
            # Plot style alternates between solid and dashed lines based on sample type
            linestyle = '-' if j == 0 else '--'
            marker = 'o' if j == 0 else 's'
            
            # Plot the means
            plt.errorbar(
                grouped.index.astype(float),
                grouped["mean"],
                yerr=grouped["ci_95"],
                fmt=f"{marker}{linestyle}",
                capsize=5,
                ecolor='gray',
                markersize=8,
                linewidth=2,
                color=color,
                label=f"{name} ({sample_type})",
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

plot_percentile_accuracy([(focus_cc, "rank_sum", "L1"), (focus_topk, "rank_sum", "BatchTopK")], percentile_range=25, figsize=(10, 6))   
plt.savefig("percentile_accuracy_rank_sum.pdf", bbox_inches="tight")
plt.show()
plot_percentile_accuracy([(focus_cc, "dec_norm_diff", "L1"), (focus_topk, "dec_norm_diff", "BatchTopK")], percentile_range=25, figsize=(10, 6))   
plt.savefig("percentile_accuracy_dec_norm_diff.pdf", bbox_inches="tight")
plt.show()


# %%
# Create a new figure for the cumulative percentile plot
plt.figure(figsize=(10, 6))
valid_df = df_cc
valid_df["beta_ratio_error_rank"] = valid_df["beta_ratio_error"].rank(ascending=True)
valid_df["beta_ratio_reconstruction_rank"] = valid_df["beta_ratio_reconstruction"].rank(ascending=True)
# Calculate the sum of ranks for easier handling
valid_df["rank_sum"] = (
    valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"]
)
valid_df["rank_sum"] = (
    valid_df["beta_ratio_error_rank"] + valid_df["beta_ratio_reconstruction_rank"]
)

# Calculate ranks for dec_norm_diff and frequency
valid_df["dec_norm_diff_rank"] = valid_df["dec_norm_diff"].rank(ascending=True)
valid_df["dec_norm_diff_sum"] = valid_df["dec_norm_diff_rank"]
# valid_df['freq_rank'] = valid_df['freq'].rank(ascending=True)

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

    mean_acc = subset[f"autointerp_{sample_type}_total_accuracy_detection"].mean()
    std_acc = subset[f"autointerp_{sample_type}_total_accuracy_detection"].std()
    count = len(subset)
    se = std_acc / np.sqrt(count)
    ci_95 = 1.96 * se


    cumulative_percentiles.append(i * 10)
    cumulative_detection_means.append(mean_acc)
    cumulative_detection_ci.append(ci_95)


    # For dec_norm_diff ranks
    threshold_dec = np.percentile(valid_df["dec_norm_diff_sum"], i * 10)
    subset_dec = valid_df[valid_df["dec_norm_diff_sum"] <= threshold_dec]

    mean_acc_dec = subset_dec[f"autointerp_{sample_type}_total_accuracy_detection"].mean()
    std_acc_dec = subset_dec[f"autointerp_{sample_type}_total_accuracy_detection"].std()
    count_dec = len(subset_dec)
    se_dec = std_acc_dec / np.sqrt(count_dec)
    ci_95_dec = 1.96 * se_dec


    cumulative_detection_means_dec.append(mean_acc_dec)
    cumulative_detection_ci_dec.append(ci_95_dec)


    # For frequency ranks
    # threshold_freq = np.percentile(valid_df['freq_rank'], i * 10)
    # subset_freq = valid_df[valid_df['freq_rank'] <= threshold_freq]

    # mean_acc_freq = subset_freq['autointerp_total_accuracy_detection'].mean()
    # std_acc_freq = subset_freq['autointerp_total_accuracy_detection'].std()
    # count_freq = len(subset_freq)
    # se_freq = std_acc_freq / np.sqrt(count_freq)
    # ci_95_freq = 1.96 * se_freq

    # cumulative_detection_means_freq.append(mean_acc_freq)
    # cumulative_detection_ci_freq.append(ci_95_freq)

# Plot the means with confidence intervals for beta ratio ranks
plt.errorbar(
    cumulative_percentiles,
    cumulative_detection_means,
    yerr=cumulative_detection_ci,
    fmt="o-",
    fmt="o-",
    capsize=5,
    ecolor="gray",
    ecolor="gray",
    markersize=8,
    linewidth=2,
    color="green",
    label="Beta Ratio Ranks",
    color="green",
    label="Beta Ratio Ranks",
)

# Plot the means with confidence intervals for dec_norm_diff ranks
plt.errorbar(
    cumulative_percentiles,
    cumulative_detection_means_dec,
    yerr=cumulative_detection_ci_dec,
    fmt="s-",
    fmt="s-",
    capsize=5,
    ecolor="gray",
    ecolor="gray",
    markersize=8,
    linewidth=2,
    color="purple",
    label="Dec Norm Diff Ranks",
    color="purple",
    label="Dec Norm Diff Ranks",
)

# # Plot the means with confidence intervals for frequency ranks
# plt.errorbar(
#     cumulative_percentiles,
#     cumulative_detection_means_freq,
#     yerr=cumulative_detection_ci_freq,
#     fmt='^-.',
#     capsize=5,
#     ecolor='gray',
#     markersize=8,
#     linewidth=2,
#     color='orange',
#     label='Frequency Ranks'
# )

# Add labels and title
plt.xlabel("Cumulative Percentile of Ranks")
plt.ylabel("Average Detection Accuracy")
plt.title(
    "Detection Accuracy by Cumulative Percentile of Ranks\nwith 95% Confidence Intervals"
)
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
    x="rank_sum", y="dec_norm_diff_rank", c=f"autointerp_{sample_type}_total_accuracy_detection", cmap="viridis"
)
# %%
valid_df.plot.scatter(x="rank_sum", y="freq", c=f"autointerp_{sample_type}_total_accuracy_detection", logy=True)
# %%
# Create a 2D histogram plot
plt.figure(figsize=(10, 8))
hist = plt.hist2d(valid_df["rank_sum"], valid_df["freq_rank"], bins=30, cmap="viridis")
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
hist = plt.hist2d(valid_df["rank_sum"], valid_df["freq_rank"], bins=30, cmap="viridis")
plt.colorbar(label="Count")
plt.xlabel("Rank Sum")
plt.ylabel("Frequency Rank")
plt.title("2D Histogram of Rank Sum vs Frequency Rank")
plt.tight_layout()
plt.show()

# %%

valid_df.plot.scatter(x="rank_sum", y="freq", c=f"autointerp_{sample_type}_total_accuracy_detection", logy=True)
# %%
valid_df.plot.scatter(x="rank_sum", y="freq", c=f"autointerp_{sample_type}_total_accuracy_detection", logy=True)

# %%
plt.figure(figsize=(10, 8))
tmpdf = valid_df.query("0 < beta_ratio_reconstruction <= 2")
hist = plt.hist2d(
    tmpdf["beta_ratio_reconstruction"],
    np.log10(tmpdf["freq"]),
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
    data,
    percentile_range=10,
    metric=f"autointerp_{sample_type}_total_accuracy_detection",
    figsize=(10, 6),
    include_metrics=None,
    show_mean=True,
):
    """
    Create a percentile comparison plot showing how a metric varies across different percentile bins
    of various ranking metrics.


    Args:
        data: Either a single DataFrame or a list of (DataFrame, name) tuples containing the data with rank columns
        percentile_range: Size of each percentile bin (default: 10)
        metric: The metric to plot on y-axis (default: 'autointerp_total_accuracy_detection')
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

    # Convert single dataframe to list format
    if not isinstance(data, list):
        data_list = [(data, "")]
    else:
        data_list = data
        
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

    # Define different colors and markers for multiple dataframes
    df_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    df_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'p']

    # Create bin labels
    bin_labels = np.arange(0, 100, percentile_range)
    
    # Process each dataframe
    for df_idx, (valid_df, df_name) in enumerate(data_list):
        valid_df = valid_df.copy()
        
        # Calculate mean for this dataset
        dataset_mean = valid_df[metric].mean()
        
        # Ensure rank_sum is calculated
        if "rank_sum" not in valid_df.columns and "beta_ratio" in include_metrics:
            valid_df["rank_sum"] = (
                valid_df["beta_ratio_error_rank"]
                + valid_df["beta_ratio_reconstruction_rank"]
            )

        # Calculate ranks for metrics if they don't exist
        if (
            "dec_norm_diff_rank" not in valid_df.columns
            and "dec_norm_diff" in include_metrics
        ):
            valid_df["dec_norm_diff_rank"] = valid_df["dec_norm_diff"].rank(ascending=True)

        if "freq_rank" not in valid_df.columns and "freq" in include_metrics:
            valid_df["freq_rank"] = valid_df["freq"].rank(ascending=True)

        # Plot each included metric
        for metric_name in include_metrics:
            style = metric_styles[metric_name].copy()
            rank_col = style["rank_col"]
            
            # Modify label if multiple dataframes
            if df_name:
                style["label"] = f"{df_name}"
                
            # Assign different colors and markers for multiple dataframes
            if len(data_list) > 1:
                style["color"] = df_colors[df_idx % len(df_colors)]
                style["marker"] = df_markers[df_idx % len(df_markers)]

            # Create percentile bins
            percentiles = np.percentile(
                valid_df[rank_col][valid_df[rank_col].notna()], np.arange(0, 101, percentile_range)
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

            if show_mean:
                # Plot mean horizontal line for this dataset with matching color
                plt.axhline(
                    y=dataset_mean, 
                    color=style["color"], 
                    linestyle=':', 
                    linewidth=2,
                    alpha=0.7,
                )
            

    # Add labels and title
    plt.xlabel(r"Percentile Bins of $rank(\nu^r) + rank(\nu^\epsilon)$")
    plt.ylabel(f"Detection Accuracy")
    # plt.title(f'{metric} by Percentile Bins with 95% Confidence Intervals')
    
    # Add legend if multiple dataframes or metrics
    if len(data_list) > 1 or len(include_metrics) > 1:
        plt.legend(loc='lower left')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)


    # Set x-axis ticks to match percentile bins
    x_ticks = np.arange(0, 100, percentile_range)
    x_tick_labels = [f"${i}$-${i+percentile_range}$" for i in x_ticks]
    x_tick_labels = [f"${i}$-${i+percentile_range}$" for i in x_ticks]
    plt.xticks(x_ticks, x_tick_labels)

    plt.tight_layout()
    return fig


# Example usage:
# plot_percentile_comparison(valid_df, percentile_range=10)
# plot_percentile_comparison(valid_df, percentile_range=20, metric='autointerp_acc_fuzzing')
# plot_percentile_comparison(valid_df, include_metrics=['beta_ratio', 'freq'])

fig = plot_percentile_comparison(
    [(focus_cc, "L1"), (focus_topk, "BatchTopK")], show_mean=False, metric="autointerp_similar_total_accuracy_detection", percentile_range=20, include_metrics=["beta_ratio"], figsize=(8, 4)
)
fig.savefig("autointerp_beta_ratio_percentile_comparison.pdf", bbox_inches="tight")
# plot_percentile_violin(valid_df, percentile_range=20, include_metrics=["beta_ratio"])

# %%
# Create a simple bar plot comparing CC and TopK models
plt.figure(figsize=(8, 6))
font_size = 18
plt.rcParams.update({
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
})

# Calculate means and confidence intervals
cc_mean = focus_cc["autointerp_similar_accuracy_detection_3"].mean()
cc_std = focus_cc["autointerp_similar_accuracy_detection_3"].std()
cc_ci = 1.96 * cc_std / np.sqrt(len(focus_cc))

topk_mean = focus_topk["autointerp_similar_accuracy_detection_3"].mean()
topk_std = focus_topk["autointerp_similar_accuracy_detection_3"].std()
topk_ci = 1.96 * topk_std / np.sqrt(len(focus_topk))

# Create bar plot
models = ["CrossCoder", "BatchTopK"]
means = [cc_mean, topk_mean]
errors = [cc_ci, topk_ci]

x_pos = np.arange(len(models))
plt.bar(x_pos, means, yerr=errors, capsize=10, width=0.6, color=['blue', 'red'], alpha=0.7)

# Add labels and formatting
plt.ylabel("Detection Accuracy")
plt.xticks(x_pos, models)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars
for i, (mean, error) in enumerate(zip(means, errors)):
    plt.text(i, mean + error + 0.01, f"{mean:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%

tmpdf = valid_df.query("0 < beta_ratio_reconstruction <= 1")
plt.figure(figsize=(10, 8))
hist = plt.hist2d(
    tmpdf["beta_ratio_reconstruction"],
    np.log10(tmpdf["freq"]),
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
spearman_corr, p_value = stats.spearmanr(
    valid_df["beta_ratio_reconstruction"], valid_df["freq"]
)
print(spearman_corr, p_value)
spearman_corr, p_value = stats.spearmanr(valid_df["beta_ratio_error"], valid_df["freq"])
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
valid_df.plot.scatter(
    x="beta_ratio_reconstruction",
    y="beta_ratio_error",
    c=np.log10(valid_df["freq"]),
    colorbar=True,
    cmap="viridis",
    alpha=0.5,
)
# %%
