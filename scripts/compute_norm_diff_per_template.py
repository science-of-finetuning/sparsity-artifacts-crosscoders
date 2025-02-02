from pathlib import Path
import json
import sys
import json

import torch as th
from datasets import load_dataset
from tqdm.auto import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.aggregation import MaxMetric, MinMetric
from nnterp.nnsight_utils import get_num_layers, get_layer_output
from nnterp import load_model
import seaborn as sns

sys.path.append(".")
from tools.compute_utils import RunningMeanStd
from tools.tokenization_utils import custom_chat_template
from tools.tokenization_utils import chat_template as default_chat_template
from tools.tokenization_utils import tokenize_with_ctrl_ids


th.set_grad_enabled(False)


def run_batch(
    batch,
    model,
    control_mask,
    other_mask,
    other_no_bos_mask,
    clean_activations,
    num_layers,
    results,
    temp_name,
    model_name,  # "chat" or "base"
):
    # Helper function to update metrics for a group
    def update_group_metrics(norm_diff, norms, cosim, group_mask, group_name, i):
        metrics_data = {
            "diff_norm": norm_diff[group_mask],
            "rel_diff": norm_diff[group_mask] / norms[group_mask],
            "cosim": cosim[group_mask],
        }

        for metric_name, values in metrics_data.items():
            for metric_type in results[temp_name][metric_name]:
                results[temp_name][metric_name][metric_type][i][model_name][
                    group_name
                ].update(values)

    norms = []
    cosims = []
    norm_diffs = []

    with model.trace(batch):
        for i in range(num_layers):
            output = get_layer_output(model, i).save()
            clean_act = clean_activations[i].to(output.device)

            # Compute cosine similarity
            output_norm = output.norm(dim=-1, keepdim=True)
            clean_norm = clean_act.norm(dim=-1, keepdim=True)
            cosim = (
                (output * clean_act).sum(dim=-1)
                / (output_norm * clean_norm).squeeze(-1)
            ).save()

            norm_diff = (output - clean_act).norm(dim=-1).save()
            norm = output.norm(dim=-1).save()
            norm_diffs.append(norm_diff)
            norms.append(norm)
            cosims.append(cosim)

    for i in range(num_layers):
        update_group_metrics(
            norm_diffs[i], norms[i], cosims[i], control_mask, "control", i
        )
        update_group_metrics(norm_diffs[i], norms[i], cosims[i], other_mask, "other", i)
        update_group_metrics(
            norm_diffs[i], norms[i], cosims[i], other_no_bos_mask, "other_no_bos", i
        )
        update_group_metrics(
            norm_diffs[i], norms[i], cosims[i], control_mask | other_mask, "all", i
        )


def norm_diff_templates(
    chat_templates,
    dataset,
    base_model,
    chat_model,
    num_layers,
    compare_template=None,
    compare_model=None,
    batch_size=8,
    testing=False,
    seed=42,
):
    if compare_template is None:
        compare_template = default_chat_template
    if compare_model is None:
        compare_model = chat_model
    data = dataset
    th.manual_seed(seed)
    np.random.seed(seed)
    np.random.shuffle(data)

    def base_dict(_cls):
        return {
            k: {k2: _cls() for k2 in ["control", "other", "all", "other_no_bos"]}
            for k in ["chat", "base"]
        }

    results = {
        temp_name: {
            k: {
                "mean": [base_dict(RunningMeanStd) for _ in range(num_layers)],
                "max": [base_dict(MaxMetric) for _ in range(num_layers)],
                "min": [base_dict(MinMetric) for _ in range(num_layers)],
            }
            for k in ["rel_diff", "diff_norm", "cosim"]
        }
        for temp_name in chat_templates.keys()
    }

    max_num_tokens = 5_000 if not testing else 100
    num_tokens = 0
    pbar = tqdm(total=max_num_tokens, desc="Processing Tokens")
    for i in trange(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        normal_batch = tokenize_with_ctrl_ids(
            batch_data,
            chat_model.tokenizer,
            return_assistant_tokens_mask=True,
            return_dict=True,
            chat_template=compare_template,
            truncation=True,
            max_length=1024,
            padding=True,
            return_tensors="pt",
        )
        clean_activations = []
        with compare_model.trace(normal_batch):
            for i in range(num_layers):
                clean_activations.append(get_layer_output(compare_model, i).save())

        for temp_name, c_template in chat_templates.items():
            batch = tokenize_with_ctrl_ids(
                batch_data,
                chat_model.tokenizer,
                chat_template=c_template,
                return_assistant_tokens_mask=True,
                return_dict=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
                padding=True,
            )
            control_mask = batch["ctrl_mask"]
            other_mask = ~control_mask & batch["attention_mask"].bool()
            other_no_bos_mask = other_mask.clone()
            other_no_bos_mask[:, 0] = False

            run_batch(
                batch,
                chat_model,
                control_mask,
                other_mask,
                other_no_bos_mask,
                clean_activations,
                num_layers,
                results,
                temp_name,
                "chat",
            )

            run_batch(
                batch,
                base_model,
                control_mask,
                other_mask,
                other_no_bos_mask,
                clean_activations,
                num_layers,
                results,
                temp_name,
                "base",
            )
        num_new_tokens = min(control_mask.sum().item(), other_mask.sum().item())
        num_tokens += num_new_tokens
        pbar.update(num_new_tokens)
        if num_tokens >= max_num_tokens:
            break
    return results


def plot_results(results, save_path):
    """
    Plot and save results from norm_diff_templates analysis.

    Args:
        results: Dictionary with structure:
            results[template_name][metric_type]["mean"/"max"/"min"][layer_idx][model][token_group]
            where:
                - metric_type is "rel_diff" or "diff_norm"
                - model is "chat" or "base"
                - token_group is "control", "other", or "other_no_bos"
        save_path: Path to save the plots
    """
    # Create subdirectories
    lineplot_dir = save_path / "lineplots"
    heatmap_dir = save_path / "heatmaps"
    lineplot_dir.mkdir(exist_ok=True, parents=True)
    heatmap_dir.mkdir(exist_ok=True, parents=True)

    # Save lineplots
    for temp_name, temp_metrics in results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plot_metric_comparison(
            temp_metrics["diff_norm"], ax1, f"{temp_name} - Diff Norm"
        )
        plot_metric_comparison(temp_metrics["rel_diff"], ax2, f"{temp_name} - Rel Diff")
        plt.tight_layout()
        plt.savefig(lineplot_dir / f"lineplot_{temp_name}.png", dpi=300)
        plt.close()

    # Create heatmaps
    role_setups = set()
    delimiter_setups = set()
    for key in results.keys():
        if isinstance(key, tuple):
            role, delimiter = key
        else:
            role, delimiter = key.split("-")
        role_setups.add(role)
        delimiter_setups.add(delimiter)

    role_setups = sorted(list(role_setups))
    delimiter_setups = sorted(list(delimiter_setups))

    TARGET_LAYER = 13  # The layer we want to analyze

    # Define combinations for heatmaps
    model_types = ["chat", "base"]
    token_groups = ["control", "other", "other_no_bos", "all"]
    metrics = ["rel_diff", "diff_norm"]

    for metric in metrics:
        for model in model_types:
            for token_group in token_groups:
                # Create matrix for heatmap
                matrix = np.zeros((len(delimiter_setups), len(role_setups)))

                # Fill matrix with differences
                for i, delimiter in enumerate(delimiter_setups):
                    for j, role in enumerate(role_setups):
                        key = (
                            (role, delimiter)
                            if isinstance(next(iter(results.keys())), tuple)
                            else f"{role}-{delimiter}"
                        )
                        layer_data = results[key][metric]["mean"][TARGET_LAYER]
                        matrix[i, j] = layer_data[model][token_group]["mean"]

                # Create heatmap
                plt.figure(figsize=(15, 8))
                sns.heatmap(
                    matrix,
                    xticklabels=role_setups,
                    yticklabels=delimiter_setups,
                    annot=True,
                    fmt=".3f",
                    cmap="viridis",
                )

                metric_name = (
                    "Relative Norm Differences"
                    if metric == "rel_diff"
                    else "Norm Differences"
                )
                plt.title(
                    f"{metric_name} at Layer {TARGET_LAYER}\n{model.upper()} Model - {token_group} tokens"
                )
                plt.xlabel("Assistant-User Token Setup")
                plt.ylabel("Start/End of Turn Setup")

                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha="right")

                # Adjust layout to prevent label cutoff
                plt.tight_layout()

                # Save the plot
                filename = (
                    f"heatmap_{metric}_{model}_{token_group}_layer_{TARGET_LAYER}.png"
                )
                plt.savefig(heatmap_dir / filename, dpi=300, bbox_inches="tight")
                plt.close()


def plot_metric_comparison(metric_data, ax, title):
    """Helper function to create a line plot comparison across layers"""
    labels = ["control", "other"]
    num_layers = len(metric_data["mean"])
    layers = np.arange(num_layers)

    # Get layer-wise means and standard errors
    def get_layer_stats(model_type, label):
        means = []
        stds = []
        for layer_metrics in metric_data["mean"]:
            running_mean = layer_metrics[model_type][label]
            if hasattr(running_mean, "compute"):
                stats = running_mean.compute(return_dict=True)
                mean_val = stats["mean"]
                std_val = np.sqrt(stats["var"] / stats["count"])  # Standard error
            else:
                mean_val = running_mean["mean"]
                std_val = np.sqrt(running_mean["var"] / running_mean["count"])
            means.append(mean_val)
            stds.append(std_val)
        return np.array(means), np.array(stds)

    # Plot lines for each model and token type
    for model_type, linestyle in [("chat", "-"), ("base", "--")]:
        for label, color in zip(labels, ["blue", "red"]):
            means, stds = get_layer_stats(model_type, label)
            label_name = f"{model_type}-{label}"
            ax.plot(layers, means, label=label_name, linestyle=linestyle, color=color)
            ax.fill_between(layers, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Diff Norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


# python scripts/compute_norm_diff_per_template.py --use-float16 --compare-template-delimiter blockquote --compare-model chat && python scripts/compute_norm_diff_per_template.py --use-float16 --compare-template-delimiter blockquote --compare-model base
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--device-chat", type=str, default="cuda:0")
    parser.add_argument("--device-base", type=str, default="cuda:1")
    parser.add_argument("--use-float16", "-f16", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--compare-template-role", "-r", type=str, default="default")
    parser.add_argument(
        "--compare-template-delimiter", "-d", type=str, default="default"
    )
    parser.add_argument(
        "--compare-model", type=str, choices=["base", "chat"], default="chat"
    )
    args = parser.parse_args()
    print(
        f"Comparing {args.compare_model} model with {args.compare_template_role} role and {args.compare_template_delimiter} delimiter"
    )
    testing = args.testing
    if testing:
        print("== Testing mode ==")
    lmsys_test = load_dataset("Butanium/lmsys-chat-test-set-gemma")

    device_chat = args.device_chat
    device_base = args.device_base
    chat_model = load_model(
        "google/gemma-2-2b-it",
        device_map=device_chat,
        tokenizer_kwargs={"padding_side": "right"},
        attn_implementation="eager",
        torch_dtype=th.float16 if args.use_float16 else th.bfloat16,
    )
    base_model = load_model(
        "google/gemma-2-2b",
        device_map=device_base,
        tokenizer_kwargs={"padding_side": "right"},
        attn_implementation="eager",
        torch_dtype=th.float16 if args.use_float16 else th.bfloat16,
    )
    num_layers = get_num_layers(chat_model)
    role_setups = {
        "assistant": dict(assistant_token="assistant", user_token="user"),
        "human": dict(assistant_token="model", user_token="human"),
        "assistant_human": dict(assistant_token="assistant", user_token="human"),
        "human_human": dict(assistant_token="human", user_token="human"),
        "user_user": dict(assistant_token="user", user_token="user"),
        "assistant_assistant": dict(
            assistant_token="assistant", user_token="assistant"
        ),
        "model_model": dict(assistant_token="model", user_token="model"),
        "default": dict(assistant_token="model", user_token="user"),
    }
    delimiter_setups = {
        "default": dict(
            start_of_turn_token="<start_of_turn>", end_of_turn_token="<end_of_turn>"
        ),
        "blockquote": dict(
            start_of_turn_token="<blockquote>", end_of_turn_token="</blockquote>"
        ),
        "code": dict(start_of_turn_token="<code>", end_of_turn_token="</code>"),
        "em": dict(start_of_turn_token="<em>", end_of_turn_token="</em>"),
    }
    templates = {}
    for role_setup, r_kwargs in role_setups.items():
        for delimiter_setup, d_kwargs in delimiter_setups.items():
            templates[(role_setup, delimiter_setup)] = custom_chat_template(
                chat_model.tokenizer, **r_kwargs, **d_kwargs
            )
    if not args.skip:
        results = norm_diff_templates(
            templates,
            lmsys_test["train"]["conversation"],
            base_model,
            chat_model,
            num_layers,
            batch_size=8,
            testing=testing,
            seed=42,
            compare_template=templates[
                (args.compare_template_role, args.compare_template_delimiter)
            ],
            compare_model=chat_model if args.compare_model == "chat" else base_model,
        )
    save_path = (
        Path("results/norm_diff")
        / args.compare_model
        / f"{args.compare_template_role}-{args.compare_template_delimiter}"
    )
    if testing:
        save_path = save_path / "test"
    save_path.mkdir(parents=True, exist_ok=True)
    if not args.skip:
        try:
            th.save(results, save_path / "results.pt")
        except Exception as e:
            print("failed to save results")
            print(e)
    else:
        if not (save_path / "clean_results.json").exists():
            if not (save_path / "results.pt").exists():
                raise FileNotFoundError(
                    f"results.pt does not exist in {save_path}, run with --skip to skip this step"
                )
            else:
                results = th.load(save_path / "results.pt")

    def map_base_dict(dic, is_mean):
        if is_mean:
            return {
                model: {
                    k: run_mean.compute(return_dict=True)
                    for k, run_mean in dic[model].items()
                }
                for model in ["chat", "base"]
            }
        else:
            return {
                model: {k: x.compute().item() for k, x in dic[model].items()}
                for model in ["chat", "base"]
            }

    if not args.skip or not (save_path / "clean_results.json").exists():
        clean_results = {
            temp_name: {
                k: {
                    "mean": [
                        map_base_dict(results[temp_name][k]["mean"][i], True)
                        for i in range(num_layers)
                    ],
                    "max": [
                        map_base_dict(results[temp_name][k]["max"][i], False)
                        for i in range(num_layers)
                    ],
                    "min": [
                        map_base_dict(results[temp_name][k]["min"][i], False)
                        for i in range(num_layers)
                    ],
                }
                for k in ["rel_diff", "diff_norm"]
            }
            for temp_name in templates
        }
        try:
            json.dump(
                {"-".join(k): v for k, v in clean_results.items()},
                open(save_path / "clean_results.json", "w"),
            )
        except Exception as e:
            print("failed to save clean results")
            print(e)
    else:
        clean_results = json.load(open(save_path / "clean_results.json"))
        clean_results = {tuple(k.split("-")): v for k, v in clean_results.items()}

    plot_results(clean_results, save_path)
