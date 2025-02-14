import json
import re
from pathlib import Path
from argparse import ArgumentParser
from typing import List


def add_random_means(data: dict) -> dict:
    """Add mean across random seeds for each metric type in the data."""
    # For each metric type (e.g. 'all', 'k_first', etc.)
    for metric_type, metrics in data.items():
        # For each metric (e.g. 'loss', 'kl-instruct', etc.)
        for metric_name, setups in metrics.items():
            # Group random setups by their base pattern
            random_groups = {}
            for setup_name, setup_data in setups.items():
                # Check if this is a random setup
                if "random" in setup_name and setup_name != "random":
                    # Extract the base pattern by replacing randomX with random
                    base_pattern = re.sub(r"random\d+", "random", setup_name)
                    if base_pattern not in random_groups:
                        random_groups[base_pattern] = []
                    random_groups[base_pattern].append((setup_name, setup_data))

            # Compute means for each group
            for base_pattern, random_setups in random_groups.items():
                if base_pattern in setups:
                    continue
                if len(random_setups) > 0:  # Only process if we found random setups
                    means = []
                    vars = []
                    ns = []
                    for _, setup_data in random_setups:
                        means.append(setup_data["mean"])
                        vars.append(setup_data["var"])
                        ns.append(setup_data["count"])

                    # Compute pooled statistics
                    total_n = sum(ns)
                    # Weighted mean
                    mean = sum(m * n for m, n in zip(means, ns)) / total_n
                    # Pooled variance (considering both within-group and between-group variance)
                    within_var = sum((v * n) for v, n in zip(vars, ns)) / total_n
                    between_var = (
                        sum(n * (m - mean) ** 2 for m, n in zip(means, ns)) / total_n
                    )
                    pooled_var = within_var + between_var

                    # Add the mean entry to the data
                    setups[base_pattern] = {
                        "mean": mean,
                        "var": pooled_var,
                        "count": total_n,
                    }

    return data


def process_file(file_path: Path, output_path: Path | None = None):
    """Process a single JSON file to add random means."""
    print(f"Processing {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    data = add_random_means(data)

    # If no output path specified, overwrite the input file
    output_path = output_path or file_path
    with open(str(output_path).split(".")[0] + "_fixed.json", "w") as f:
        json.dump(data, f)
    print(f"Saved results to {output_path}")


def merge_json_files(files: List[Path], output_path: Path):
    """Merge multiple JSON files into a single one. Later files take precedence."""
    print(f"Merging {len(files)} files into {output_path}")

    merged_data = {}
    for file_path in files:
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist, skipping")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
            # Update merged_data with new data, overwriting any existing values
            for metric_type, metrics in data.items():
                if metric_type not in merged_data:
                    merged_data[metric_type] = {}
                for metric_name, setups in metrics.items():
                    if metric_name not in merged_data[metric_type]:
                        merged_data[metric_type][metric_name] = {}
                    dic = merged_data[metric_type][metric_name]
                    for setup_name, setup_data in setups.items():
                        if (
                            setup_name in dic
                            and dic[setup_name]["count"] >= setup_data["count"]
                        ):
                            continue
                        dic[setup_name] = setup_data

    with open(output_path, "w") as f:
        json.dump(merged_data, f)
    print(f"Saved merged results to {output_path}")


def main():
    parser = ArgumentParser(description="Process result JSON files")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Add random means command
    means_parser = subparsers.add_parser(
        "means", help="Add random mean columns to result JSON files"
    )
    means_parser.add_argument(
        "files", nargs="+", type=Path, help="JSON files to process"
    )
    means_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for processed files. If not specified, input files will be overwritten.",
    )

    # Merge command
    merge_parser = subparsers.add_parser(
        "merge", help="Merge multiple JSON files into one"
    )
    merge_parser.add_argument("files", nargs="+", type=Path, help="JSON files to merge")
    merge_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path for the merged result",
    )

    args = parser.parse_args()

    if args.command == "means":
        for file_path in args.files:
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist, skipping")
                continue

            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.output_dir / file_path.name
            else:
                output_path = None

            process_file(file_path, output_path)

    elif args.command == "merge":
        merge_json_files(args.files, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# python scripts/edit_eval_results.py means results/interv_effects/ultrachat-gemma-50-ctrl-fixed_cerise-yak/ultrachat-gemma-50-ctrl-fixed_cerise-yak_510_result.json
# python scripts/edit_eval_results.py merge results/interv_effects/ultrachat-gemma_beige-armadillo/ultrachat-gemma_beige-armadillo_330_result_fixed.json results/interv_effects/ultrachat-gemma-50-ctrl-fixed_cerise-yak/ultrachat-gemma-50-ctrl-fixed_cerise-yak_510_result_fixed.json  --output results/interv_effects/merged_gemma_results_fixed.json

# python scripts/edit_eval_results.py means results/interv_effects/lmsys-50-fstfixed_steadfast-mastodon/lmsys-50-fstfixed_steadfast-mastodon_510_result.json

# python scripts/edit_eval_results.py merge results/interv_effects/lmsys-50-fstfixed_steadfast-mastodon/lmsys-50-fstfixed_steadfast-mastodon_510_result_fixed.json results/interv_effects/lmsys-chat-1m-validation2_finicky-raptor/lmsys-chat-1m-validation2_finicky-raptor_900_result_fixed.json --output results/interv_effects/merged_lmsys2_results_fixed.json
