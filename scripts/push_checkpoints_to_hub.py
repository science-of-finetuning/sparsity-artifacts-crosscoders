#!/usr/bin/env python3
"""
Script to push intermediate checkpoint models as branches within their respective repositories.

For each checkpoint directory, creates branches for each intermediate checkpoint step containing
both the model checkpoint and corresponding eval logs. Excludes the final model (model_final.pt).
Repositories are assumed to already exist on HuggingFace Hub.

Usage:
    python push_checkpoints_to_hub.py /path/to/checkpoints [author_name]

Example:
    python push_checkpoints_to_hub.py checkpoints/ ClementDelangue
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

# Add the tools directory to the path so we can import cc_utils
sys.path.append(str(Path(__file__).parent / "tools"))
HF_NAME = "science-of-finetuning"
from huggingface_hub import HfApi, CommitOperationAdd, repo_exists, file_exists
from cc_utils import push_dictionary_model


hf_api = None  # Will be initialized in main with optional token


def push_all_checkpoint_models(
    checkpoints_dir: Path, author: str = HF_NAME, api: HfApi | None = None, upload_final: bool = False, only_new: bool = False
) -> Dict:
    """Push intermediate checkpoint models as branches within their respective repositories.

    For each checkpoint directory, creates branches for each intermediate checkpoint step containing
    both the model checkpoint and corresponding eval logs. Excludes the final model (model_final.pt).
    Repositories are assumed to already exist on HuggingFace Hub unless upload_final=True.

    Args:
        checkpoints_dir: Directory containing model checkpoint subfolders
        author: HuggingFace author name
        api: HuggingFace API client (authenticated)
        upload_final: If True, create repos and upload model_final.pt if repo doesn't exist
        only_new: If True, only process repos that don't exist yet (skip existing repos)

    Returns:
        Dictionary mapping {repo_name: {branch_name: success_status}}
    """
    if isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory {checkpoints_dir} does not exist")
        return {}

    results = {}
    api = api or HfApi()

    # Process each checkpoint directory
    for model_dir in checkpoints_dir.iterdir():
        if not model_dir.is_dir():
            continue

        config_path = model_dir / "config.json"
        if not config_path.exists():
            print(f"Skipping {model_dir.name}: no config.json found")
            continue

        repo_name = model_dir.name
        repo_id = f"{author}/{repo_name}"
        results[repo_name] = {}

        print(f"\nProcessing repository: {repo_id}")

        # Check if repository exists
        repo_already_exists = repo_exists(repo_id=repo_id, repo_type="model")
        
        if only_new and repo_already_exists:
            print(f"  Skipping {repo_id}: repository already exists (--only-new mode)")
            continue

        if not repo_already_exists:
            if not upload_final:
                print(f"  Skipping {repo_id}: repository not found (use --upload-final to create it)")
                continue

            print(f"  Repository not found. Creating {repo_id} and initializing with model_final.pt")

            model_final = model_dir / "model_final.pt"
            assert model_final.exists(), f"Missing {model_final}. Can't initialize repo without final model."
            assert config_path.exists(), f"Missing {config_path}. Can't initialize repo without config.json."

            # Use cc_utils.push_dictionary_model to properly initialize the repo
            push_dictionary_model(model_final, author=author)
            print(f"  Successfully initialized repository {repo_id}")

        # Find all checkpoint files and their corresponding eval logs
        checkpoint_files = {}
        eval_files = {}

        # Collect checkpoint files (excluding model_final.pt)
        for pt_file in model_dir.glob("checkpoint_*.pt"):
            # Extract step number from filename (e.g., "checkpoint_20000.pt" -> "20000")
            step_match = pt_file.name.replace("checkpoint_", "").replace(".pt", "")
            if step_match.isdigit():
                step = step_match
            else:
                continue

            checkpoint_files[step] = pt_file

        # Collect eval log files (excluding last_eval_logs.pt)
        for pt_file in model_dir.glob("eval_logs_*.pt"):
            # Extract step number from filename (e.g., "eval_logs_20000.pt" -> "20000")
            step_match = pt_file.name.replace("eval_logs_", "").replace(".pt", "")
            if step_match.isdigit():
                step = step_match
            else:
                continue

            eval_files[step] = pt_file

        # Sort steps numerically
        if not checkpoint_files:
            print(f"  Warning: No checkpoint_*.pt files found in {model_dir}")
            continue

        steps = sorted(checkpoint_files.keys(), key=int)
        first_step = steps[0]

        # Create/update branches for each checkpoint step
        for step in steps:
            if step not in eval_files:
                print(f"  Warning: No eval logs found for checkpoint step {step}")
                continue

            branch_name = f"checkpoint-{step}"
            checkpoint_file = checkpoint_files[step]
            eval_file = eval_files[step]

            try:
                # Check if branch exists
                refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
                branch_exists = any(ref.name == branch_name for ref in refs.branches)

                print(
                    f"  Creating/updating branch '{branch_name}' with {checkpoint_file.name} and {eval_file.name}"
                )

                # Prepare commit operations
                operations = [
                    CommitOperationAdd(
                        path_in_repo=checkpoint_file.name, path_or_fileobj="model.pt"
                    ),
                    CommitOperationAdd(
                        path_in_repo=eval_file.name, path_or_fileobj="eval_logs.pt"
                    ),
                ]

                # Upload config as trainer_config.json only once (with first checkpoint)
                if step == first_step and config_path.exists():
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo="trainer_config.json",
                            path_or_fileobj=config_path,
                        )
                    )

                if not branch_exists:
                    # Create a new branch from default branch
                    api.create_branch(
                        repo_id=repo_id,
                        repo_type="model",
                        branch=branch_name,
                    )
                    # Then commit files to that branch
                    api.create_commit(
                        repo_id=repo_id,
                        repo_type="model",
                        operations=operations,
                        commit_message=(
                            f"Initialize {branch_name}: {checkpoint_file.name} + {eval_file.name}"
                        ),
                        revision=branch_name,
                    )
                else:
                    # Update existing branch
                    api.create_commit(
                        repo_id=repo_id,
                        repo_type="model",
                        operations=operations,
                        commit_message=(
                            f"Update {branch_name}: {checkpoint_file.name} + {eval_file.name}"
                        ),
                        revision=branch_name,
                    )

                results[repo_name][branch_name] = True
                print(f"  Successfully created branch '{branch_name}' in {repo_id}")

            except Exception as e:
                print(f"  Failed to create branch '{branch_name}': {e}")
                results[repo_name][branch_name] = False
                continue

    # Print summary
    successful_uploads = sum(
        sum(1 for success in repo_results.values() if success)
        for repo_results in results.values()
    )

    print(f"\n{'='*60}")
    print(f"Upload Summary:")
    print(f"Processed {len(results)} repositories")
    print(f"Successfully created {successful_uploads} branches total")

    for repo_name, repo_results in results.items():
        successful_branches = [branch for branch, success in repo_results.items() if success]
        failed_branches = [branch for branch, success in repo_results.items() if not success]

        print(f"  {repo_name}:")
        if successful_branches:
            print(f"    ✓ {len(successful_branches)} branches: {successful_branches}")
        if failed_branches:
            print(f"    ✗ {len(failed_branches)} failed: {failed_branches}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Push checkpoint models as branches to HuggingFace Hub repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s checkpoints/
  %(prog)s /path/to/checkpoints ClementDelangue
        """
    )
    parser.add_argument(
        "checkpoints_dir",
        help="Directory containing model checkpoint subfolders"
    )
    parser.add_argument(
        "author",
        nargs="?",
        default=HF_NAME,
        help=f"HuggingFace author name (default: {HF_NAME})"
    )
    parser.add_argument(
        "--token",
        dest="token",
        default=None,
        help="Hugging Face access token (overrides environment if provided)",
    )
    parser.add_argument(
        "--upload-final",
        dest="upload_final",
        action="store_true",
        help="Create repos and upload model_final.pt if repository doesn't exist",
    )
    parser.add_argument(
        "--only-new",
        dest="only_new",
        action="store_true",
        help="Only process repositories that don't exist yet (skip existing repos)",
    )

    args = parser.parse_args()

    checkpoints_path = Path(args.checkpoints_dir)
    if not checkpoints_path.exists():
        print(f"Error: Checkpoints directory '{checkpoints_path}' does not exist")
        sys.exit(1)

    if not checkpoints_path.is_dir():
        print(f"Error: '{checkpoints_path}' is not a directory")
        sys.exit(1)

    print(f"Starting upload from {checkpoints_path} as user {args.author}")
    print("=" * 60)

    try:
        api = HfApi(token=args.token) if args.token is not None else HfApi()
        # --only-new implies --upload-final
        upload_final = args.upload_final or args.only_new
        results = push_all_checkpoint_models(checkpoints_path, args.author, api=api, upload_final=upload_final, only_new=args.only_new)
        if not results:
            print("No repositories were processed successfully.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nUpload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during upload: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
