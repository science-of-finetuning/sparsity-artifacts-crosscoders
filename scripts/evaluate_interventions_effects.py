import sys

sys.path.append(".")
from datasets import load_from_disk
from setup_to_eval import *
from dlabutils import model_path
from pathlib import Path
from argparse import ArgumentParser
import json
import pandas as pd
import torch as th
import wandb
from evaluate import evaluate_interventions
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer-to-stop", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/test/lmsys_chat"
    )
    parser.add_argument(
        "--crosscoder-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--feature-df-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--it-only-feature-list-path", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-path", type=Path, default=Path("results-runai"))
    args = parser.parse_args()
    instruct_model = AutoModelForCausalLM.from_pretrained(
        model_path("google/gemma-2-2b-it"),
        torch_dtype=th.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path("google/gemma-2-2b-it"))
    instruct_model.tokenizer = tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path("google/gemma-2-2b"),
        device_map="cuda",
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
    )
    dataset = load_from_disk(args.dataset_path)
    if args.test:
        dataset = dataset.select(range(300))
    else:
        dataset = dataset.select(range(30_000))
    device = (
        args.device
        if args.device != "auto"
        else "cuda" if th.cuda.is_available() else "cpu"
    )
    crosscoder = CrossCoder.from_pretrained(args.crosscoder_path, device=device)
    df = pd.read_csv(args.feature_df_path)
    it_only_features = df[df["tag"] == "IT only"].index.tolist()
    base_only_features = df[df["tag"] == "Base only"].index.tolist()
    if args.it_only_feature_list_path is not None:
        it_only_features = pd.read_json(args.it_only_feature_list_path).index.tolist()
        print(f"Using IT only features: {it_only_features}")
    if args.name is None and args.test:
        args.name = "test"
    project = "perplexity-comparison"
    if args.test:
        project += "-test"
    wandb.init(project=project, name=args.name)
    seeds = list(range(10))
    fn_dict = {}
    # fn_dict = create_half_fn_dict_main(crosscoder, it_only_features, base_only_features)
    # fn_dict.update(create_half_fn_dict_no_cross())
    # fn_dict.update(create_half_fn_dict_steer_seeds(crosscoder, seeds, len(it_only_features)))
    # fn_dict.update(create_half_fn_dict_remove_seeds(crosscoder, seeds, len(it_only_features)))
    # fn_dict.update(
    #     create_half_fn_dict_secondary(crosscoder, it_only_features, base_only_features)
    # )
    ## fn_dict.update(
    ##     create_it_only_ft_fn_dict(crosscoder, it_only_features, base_only_features)
    ## )
    # fn_dict.update(create_tresholded_baseline_half_fns(crosscoder, threshold=10))
    # fn_dict.update(
    #     create_tresholded_baseline_random_half_fns(
    #         crosscoder, seeds, len(INTERESTING_FEATURES), threshold=10
    #     )
    # )
    # fn_dict.update(
    #     create_tresholded_baseline_random_half_fns(crosscoder, seeds, 1, threshold=10)
    # )
    fn_dict.update(
        create_half_fn_thresholded_features(crosscoder, steering_factor=100.0)
    )
    fn_dict.update(
        create_tresholded_it_baseline_half_fns(crosscoder, threshold=10)
    )
    # fn_dict.update(
    #     create_half_fn_dict_steer_seeds(
    #         crosscoder, seeds, len(INTERESTING_FEATURES), threshold=10
    #     )
    # )
    # fn_dict.update(
    #     create_half_fn_dict_remove_seeds(
    #         crosscoder, seeds, len(INTERESTING_FEATURES), threshold=10
    #     )
    # )
    # fn_dict.update(create_half_fn_dict_steer_seeds(crosscoder, seeds, 1, threshold=10))
    # fn_dict.update(create_half_fn_dict_remove_seeds(crosscoder, seeds, 1, threshold=10))
    # fn_dict["test-mask"] = TestMaskPreprocessFn()
    result = evaluate_interventions(
        base_model,
        instruct_model,
        dataset["conversation"],
        # create_half_fn_dict_main(crosscoder, it_only_features, base_only_features),
        fn_dict,
        layer_to_stop=args.layer_to_stop,
        batch_size=args.batch_size,
        device=device,
        max_seq_len=args.max_seq_len,
        log_every=args.log_every,
        save_path=args.save_path,
    )
    args.save_path.mkdir(parents=True, exist_ok=True)
    wdb_name = wandb.run.name
    with open(args.save_path / f"{wdb_name}_result.json", "w") as f:
        json.dump(result, f)
