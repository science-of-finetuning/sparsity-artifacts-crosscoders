from argparse import ArgumentParser
from pathlib import Path

import torch as th

from evaluate_interventions_effects import kl_experiment
from tools.utils import load_hf_model
from tools.configs import MODEL_CONFIGS
from coolname import generate_slug
from nnterp.nnsight_utils import get_num_layers
from loguru import logger

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="science-of-finetuning/ultrachat_200k_gemma-2-2b-it-generated",
    )
    parser.add_argument("--dataset-col", type=str, default="messages")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--base-device", type=str, default="cuda")
    parser.add_argument("--chat-device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-path", type=Path, default=Path("results/layer_sweep_kl")
    )
    parser.add_argument("--k-first", type=int, default=10)
    parser.add_argument("--checkpoint", type=int, default=10)
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--chat-model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--model-dtype", type=str, default="bfloat16")
    parser.add_argument("--skip-token-level-replacement", action="store_true")
    parser.add_argument("--layer-step", type=int, default=1)
    args = parser.parse_args()
    print(f"using args: {args}")
    device = (
        args.device
        if args.device != "auto"
        else "cuda" if th.cuda.is_available() else "cpu"
    )
    chat_model = load_hf_model(args.chat_model, torch_dtype=args.model_dtype)
    num_layers = get_num_layers(chat_model)
    token_level_replacement = None
    if args.base_model in MODEL_CONFIGS and not args.skip_token_level_replacement:
        token_level_replacement = MODEL_CONFIGS[args.base_model][
            "token_level_replacement"
        ]
        logger.info(f"Using token level replacement: {token_level_replacement}")
    else:
        if args.base_model in MODEL_CONFIGS:
            logger.info(
                f"Skipping token level replacement for {args.base_model} as --skip-token-level-replacement flag is set"
            )
        else:
            logger.info(
                f"Skipping token level replacement for {args.base_model} as it is not in MODEL_CONFIGS"
            )
    slug = generate_slug(2)
    base_model = load_hf_model(args.base_model, torch_dtype=args.model_dtype)
    for layer in range(0, num_layers, args.layer_step):
        print(f"Running layer {layer} of {num_layers}")
        kl_experiment(
            dictionary=None,
            base_model=base_model,
            chat_model=chat_model,
            tokenizer_name=args.chat_model,
            dataset_name=args.dataset,
            split=args.split,
            dataset_col=args.dataset_col,
            latent_df=None,
            layer_to_stop=layer,
            batch_size=args.batch_size,
            device=device,
            max_seq_len=args.max_seq_len,
            log_every=args.log_every,
            k_first=args.k_first,
            checkpoint_every=args.checkpoint,
            save_path=args.save_path / (slug + args.base_model.split("/")[-1]),
            test=args.test,
            max_num_tokens=args.max_num_tokens,
            name=f"{slug}_layer_{layer}"
            + ("_skip_tok_replace" if token_level_replacement else "")
            + ("_test" if args.test else ""),
            add_coolname=False,
            token_level_replacement=token_level_replacement,
        )
