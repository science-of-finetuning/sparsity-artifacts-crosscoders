import sys
from pathlib import Path
from typing import Callable
import json
from argparse import ArgumentParser


import torch as th
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.dictionary import CrossCoder

sys.path.append(str(Path(__file__).parent.parent))

from tools.setup_to_eval import (
    HalfStepPreprocessFn,
    IdentityPreprocessFn,
    SwitchPreprocessFn,
    CrossCoderReconstruction,
    CrossCoderSteeringFeature,
    CrossCoderOutProjection,
)
from tools.split_gemma import split_gemma


CHAT_TEMPLATE = open("templates/gemma_chat_template.jinja").read()

FEW_SHOT = [
    {"role": "user", "content": "You are a helpful assistant. What is 5+5?"},
    {"role": "assistant", "content": "10"},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "I believe it is Paris."},
]


def generate_with_intervention(
    input_ids: th.Tensor,
    attention_mask: th.Tensor,
    tokenizer: AutoTokenizer,
    base_model,
    instruct_model,
    preprocess_before_last_half_fn: HalfStepPreprocessFn,
    layer_to_stop=13,
    device="cuda",
    max_seq_len=512,
    do_sample=True,
    temperature=1.0,
    stop_condition: Callable[[th.Tensor, th.Tensor], bool] | None = None,
    max_new_tokens=100,
    all_layers_process_fn: Callable[[th.Tensor], th.Tensor] | None = None,
):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    new_tokens = 0
    while True:
        # Get base and instruct activations
        base_activations, *base_other_outputs = base_model.first_half_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_idx=layer_to_stop,
        )

        instruct_activations, *instruct_other_outputs = (
            instruct_model.first_half_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_idx=layer_to_stop,
            )
        )
        base_activations_edited, instruct_activations_edited, _ = (
            preprocess_before_last_half_fn(base_activations, instruct_activations)
        )
        final_out = None
        if base_activations_edited is not None:
            final_out = base_model.second_half_forward(
                base_activations_edited,
                *base_other_outputs,
                layer_idx=layer_to_stop,
                return_dict=True,
                all_layers_process_fn=all_layers_process_fn,
            )
        elif instruct_activations_edited is not None:
            final_out = instruct_model.second_half_forward(
                instruct_activations_edited,
                *instruct_other_outputs,
                layer_idx=layer_to_stop,
                return_dict=True,
                all_layers_process_fn=all_layers_process_fn,
            )
        logits = final_out.logits[:, -1, :]

        # next token
        if do_sample:
            next_token = th.multinomial(
                th.softmax(logits / temperature, dim=-1), num_samples=1
            )
        else:
            next_token = th.argmax(logits, dim=-1).unsqueeze(-1)

        if (
            stop_condition is not None
            and stop_condition(next_token, attention_mask)
            or next_token == tokenizer.eos_token_id
            or len(input_ids) > max_seq_len
            or new_tokens > max_new_tokens
        ):
            break

        input_ids = th.cat([input_ids, next_token], dim=-1)
        attention_mask = th.cat([attention_mask, th.ones_like(next_token)], dim=-1)
        new_tokens += 1
    return input_ids


def stop_condition_eot(next_token: th.Tensor, attention_mask: th.Tensor):
    return th.any(next_token == 107)


@th.inference_mode()
def chat_with_interventions(
    user_input: str = None,
    base_model: AutoModelForCausalLM = None,
    instruct_model: AutoModelForCausalLM = None,
    preprocess_before_last_half_fn: HalfStepPreprocessFn = None,
    layer_to_stop=13,
    device="cuda",
    max_seq_len=1024,
    max_new_tokens=200,
    max_turns=10,
    do_sample=True,
    add_few_shot=False,
    all_layers_process_fn: Callable[[th.Tensor], th.Tensor] | None = None,
):
    """Interactive chat function that applies interventions during generation."""
    assert (
        base_model is not None and instruct_model is not None
    ), "base_model and instruct_model must be provided"

    base_model = split_gemma(base_model)
    instruct_model = split_gemma(instruct_model)
    tokenizer = instruct_model.tokenizer

    conversation = []
    if add_few_shot:
        conversation.extend(FEW_SHOT)
    print("Starting chat (type 'quit' to end)")

    turn = 0
    while True:
        # Get user input
        if user_input is None:
            user_input = input("\nUser: ")
        else:
            print(f"\nUser: {user_input}")

        if user_input.lower() == "quit":
            break

        conversation.append({"role": "user", "content": user_input})

        # Tokenize conversation
        batch_tokens = tokenizer.apply_chat_template(
            [conversation],
            tokenize=True,
            return_assistant_tokens_mask=True,
            chat_template=CHAT_TEMPLATE,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )

        # Truncate to max length
        batch_tokens["input_ids"] = batch_tokens["input_ids"][:, -max_seq_len:].to(
            device
        )
        len_input_ids = batch_tokens["input_ids"].shape[1]
        output_ids = generate_with_intervention(
            input_ids=batch_tokens["input_ids"],
            attention_mask=batch_tokens["attention_mask"],
            tokenizer=tokenizer,
            base_model=base_model,
            instruct_model=instruct_model,
            preprocess_before_last_half_fn=preprocess_before_last_half_fn,
            stop_condition=stop_condition_eot,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            layer_to_stop=layer_to_stop,
            all_layers_process_fn=all_layers_process_fn,
        )
        response = tokenizer.decode(
            output_ids[0][len_input_ids:], skip_special_tokens=True
        )

        if len(response.strip()) > 0:
            print(f"\nAssistant:\n{response}")
        else:
            print("No readable response from model")
            print(
                tokenizer.decode(
                    output_ids[0][len_input_ids:], skip_special_tokens=False
                )
            )
        # Add assistant response to conversation history (using instruct model response)
        conversation.append({"role": "assistant", "content": response})
        turn += 1
        if turn >= max_turns:
            break
    return conversation


def feature_ablation(
    user_requests: list[str],
    base_model,
    instruct_model,
    preprocess_before_last_half_fn: HalfStepPreprocessFn,
    identity_preprocess_fn: HalfStepPreprocessFn,
    layer_to_stop=13,
    device="cuda",
    max_seq_len=1024,
    max_new_tokens=200,
    add_few_shot=False,
    all_layers_process_fn: Callable[[th.Tensor], th.Tensor] | None = None,
):
    results = []
    for user_request in user_requests:
        print(f"IDENTITY")
        identity_conversation = chat_with_interventions(
            user_request,
            base_model,
            instruct_model,
            identity_preprocess_fn,
            do_sample=False,
            layer_to_stop=layer_to_stop,
            device=device,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            max_turns=1,
            add_few_shot=add_few_shot,
        )
        print(f"STEERED")
        conversation = chat_with_interventions(
            user_request,
            base_model,
            instruct_model,
            do_sample=False,
            preprocess_before_last_half_fn=preprocess_before_last_half_fn,
            layer_to_stop=layer_to_stop,
            device=device,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            max_turns=1,
            add_few_shot=add_few_shot,
            all_layers_process_fn=all_layers_process_fn,
        )

        results.append(
            {
                "steered": conversation,
                "normal": identity_conversation,
            }
        )
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer-to-stop", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--feature-index", type=int, default=None)
    parser.add_argument("--use-it-only", action="store_true")
    parser.add_argument("--use-base-only", action="store_true")
    parser.add_argument("--continue-with", choices=["base", "chat"], default="chat")
    parser.add_argument("--feature-decoder", choices=["base", "chat"], default="base")
    parser.add_argument("--activation", choices=["base", "chat"], default="chat")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--ignore-encoder", action="store_true")
    parser.add_argument("--feature-ablation-file", type=Path, default=None)
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--max-new-tokens", "-MNT", type=int, default=200)
    parser.add_argument("--switch", action="store_true")
    parser.add_argument("--layer-sweep", action="store_true")
    parser.add_argument(
        "--dataset-path", type=str, default="./datasets/test/lmsys_chat"
    )
    parser.add_argument("--crosscoder-path", type=str, required=True)
    parser.add_argument(
        "--feature-df-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--it-only-feature-list-path", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-path", type=Path, default=Path("feature_ablation_results")
    )
    parser.add_argument("--all-layers", "-AL", action="store_true")
    parser.add_argument("--project-out", "-PO", action="store_true")

    args = parser.parse_args()

    if args.layer_sweep:
        assert (
            args.feature_ablation_file is not None
        ), "Feature ablation file must be provided for layer sweep"
        assert args.switch, "Layer sweep requires switch"

    instruct_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=th.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    instruct_model.tokenizer = tokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map="cuda",
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
    )
    dataset = load_from_disk(args.dataset_path)
    if args.test:
        dataset = dataset.select(range(100))
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

    features_indices = []
    if args.feature_index is not None:
        features_indices.append(args.feature_index)
    else:
        if args.use_it_only:
            features_indices.extend(it_only_features)
        if args.use_base_only:
            features_indices.extend(base_only_features)

    id_fun = IdentityPreprocessFn(continue_with=args.continue_with)

    if args.switch:
        print(f"Switching to {args.continue_with}")
        id_fun = IdentityPreprocessFn(
            continue_with="chat" if args.continue_with == "base" else "base"
        )
        fun = SwitchPreprocessFn(continue_with=args.continue_with)
    elif len(features_indices) != 0:
        print("Steering Args:")
        print(f"Activation: {args.activation}")
        print(f"Feature Decoder: {args.feature_decoder}")
        print(f"Features to steer: {features_indices}")
        print(f"Continue with: {args.continue_with}")
        print(f"Scale: {args.scale}")
        print(f"Ignore encoder: {args.ignore_encoder}")
        print(f"All layers: {args.all_layers}")
        print(f"Project out: {args.project_out}")
        if args.project_out:
            fun = CrossCoderOutProjection(
                crosscoder,
                steer_activations_of=args.activation,
                steer_with_features_from=args.feature_decoder,
                features_to_steer=features_indices,
                continue_with=args.continue_with,
                scale_steering_feature=args.scale,
            )
        else:
            fun = CrossCoderSteeringFeature(
                crosscoder,
                steer_activations_of=args.activation,
                steer_with_features_from=args.feature_decoder,
                features_to_steer=features_indices,
                continue_with=args.continue_with,
                scale_steering_feature=args.scale,
                ignore_encoder=args.ignore_encoder,
            )
        print(f"STEERING FEATURES: {features_indices}")
    else:
        print(f"No steering features, continuing with {args.continue_with}")
        fun = IdentityPreprocessFn(continue_with=args.continue_with)

    if args.feature_ablation_file is not None:
        with open(args.feature_ablation_file) as f:
            user_requests = json.load(f)
        if args.layer_sweep:
            all_results = {}
            for layer_to_stop in range(1, base_model.config.num_hidden_layers + 1):
                results = feature_ablation(
                    user_requests,
                    base_model,
                    instruct_model,
                    fun,
                    id_fun,
                    device=device,
                    max_seq_len=args.max_seq_len,
                    max_new_tokens=200,
                    add_few_shot=args.few_shot,
                    all_layers_process_fn=fun.all_layers if args.all_layers else None,
                    layer_to_stop=layer_to_stop,
                )
                all_results[layer_to_stop] = results
            # name of the file is the name of the feature ablation file
            postfix = f"{args.feature_index}"
            if args.continue_with == "base":
                postfix += "_to_base"
            if args.scale != 1.0:
                postfix += f"_scale_{args.scale}"
            if args.all_layers:
                postfix += "_all_layers"
            if args.project_out:
                postfix += "_project_out"
            if args.switch:
                postfix += "_switch"
            with open(
                args.save_path
                / f"sweep_{args.feature_ablation_file.stem}_ablation_{postfix}.json",
                "w",
            ) as f:
                json.dump(all_results, f, indent=4)
        else:
            results = feature_ablation(
                user_requests,
                base_model,
                instruct_model,
                fun,
                id_fun,
                layer_to_stop=args.layer_to_stop,
                device=device,
                max_seq_len=args.max_seq_len,
                max_new_tokens=200,
                add_few_shot=args.few_shot,
                all_layers_process_fn=fun.all_layers if args.all_layers else None,
            )
            # name of the file is the name of the feature ablation file
            postfix = f"{args.feature_index}"
            if args.continue_with == "base":
                postfix += "_to_base"
            if args.scale != 1.0:
                postfix += f"_scale_{args.scale}"
            if args.all_layers:
                postfix += "_all_layers"
            if args.project_out:
                postfix += "_project_out"
            if args.switch:
                postfix += "_switch"
        with open(
            args.save_path
            / f"{args.feature_ablation_file.stem}_ablation_{postfix}.json",
            "w",
        ) as f:
            json.dump(results, f, indent=4)
    else:
        results = chat_with_interventions(
            base_model,
            instruct_model,
            preprocess_before_last_half_fn=fun,
            layer_to_stop=args.layer_to_stop,
            device=device,
            max_seq_len=args.max_seq_len,
            add_few_shot=args.few_shot,
            all_layers_process_fn=fun.all_layers if args.all_layers else None,
        )
