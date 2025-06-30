import sys
import argparse
from pathlib import Path
import re
import shutil
import torch as th
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm, trange

file_path = Path(__file__)
sys.path.append(str(file_path.parent.parent))
sys.path.append(str(file_path.parent.parent.parent))

from tools.tokenization_utils import patch_tokenizer


def setup_model(
    model_name: str,
    end_of_turn_token: str | None = None,
):
    """Setup model and tokenizer with proper patching."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer = patch_tokenizer(
        tokenizer,
        model_name,
        end_of_turn_token=end_of_turn_token,
    )
    if tokenizer.pad_token is None:
        raise ValueError(
            "Tokenizer has no pad token, please provide one in the tokenizer_kwargs"
        )
    if tokenizer.pad_token == tokenizer.bos_token:
        raise ValueError(
            "Pad token is the same as the start of turn token, please provide a different token or update the cleaning code"
        )
    if tokenizer.end_of_turn_token is None:
        raise ValueError(
            "Tokenizer has no end of turn token, please provide one in the tokenizer_kwargs"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=th.bfloat16,
        attn_implementation="eager" if "gemma" in model_name else None,
    )
    return tokenizer, model


def filter_dataset(dataset, tokenizer, max_text_length=1024, min_response_length=128):
    """Filter dataset based on text length constraints."""
    toks = tokenizer.apply_chat_template(dataset, add_generation_prompt=True)
    return [
        prompt
        for prompt, tok in zip(dataset, toks)
        if len(tok) <= max_text_length - min_response_length
    ]


def process_response(response: str, prompt_length: int, tokenizer):
    """Process generated response by removing padding and special tokens."""
    cleaned_response = re.sub(rf"^({re.escape(tokenizer.pad_token)})+", "", response)

    # Extract response after prompt and BOS token
    cleaned_response = cleaned_response[prompt_length:]

    is_truncated = True
    if cleaned_response.endswith(tokenizer.end_of_turn_token):
        cleaned_response = cleaned_response[: -len(tokenizer.end_of_turn_token)]
        is_truncated = False
    else:
        cleaned_response = re.sub(
            rf"{re.escape(tokenizer.pad_token)}+$", "", cleaned_response
        )

    return cleaned_response, is_truncated


def generate_responses(
    model,
    tokenizer,
    dataset,
    to_generate_tokens=200_000,
    batch_size=64,
    max_length=1024,
    device="cuda",
):
    """Generate responses using the model."""
    if to_generate_tokens != -1:
        pbar = tqdm(total=to_generate_tokens, desc="Generating data")
    generated_data = []
    truncation_status = []
    num_tokens = 0

    for i in trange(0, len(dataset), batch_size):
        batch = dataset[i : min(i + batch_size, len(dataset))]
        batch = [[conv[0]] for conv in batch]
        request_batch = [conv[0] for conv in batch]

        prompts = tokenizer.apply_chat_template(
            batch, add_generation_prompt=True, tokenize=False
        )
        prompt_lengths = [len(p) for p in prompts]

        tokenized = tokenizer.apply_chat_template(
            batch,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        min_length = min(
            len(t)
            for t in tokenizer.apply_chat_template(
                batch,
                add_generation_prompt=True,
                tokenize=True,
                truncation=True,
                max_length=max_length,
            )
        )

        out = model.generate(
            **tokenized,
            max_new_tokens=max_length - min_length,
            do_sample=False,
            stop_strings=[tokenizer.end_of_turn_token],
            tokenizer=tokenizer,
        )
        out = tokenizer.batch_decode(out, skip_special_tokens=False)
        responses = []
        for o, pl in zip(out, prompt_lengths):
            cleaned_response, is_truncated = process_response(o, pl, tokenizer)
            responses.append(cleaned_response)
            truncation_status.append(is_truncated)
        convs = [
            [request_batch[i], {"role": "assistant", "content": responses[i]}]
            for i in range(len(request_batch))
        ]
        generated_data.extend(convs)
        if to_generate_tokens != -1:
            num_generated_tokens = sum(len(t) for t in tokenizer(responses).input_ids)
            num_tokens += num_generated_tokens
            pbar.update(num_generated_tokens)

        if num_tokens >= to_generate_tokens and to_generate_tokens != -1:
            break

    return generated_data, truncation_status


def create_dataset(
    generated_data,
    original_dataset,
    truncation_status,
    gen_column_name: str,
):
    """Create final dataset with configurable column names."""
    assert (
        len(generated_data) == len(original_dataset) == len(truncation_status)
    ), f"Lengths are not equal: {len(generated_data)}, {len(original_dataset)}, {len(truncation_status)}"
    dataset_list = [
        {
            gen_column_name: gen_conv,
            "original_messages": orig_conv,
            f"{gen_column_name}_truncated": is_truncated,
        }
        for gen_conv, orig_conv, is_truncated in zip(
            generated_data, original_dataset, truncation_status
        )
    ]
    return Dataset.from_list(dataset_list)


@th.no_grad
def generate_dataset_responses(
    model_name: str,
    dataset_name: str,
    dataset_split: str = "test_sft",
    gen_column_name: str = "messages",
    to_generate_tokens: int = 200_000,
    batch_size: int = 64,
    max_length: int = 1024,
    max_text_length: int = 1024,
    min_response_length: int = 128,
    end_of_turn_token: str | None = None,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
):
    """Generate responses using language models and create a dataset."""
    # Setup model and tokenizer
    tokenizer, model = setup_model(
        model_name,
        end_of_turn_token=end_of_turn_token,
    )
    print(f"Using model: {model_name}")

    # Load and filter dataset
    if dataset_name is None:
        print("No dataset specified, please provide --dataset_name")
        return

    raw_dataset = load_dataset(dataset_name)[dataset_split]["messages"]
    print(f"Loaded dataset: {dataset_name}, split: {dataset_split}")

    filtered_dataset = filter_dataset(
        raw_dataset,
        tokenizer,
        max_text_length=max_text_length,
        min_response_length=min_response_length,
    )
    print(
        f"Filtered dataset length: {len(filtered_dataset)}, Original length: {len(raw_dataset)}"
    )

    # Generate responses
    generated_data, truncation_status = generate_responses(
        model,
        tokenizer,
        filtered_dataset,
        to_generate_tokens=to_generate_tokens,
        batch_size=batch_size,
        max_length=max_length,
    )

    # Create and save dataset
    dataset = create_dataset(
        generated_data,
        filtered_dataset,
        truncation_status,
        gen_column_name=gen_column_name,
    )

    # Create output directory if it doesn't exist
    output_path = Path("./data") / dataset_name
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_path))
    print(f"Dataset saved to: {output_path}")

    # Print summary statistics
    truncated_count = sum(truncation_status)
    print(f"Generated {len(generated_data)} responses")
    print(
        f"Truncated responses: {truncated_count} ({truncated_count/len(generated_data)*100:.1f}%)"
    )
    # Upload to HuggingFace Hub if requested
    if push_to_hub:
        hub_repo_id = hub_repo_id or dataset_name
        print(f"Uploading dataset to HuggingFace Hub: {hub_repo_id}")
        dataset.push_to_hub(hub_repo_id)
        print(
            f"Successfully uploaded to: https://huggingface.co/datasets/{hub_repo_id}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate responses using language models"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="science-of-finetuning/ultrachat_200k_generated",
        help="Dataset name to load",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="test_sft", help="Dataset split to use"
    )
    parser.add_argument(
        "--gen-column-name",
        type=str,
        default="messages",
        help="Name for the column to store generated messages",
    )
    parser.add_argument(
        "--to-generate-tokens",
        type=int,
        default=200_000,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for generation"
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=1024,
        help="Maximum text length for filtering",
    )
    parser.add_argument(
        "--min-response-length",
        type=int,
        default=128,
        help="Minimum response length for filtering",
    )
    parser.add_argument(
        "--end-of-turn-token",
        type=str,
        default=None,
        help="End of turn token for the tokenizer",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the generated dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="Repository ID for HuggingFace Hub upload (defaults to dataset_name)",
    )

    args = parser.parse_args()
    generate_dataset_responses(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        gen_column_name=args.gen_column_name,
        to_generate_tokens=args.to_generate_tokens,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_text_length=args.max_text_length,
        min_response_length=args.min_response_length,
        end_of_turn_token=args.end_of_turn_token,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
    )
