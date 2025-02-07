import sys

sys.path.append("..")

from pathlib import Path
import torch as th
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm, trange
from huggingface_hub import hf_api
from tempfile import NamedTemporaryFile


def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it", padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        device_map="auto",
        torch_dtype=th.bfloat16,
        attn_implementation="eager",
    )
    return tokenizer, model


def filter_dataset(dataset, tokenizer, max_text_length=1024, min_response_length=128):
    toks = tokenizer.apply_chat_template(dataset, add_generation_prompt=True)
    return [
        prompt
        for prompt, tok in zip(dataset, toks)
        if len(tok) <= max_text_length - min_response_length
    ]


def process_response(response, prompt_length):
    cleaned_response = response.replace("<pad>", "")[prompt_length + 5 :]
    is_truncated = True
    if cleaned_response.endswith("<end_of_turn>"):
        cleaned_response = cleaned_response[:-12]
        is_truncated = False
    return cleaned_response, is_truncated


def generate_responses(
    model,
    tokenizer,
    dataset,
    to_generate_tokens=200_000,
    batch_size=64,
    max_length=1024,
):
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

        tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
        min_length = min(
            len(t)
            for t in tokenizer(prompts, truncation=True, max_length=max_length)[
                "input_ids"
            ]
        )

        out = model.generate(
            **tokenized,
            max_new_tokens=max_length - min_length,
            do_sample=False,
            stop_strings=["<end_of_turn>"],
            tokenizer=tokenizer,
        )
        out = tokenizer.batch_decode(out, skip_special_tokens=False)
        responses = []
        for o, pl in zip(out, prompt_lengths):
            cleaned_response, is_truncated = process_response(o, pl)
            responses.append(cleaned_response)
            truncation_status.append(is_truncated)
        convs = [
            [request_batch[i], {"role": "assistant", "content": responses[i]}]
            for i in range(len(request_batch))
        ]
        generated_data.extend(convs)

        num_generated_tokens = sum(len(t) for t in tokenizer(responses).input_ids)
        num_tokens += num_generated_tokens
        pbar.update(num_generated_tokens)

        if num_tokens >= to_generate_tokens:
            break

    return generated_data, truncation_status


def create_dataset(generated_data, dataset, truncation_status):
    assert len(generated_data) == len(dataset) == len(truncation_status)
    dataset_list = [
        {
            "messages": gen_conv,
            "original_messages": orig_conv,
            "truncated": is_truncated,
        }
        for gen_conv, orig_conv, is_truncated in zip(
            generated_data, dataset, truncation_status
        )
    ]
    return Dataset.from_list(dataset_list)


@th.no_grad
def main():
    tokenizer, model = setup_model()

    raw_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")["test_sft"]["messages"]
    filtered_dataset = filter_dataset(raw_dataset, tokenizer)
    print(
        f"Filtered dataset length: {len(filtered_dataset)}, Original length: {len(raw_dataset)}"
    )

    generated_data, truncation_status = generate_responses(
        model, tokenizer, filtered_dataset
    )
    dataset = create_dataset(generated_data, filtered_dataset, truncation_status)

    dataset.save_to_disk("../data/ultrachat_200k_gemma-2-2b-it-generated")


if __name__ == "__main__":
    main()
