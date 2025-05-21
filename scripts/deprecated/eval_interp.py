import sys
from argparse import ArgumentParser
import json
from pathlib import Path
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as th
from dictionary_learning import CrossCoder
from datasets import load_from_disk
from tqdm.auto import tqdm
import einops

sys.path.append(".")

from tools.split_model import split_model


@th.inference_mode()
def evaluate_interpretation(
    base_model,
    instruct_model,
    crosscoder: CrossCoder,
    dataset,
    feature_index,
    get_predicted_mask,
    batch_size=8,
    device="cuda",
    # max_seq_len=1024,
    layer_to_stop=13,
):
    """
    Evaluate the interpretation of a feature. Return the activation of a feature over the predicted to be active tokens
    and the activation of the feature over the other tokens.

    Args:
        base_model: The base model to evaluate.
        instruct_model: The instruct model to evaluate.
        dataset: The dataset to evaluate on.
        feature_index: The index of the feature to evaluate.
        get_predicted_mask: A function that takes an input_ids tensor and returns a predicted mask.
        The mask should be a boolean tensor of shape (batch_size, seq_len) where True indicates the token is predicted to be active.
        batch_size: The batch size to use for evaluation.
        device: The device to use for evaluation.
        layer_to_stop: The layer to take the activations from.
    """
    base_model = split_model(base_model)
    instruct_model = split_model(instruct_model)
    true_activations = []
    false_activations = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        batch_tokens = instruct_model.tokenizer(
            batch,
            # return_assistant_tokens_mask=True,
            # chat_template=CHAT_TEMPLATE,
            # return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        input_ids = batch_tokens["input_ids"].to(device)
        attention_mask = batch_tokens["attention_mask"].to(device)
        # assistant_mask = th.tensor(batch_tokens["assistant_masks"]).bool().to(device)
        predicted_mask = get_predicted_mask(input_ids)
        base_activations, *_ = base_model.first_half_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_idx=layer_to_stop,
        )
        instruct_activations, *_ = instruct_model.first_half_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_idx=layer_to_stop,
        )
        cc_input = th.stack([base_activations, instruct_activations], dim=2).float()
        cc_input = einops.rearrange(cc_input, "b s l d -> (b s) l d")
        cc_activations = crosscoder.encoder(cc_input, select_features=[feature_index])
        cc_activations = einops.rearrange(
            cc_activations, "(b s) f -> b s f", b=len(batch)
        ).squeeze(2)
        attn_mask = attention_mask.bool()
        pred_act = cc_activations[predicted_mask & attn_mask]
        false_act = cc_activations[(~predicted_mask) & attn_mask]
        true_activations.append(pred_act.cpu())
        false_activations.append(false_act.cpu())
    return th.cat(true_activations), th.cat(false_activations)


def generate_questions(templates, words, langs):

    all_questions = []

    # For each template
    for template in templates:
        # Replace {{WORD}} with each word from the OK category
        for word in words:
            # Handle templates with {{NUM}}
            if "{{NUM}}" in template:
                # Generate with different numbers, e.g., 1-5
                for num in range(1, 6):
                    question = template.replace("{{WORD}}", word).replace(
                        "{{NUM}}", str(num)
                    )
                    all_questions.append(question)
            # Handle templates with {{LANG}}
            elif "{{LANG}}" in template:
                # Example languages
                for lang in langs:
                    question = template.replace("{{WORD}}", word).replace(
                        "{{LANG}}", lang
                    )
                    all_questions.append(question)
            # Simple word replacement
            else:
                question = template.replace("{{WORD}}", word)
                all_questions.append(question)

    return all_questions


# Usage:
def map_dataset(dataset, tokenizer, max_seq=1024) -> list[str]:
    if len(dataset) > 0 and isinstance(dataset[0], str):
        print("Mapping dataset to chat format")
        dataset = [[{"role": "user", "content": s}] for s in dataset]
    tokenized = tokenizer.apply_chat_template(
        dataset,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).input_ids
    return [
        s[5:]
        for s, toks in zip(
            tokenizer.apply_chat_template(
                dataset,
                tokenize=False,
                add_generation_prompt=True,
            ),
            tokenized,
        )
        if len(toks) < max_seq
    ]


def evaluate_geopo_feature_51408(args):

    def get_predicted_mask(batch_tokens, _assistant_mask=None):
        mask = th.zeros_like(batch_tokens, dtype=th.bool)
        mask[:, -4] = True
        return mask

    instruct_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=th.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=th.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    save_path = Path("results/51408_global_political")
    save_path.mkdir(exist_ok=True, parents=True)
    crosscoder = CrossCoder.from_pretrained(args.crosscoder_path, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    instruct_model.tokenizer = tokenizer
    dataset_json = json.load(open("data/geopo_feature_51408.json"))["questions"]
    datasets = {
        "local_neutral": [q["local_neutral"] for q in dataset_json],
        "local_neutral_complex": [q["local_neutral_complex"] for q in dataset_json],
        "local_political": [q["local_political"] for q in dataset_json],
        "continental_political": [q["continental_political"] for q in dataset_json],
        "global_controversial": [q["global_controversial"] for q in dataset_json],
    }
    lmsys = load_from_disk(args.chat_dataset_path)
    lmsys_q_dataset = [[conv[0]] for conv in lmsys["conversation"]]
    lmsys_q_dataset = map_dataset(lmsys_q_dataset, tokenizer)
    if args.test:
        lmsys_q_dataset = lmsys_q_dataset[:16]
    else:
        th.manual_seed(42)
        indices = th.randperm(len(lmsys_q_dataset))
        lmsys_q_dataset = [lmsys_q_dataset[i] for i in indices[:5000]]
    true_activations, false_activations = evaluate_interpretation(
        base_model,
        instruct_model,
        crosscoder,
        lmsys_q_dataset,
        51408,
        get_predicted_mask,
        batch_size=args.batch_size,
    )
    results_lmsys = {
        "mean_true_activations": true_activations.mean().item(),
        "mean_false_activations": false_activations.mean().item(),
        "max_true_activation": true_activations.max().item(),
        "max_false_activation": false_activations.max().item(),
        "min_true_activation": true_activations.min().item(),
        "true_activations": true_activations.tolist(),
        "false_activations": false_activations.tolist(),
    }
    with open(save_path / "results_lmsys.json", "w") as f:
        json.dump(results_lmsys, f)
    if args.lmsys_only:
        return

    results_glob_political = {}
    for dataset_name, dataset in datasets.items():
        if args.test:
            dataset = dataset[:16]
        dataset = map_dataset(dataset, tokenizer)
        true_activations, false_activations = evaluate_interpretation(
            base_model,
            instruct_model,
            crosscoder,
            dataset,
            51408,
            get_predicted_mask,
            batch_size=args.batch_size,
        )
        results_glob_political[dataset_name] = {
            "mean_true_activations": true_activations.mean().item(),
            "mean_false_activations": false_activations.mean().item(),
            "max_true_activation": true_activations.max().item(),
            "max_false_activation": false_activations.max().item(),
            "min_true_activation": true_activations.min().item(),
            "true_activations": true_activations.tolist(),
            "false_activations": false_activations.tolist(),
        }
    with open(save_path / "results_glob_political.json", "w") as f:
        json.dump(results_glob_political, f)

    with open("data/question_templates_51408.json", "r") as f:
        data = json.load(f)

    words = data["WORD"]
    langs = data["LANG"]
    templates = data["question-templates"]
    all_datasets = defaultdict(lambda: defaultdict(list))
    for template_name, template in tqdm(templates.items()):
        for word_category, word_list in words.items():
            questions = generate_questions(template, word_list, langs)
            all_datasets[template_name][word_category] = questions
    with open(save_path / "all_datasets.json", "w") as f:
        json.dump(all_datasets, f, indent=4)

    template_results = defaultdict(dict)
    for template_name, template in tqdm(templates.items()):
        for word_category, word_list in words.items():
            dataset = all_datasets[template_name][word_category]
            if args.test:
                dataset = dataset[:16]
            dataset = map_dataset(dataset, tokenizer)
            true_activations, false_activations = evaluate_interpretation(
                base_model,
                instruct_model,
                crosscoder,
                dataset,
                51408,
                get_predicted_mask,
                batch_size=args.batch_size,
            )
            template_results[template_name][word_category] = {
                "mean_true_activations": true_activations.mean().item(),
                "mean_false_activations": false_activations.mean().item(),
                "max_true_activation": true_activations.max().item(),
                "max_false_activation": false_activations.max().item(),
                "min_true_activation": true_activations.min().item(),
                "true_activations": true_activations.tolist(),
                "false_activations": false_activations.tolist(),
            }
    with open(save_path / "template_results.json", "w") as f:
        json.dump(template_results, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crosscoder-path", type=str, required=True)
    parser.add_argument(
        "--chat-dataset-path", type=Path, default="./datasets/test/lmsys_chat"
    )
    parser.add_argument("--lmsys-only", action="store_true", default=False)
    args = parser.parse_args()
    evaluate_geopo_feature_51408(args)
