import sys

sys.path.append(".")

from argparse import ArgumentParser
from dlabutils import model_path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as th
import json
from collections import defaultdict
from evaluate import evaluate_interpretation
from dictionary_learning import CrossCoder
from datasets import load_from_disk
from tqdm.auto import tqdm
from string import Template
import json
from pathlib import Path


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
        model_path("google/gemma-2-2b-it"),
        torch_dtype=th.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path("google/gemma-2-2b"),
        torch_dtype=th.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    save_path = Path("results/51408_global_political")
    save_path.mkdir(exist_ok=True, parents=True)
    crosscoder = CrossCoder.from_pretrained(args.crosscoder_path, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path("google/gemma-2-2b-it"))
    instruct_model.tokenizer = tokenizer
    dataset_json = json.load(open("data/geopo_feature_51408.json"))["questions"]
    datasets = {
        "local_neutral": [q["local_neutral"] for q in dataset_json],
        "local_neutral_complex": [q["local_neutral_complex"] for q in dataset_json],
        "local_political": [q["local_political"] for q in dataset_json],
        "continental_political": [q["continental_political"] for q in dataset_json],
        "global_controversial": [q["global_controversial"] for q in dataset_json],
    }
    lmsys = load_from_disk(
        "/dlabscratch1/jminder/repositories/representation-structure-comparison/datasets/test/lmsys_chat"
    )
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
    parser.add_argument(
        "--crosscoder-path",
        type=str,
        default="/dlabscratch1/jminder/repositories/representation-structure-comparison/checkpoints/l13-mu4.1e-02-lr1e-04/ae_final.pt",
    )
    parser.add_argument("--lmsys-only", action="store_true", default=False)
    args = parser.parse_args()
    evaluate_geopo_feature_51408(args)
