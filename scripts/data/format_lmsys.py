# %%
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from tools.configs import HF_NAME
source_dataset = f"{HF_NAME}/lmsys-chat-1m-chat-formatted"
target_dataset = f"{HF_NAME}/lmsys-chat-1m-chat-formatted"


# %%
def apply_connor_chat_template(conv):
    return (
        "\n".join(
            [
                ("Assistant: " if msg["role"] == "assistant" else "User: ")
                + msg["content"]
                for msg in conv
            ]
        )
        + "\n"
    )


# %% [markdown]
# ## Make dataset
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source-dataset", type=str, default=source_dataset)
    parser.add_argument("--target-dataset", type=str, default=target_dataset)
    parser.add_argument("--tokenizer", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--name", "-n", required=True)
    args = parser.parse_args()
    # %%
    tokenizer = None
    if args.tokenizer != "connor":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    ds = load_dataset(args.source_dataset)

    # %%
    formated_conversations = defaultdict(list)
    for split in ds.keys():
        dsl = ds[split]["conversation"]
        for i in tqdm(range(0, len(dsl), 100), desc=f"Processing {split}"):
            conversations = dsl[i : min(i + 100, len(dsl))]
            if args.tokenizer == "connor":
                formated_conversations[split].extend(
                    apply_connor_chat_template(c) for c in conversations
                )
            else:
                if tokenizer.bos_token is None:
                    formated_conversations[split].extend(
                        tokenizer.apply_chat_template(conversations, tokenize=False)
                    )
                else:
                    tokens = tokenizer.apply_chat_template(conversations, tokenize=True)
                    assert all(t[0] == tokenizer.bos_token_id for t in tokens)
                    tokens = [t[1:] for t in tokens]
                    formated_conversations[split].extend(tokenizer.batch_decode(tokens))

    # %%
    # create a new dataset
    ds_formatted = ds
    for split in ds.keys():
        ds_formatted[split] = ds_formatted[split].add_column(
            f"text_{args.name}", formated_conversations[split]
        )
    # Save the new dataset to disk
    ds_formatted.push_to_hub(args.target_dataset)
