from pathlib import Path
import warnings
import torch as th
import re

from loguru import logger
from transformers import AutoTokenizer


class IncompleteTokenizerProxy:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        if name == "ctrl_template" and self.tokenizer.ctrl_template is None:
            raise AttributeError(
                "Tokenizer was not patched using tools.tokenization_utils.patch_tokenizer with a control template, so can't be used to compute a control mask"
            )
        elif (
            name
            in [
                "start_of_turn_token",
                "end_of_turn_token",
                "start_of_turn_token_id",
                "end_of_turn_token_id",
            ]
            and getattr(self.tokenizer, name) is None
        ):
            raise AttributeError(
                f"Tokenizer was not patched using tools.tokenization_utils.patch_tokenizer with a {name}."
            )
        return getattr(self.tokenizer, name)

    def __setattr__(self, name, value):
        if name == "tokenizer":
            super().__setattr__(name, value)
            return
        return setattr(self.tokenizer, name, value)


template_path = Path(__file__).parent.parent / "templates"
with open(template_path / "gemma_chat_template.jinja", "r") as f:
    GEMMA_CHAT_TEMPLATE = f.read()
    chat_template = GEMMA_CHAT_TEMPLATE  # for backwards compatibility
with open(template_path / "llama3.1_chat_template.jinja", "r") as f:
    LLAMA3_1_CHAT_TEMPLATE = f.read()
with open(template_path / "gemma_chat_template_ctrl_tokens.jinja", "r") as f:
    GEMMA_CTRL_TEMPLATE = f.read()
with open(template_path / "llama3.1_chat_template_ctrl_tokens.jinja", "r") as f:
    LLAMA3_1_CTRL_TEMPLATE = f.read()
with open(template_path / "customizable_gemma_chat_template.jinja", "r") as f:
    CUSTOMIZABLE_CHAT_TEMPLATE = f.read()
with open(
    template_path / "customizable_gemma_chat_template_ctrl_tokens.jinja", "r"
) as f:
    CUSTOMIZABLE_CTRL_TEMPLATE = f.read()
GEMMA_TOKENIZER = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
GEMMA_START_OF_TURN_TOKEN_ID = GEMMA_TOKENIZER.encode(
    "<start_of_turn>", add_special_tokens=False
)[0]
GEMMA_END_OF_TURN_TOKEN_ID = GEMMA_TOKENIZER.encode(
    "<end_of_turn>", add_special_tokens=False
)[0]

sample_batch = [
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "My name is Gemma."},
    ],
    [
        {"role": "user", "content": "Hello, my name is John."},
        {"role": "assistant", "content": "Hello, John. How can I help you today?"},
        {"role": "user", "content": "I'd like to know more about your features."},
        {"role": "assistant", "content": "I'm sorry, I can't answer that question."},
        {"role": "user", "content": "What is your favorite color?"},
        {"role": "assistant", "content": "My favorite color is blue."},
    ],
]


def patch_tokenizer(
    tokenizer,
    model_name: str,
    ctrl_template: str | None = None,
    chat_template: str | None = None,
    end_of_turn_token: str | None = None,
    start_of_turn_token: str | None = None,
    pad_token: str | None = None,
):
    if "gemma-2" in model_name.lower():
        if chat_template is None:
            chat_template = GEMMA_CHAT_TEMPLATE
        if ctrl_template is None:
            ctrl_template = GEMMA_CTRL_TEMPLATE
        if start_of_turn_token is None:
            start_of_turn_token = "<start_of_turn>"
        if end_of_turn_token is None:
            end_of_turn_token = "<end_of_turn>"
    elif (
        "meta-llama/Meta-Llama-3.1".lower() in model_name.lower()
        or "meta-llama/Llama-3.2".lower() in model_name.lower()
    ):
        if chat_template is None:
            chat_template = LLAMA3_1_CHAT_TEMPLATE
        if ctrl_template is None:
            ctrl_template = LLAMA3_1_CTRL_TEMPLATE
        if end_of_turn_token is None:
            end_of_turn_token = "<|eot_id|>"
        if start_of_turn_token is None:
            start_of_turn_token = "<|start_header_id|>"
        if pad_token is None:
            pad_token = "<|end_of_text|>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = pad_token or tokenizer.eos_token or tokenizer.bos_token
        if pad_token is not None:
            pad_token_id = tokenizer.encode(pad_token, add_special_tokens=False)
            if len(pad_token_id) != 1:
                raise ValueError(f"Pad token must be a single token: {pad_token_id}")
            tokenizer.pad_token_id = pad_token_id[0]
        else:
            tokenizer.pad_token_id = (
                tokenizer.eos_token_id
                if tokenizer.eos_token is not None
                else tokenizer.bos_token_id
            )
        if tokenizer.pad_token_id is None:
            raise ValueError("Pad token couldn't be set automatically")
    use_proxy = False
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    if ctrl_template is None:
        warnings.warn(
            f"No control template provided, you won't be able to use the control token mask for {model_name}"
        )
        tokenizer.ctrl_template = None
        use_proxy = True
    else:
        tokenizer.ctrl_template = ctrl_template
    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat template, please provide one in the tokenizer_kwargs"
        )
    generation_pattern = re.compile(r"\{%\s*generation\s*%\}")
    if not generation_pattern.search(tokenizer.chat_template):
        raise ValueError(
            f"Chat template for {model_name}"
            " does not contain {% generation %} keyword"
        )
    tokenizer.end_of_turn_token = end_of_turn_token
    if end_of_turn_token is None:
        warnings.warn(
            "No end of turn token provided, you won't be able to use tokenizer.end_of_turn_token"
        )
        use_proxy = True
        tokenizer.end_of_turn_token_id = None
    else:
        id = tokenizer.encode(end_of_turn_token, add_special_tokens=False)
        if len(id) != 1:
            raise ValueError(f"end of turn token must be a single token: {id}")
        tokenizer.end_of_turn_token_id = id[0]
    tokenizer.start_of_turn_token = start_of_turn_token
    if start_of_turn_token is None:
        warnings.warn(
            "No start of turn token provided, you won't be able to use tokenizer.start_of_turn_token"
        )
        use_proxy = True
        tokenizer.start_of_turn_token_id = None
    else:
        id = tokenizer.encode(start_of_turn_token, add_special_tokens=False)
        if len(id) != 1:
            raise ValueError(f"start of turn token must be a single token: {id}")
        tokenizer.start_of_turn_token_id = id[0]

    if use_proxy:
        tokenizer = IncompleteTokenizerProxy(tokenizer)
    return tokenizer


def sanitize(tok):
    return tok.replace("\r", "\\r").replace("\n", "\\n")


def tokenize_with_ctrl_mask(
    convs: list[list[dict[str, str]]],
    tokenizer,
    **tokenizer_kwargs,
) -> dict:
    """
    Tokenizes conversations with a control mask indicating chat control tokens.

    This function tokenizes a list of conversations using a custom template for chat control tokens. It returns a dictionary containing the tokenized conversations, attention mask, control mask, and assistant masks.

    Args:
        convs (list[list[dict[str, str]]]): A list of conversations, where each conversation is a list of dictionaries containing 'role' and 'content'.
        tokenizer: The tokenizer to use for tokenization.
        **tokenizer_kwargs: Additional keyword arguments to pass to the tokenizer.

    Returns:
        dict: A dictionary containing the tokenized conversations, attention_mask, ctrl_mask, and assistant_masks
    """
    # Update tokenizer_kwargs with default settings for control mask and attention mask
    kwargs = tokenizer_kwargs.copy()
    kwargs.update(
        dict(
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            return_dict=True,
            chat_template=tokenizer.ctrl_template,
        )
    )
    ctrl_tok_dict = tokenizer.apply_chat_template(
        convs,
        **kwargs,
    )
    ctrl_mask = th.tensor(ctrl_tok_dict["assistant_masks"], dtype=th.bool)
    tokenizer_kwargs["return_dict"] = True
    tokenizer_kwargs["return_assistant_tokens_mask"] = True
    tokenizer_kwargs["return_tensors"] = "pt"
    if tokenizer.chat_template is None and "chat_template" not in tokenizer_kwargs:
        raise ValueError(
            "Tokenizer has no chat template, please provide one in the tokenizer_kwargs"
        )
    else:
        chat_template = tokenizer_kwargs.get("chat_template", tokenizer.chat_template)
        if "generation" not in chat_template:
            raise ValueError("Chat template does not contain {% generation %} keyword")
    tok_dict = tokenizer.apply_chat_template(convs, **tokenizer_kwargs)
    if tok_dict["attention_mask"].shape != ctrl_tok_dict["attention_mask"].shape:
        raise ValueError(
            f"attention_mask shapes are not the same: {tok_dict['attention_mask'].shape} != {ctrl_tok_dict['attention_mask'].shape}\n"
            "This means your chat template is not the same as the control template"
        )
    tok_dict["ctrl_mask"] = ctrl_mask
    tok_dict["assistant_masks"] = th.tensor(tok_dict["assistant_masks"], dtype=th.bool)
    return tok_dict


def tokenize_with_ctrl_ids(
    convs: list[list[dict[str, str]]],
    tokenizer,
    **tokenizer_kwargs,
) -> dict:
    """
    Same as tokenize_with_ctrl_mask, but labels the control tokens from 1 to 10 instead all True
    """
    tok_dict = tokenize_with_ctrl_mask(convs, tokenizer, **tokenizer_kwargs)
    mask = tok_dict["ctrl_mask"]
    ids = mask.to(th.int)
    n_ctrl_toks = ids.sum()
    rep_1_10 = th.arange(1, 11, dtype=th.int).repeat(n_ctrl_toks // 10 + 1)[
        :n_ctrl_toks
    ]
    ids[mask] = rep_1_10
    tok_dict["ctrl_ids"] = ids
    return tok_dict


def custom_chat_template(
    tokenizer,
    *,
    start_of_turn_token="<start_of_turn>",
    end_of_turn_token="<end_of_turn>",
    user_token="user",
    assistant_token="model",
    enforce_length=True,
    ctrl_tokens=False,
):
    """
    Create a custom chat template with alternative tokens

    Args:
        tokenizer: The tokenizer to use
        start_of_turn_token: The token to use for the start of a turn
        end_of_turn_token: The token to use for the end of a turn
        user_token: The token to use for the user name
        assistant_token: The token to use for the assistant name
        enforce_length: Whether to enforce that the tokens are single tokens

    Returns:
        The custom chat template, to be used with tokenizer.apply_chat_template(..., chat_template=chat_template)
    """
    if enforce_length:
        assert (
            len(tokenizer.tokenize(start_of_turn_token)) == 1
        ), "start_of_turn_token must be a single token"
        assert (
            len(tokenizer.tokenize(end_of_turn_token)) == 1
        ), "end_of_turn_token must be a single token"
        assert (
            len(tokenizer.tokenize(user_token)) == 1
        ), "user_token must be a single token"
        assert (
            len(tokenizer.tokenize(assistant_token)) == 1
        ), "assistant_token must be a single token"
    if ctrl_tokens:
        template = CUSTOMIZABLE_CTRL_TEMPLATE
    else:
        template = CUSTOMIZABLE_CHAT_TEMPLATE
    template = (
        template.replace("<start_of_turn>", sanitize(start_of_turn_token))
        .replace("<end_of_turn>", sanitize(end_of_turn_token))
        .replace("<user>", sanitize(user_token))
        .replace("model", sanitize(assistant_token))
    )
    if ctrl_tokens:
        original_template = GEMMA_CTRL_TEMPLATE
    else:
        original_template = GEMMA_CHAT_TEMPLATE
    if enforce_length:
        tokenized = tokenizer.apply_chat_template(
            sample_batch,
            chat_template=original_template,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        custom_tokenized = tokenizer.apply_chat_template(
            sample_batch,
            chat_template=template,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        gt_inp_len = list(map(len, tokenized["input_ids"]))
        custom_inp_len = list(map(len, custom_tokenized["input_ids"]))
        assert (
            gt_inp_len == custom_inp_len
        ), f"input_ids lens are not the same: {gt_inp_len} != {custom_inp_len}"
        assert (
            tokenized["assistant_masks"] == custom_tokenized["assistant_masks"]
        ), f"assistant_masks are not the same: {tokenized['assistant_masks']} != {custom_tokenized['assistant_masks']}"
    return template


def tokens_to_conv(
    tokens: list[int],
    tokenizer: AutoTokenizer,
    skip_n_after_sot: int = 0,
    skip_n_after_eot: int = 0,
) -> list[dict[str, str]]:
    """
    Convert a list of tokens to a conversation.

    Assumes:
    - No system prompt
    - First message is user
    - Tokens are tokenized with the tokenizer.apply_chat_template method
    """
    original_tokens = tokens
    if tokens[0] == tokenizer.bos_token_id:
        tokens = tokens[1:]
    assert (
        tokens[0] == tokenizer.start_of_turn_token_id
    ), f"Expected start of turn token after bos token, got {tokens[0]}: {tokenizer.convert_ids_to_tokens(tokens[0])} of\n```{tokenizer.decode(tokens)}```"
    conversation = []
    role = "user"
    role_switch = {"user": "assistant", "assistant": "user"}
    eot_indexs = [
        i for i, t in enumerate(tokens) if t == tokenizer.end_of_turn_token_id
    ]
    last_index = 1 + skip_n_after_sot  # skip the first start of turn tokens
    for i, eot_index in enumerate(eot_indexs):
        if last_index == eot_index:
            conversation.append({"role": role, "content": ""})
        else:
            assert (
                last_index < eot_index
            ), f"last_index {last_index} is after eot_index {eot_index}"
            turn_tokens = tokens[last_index:eot_index]
            conversation.append(
                {"role": role, "content": tokenizer.decode(turn_tokens)}
            )
        last_index = eot_index + skip_n_after_eot + 1
        if last_index < len(tokens):
            assert (
                tokens[last_index] == tokenizer.start_of_turn_token_id
            ), f"Expected start of turn token after end of turn token, got {tokens[last_index]}: {tokenizer.convert_ids_to_tokens(tokens[last_index])}"
        last_index += skip_n_after_sot + 1
        role = role_switch[role]
    if tokens[-1] != tokenizer.end_of_turn_token_id:
        conversation.append(
            {
                "role": role,
                "content": tokenizer.decode(tokens[last_index:])
                + tokenizer.end_of_turn_token,
            }
        )  # add <end_of_turn> to protect \n from being trimmed when applying the chat template
    assert (
        len(conversation) > 0
    ), f"Conversation is empty for tokens {tokenizer.decode(original_tokens)}"
    if tokenizer.apply_chat_template(conversation, tokenize=False) != tokenizer.decode(
        original_tokens
    ):
        # Compare strings char by char and show context when they differ
        str1 = tokenizer.apply_chat_template(conversation, tokenize=False)
        str2 = tokenizer.decode(original_tokens)
        for i, (c1, c2) in enumerate(zip(str1, str2)):
            if c1 != c2:
                start = max(0, i - 20)
                end = min(len(str1), i + 20)
                context = str1[start:end]
                print(f"Strings differ at position {i}")
                print(
                    f"Context: {context[:i-start]}\033[91m{c1}\033[0m{context[i-start+1:]}"
                )
                print(f"Expected: {str2[start:i]}\033[91m{c2}\033[0m{str2[i+1:end]}")
                raise ValueError("Decoded conversation does not match original string")
    tokens1 = tokenizer.apply_chat_template(conversation)
    if len(tokens1) < len(original_tokens):
        # Show missing tokens in red
        print("Tokenized chat is shorter than original")
        print("Missing tokens:")
        print(
            "\033[91m"
            + " ".join(tokenizer.convert_ids_to_tokens(original_tokens[len(tokens1) :]))
            + "\033[0m"
        )
        raise ValueError("Tokenized chat is shorter than original")

    if tokens1[: len(original_tokens)] != original_tokens:
        # Compare tokens one by one and show context when they differ
        tokens2 = original_tokens
        for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
            if t1 != t2:
                start = max(0, i - 5)
                end = min(len(tokens1), i + 5)
                context1 = tokenizer.convert_ids_to_tokens(tokens1[start:end])
                context2 = tokenizer.convert_ids_to_tokens(tokens2[start:end])
                print(f"Tokens differ at position {i}")
                print(
                    f"Context: {context1[:i-start]}\033[91m{tokenizer.convert_ids_to_tokens([t1])[0]}\033[0m{context1[i-start+1:]}"
                )
                print(
                    f"Expected: {context2[:i-start]}\033[91m{tokenizer.convert_ids_to_tokens([t2])[0]}\033[0m{context2[i-start+1:]}"
                )
                raise ValueError(
                    "Tokenized conversation does not match original tokens"
                )
        raise ValueError("Conversation does not match tokens")
    return conversation


def gemma_tokens_to_conv(
    tokens: list[int],
    tokenizer: AutoTokenizer | None = None,
):
    if tokenizer is None:
        tokenizer = patch_tokenizer(
            AutoTokenizer.from_pretrained("google/gemma-2-2b-it"),
            model_name="google/gemma-2-2b-it",
        )
    return tokens_to_conv(
        tokens,
        tokenizer,
        skip_n_after_sot=2,
        skip_n_after_eot=1,
    )
