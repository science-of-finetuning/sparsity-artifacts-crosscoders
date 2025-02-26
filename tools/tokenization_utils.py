from pathlib import Path
import warnings
import torch as th
import re


class TokenizerNoCtrlTemplateProxy:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        if name == "ctrl_template":
            raise AttributeError(
                "Tokenizer was not patched with a control template, so can't be used to compute a control mask"
            )
        return getattr(self.tokenizer, name)


template_path = Path(__file__).parent.parent / "templates"
with open(template_path / "gemma_chat_template.jinja", "r") as f:
    GEMMA_CHAT_TEMPLATE = f.read()
    chat_template = GEMMA_CHAT_TEMPLATE  # for backwards compatibility
with open(template_path / "gemma_chat_template_ctrl_tokens.jinja", "r") as f:
    CTRL_TEMPLATE = f.read()
with open(template_path / "customizable_gemma_chat_template.jinja", "r") as f:
    CUSTOMIZABLE_CHAT_TEMPLATE = f.read()
with open(
    template_path / "customizable_gemma_chat_template_ctrl_tokens.jinja", "r"
) as f:
    CUSTOMIZABLE_CTRL_TEMPLATE = f.read()

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
    tokenizer, model_name: str, ctrl_template: str = None, chat_template: str = None
):
    if model_name in ("google/gemma-2-2b-it", "gemma-2"):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
        tokenizer.ctrl_template = CTRL_TEMPLATE
    else:
        if chat_template is not None:
            tokenizer.chat_template = chat_template
        if ctrl_template is None:
            warnings.warn(
                f"No control template provided, you won't be able to use the control token mask for {model_name}"
            )
            tokenizer = TokenizerNoCtrlTemplateProxy(tokenizer)
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
        original_template = CTRL_TEMPLATE
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
