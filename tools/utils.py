import warnings
from pathlib import Path

import torch as th


from tools.compute_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tools.cc_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tools.plotting_utils import *  # pylint: disable=unused-wildcard-import,wildcard-import

template_path = (
    Path(__file__).parent.parent / "templates" / "gemma_chat_template_ctrl_tokens.jinja"
)
chat_template_path = (
    Path(__file__).parent.parent / "templates" / "gemma_chat_template.jinja"
)
with open(template_path, "r") as f:
    ctrl_template = f.read()
with open(chat_template_path, "r") as f:
    chat_template = f.read()


def tokenize_with_ctrl_mask(
    convs: list[list[dict[str, str]]],
    tokenizer,
    **tokenizer_kwargs,
) -> dict:
    """
    Create a mask that is 1 for chat control tokens and 0 for other tokens
    """
    kwargs = tokenizer_kwargs.copy()
    kwargs.update(
        dict(
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            return_dict=True,
            chat_template=ctrl_template,
        )
    )
    ctrl_mask = th.tensor(
        tokenizer.apply_chat_template(
            convs,
            **kwargs,
        )["assistant_masks"],
        dtype=th.bool,
    )
    if "chat_template" in tokenizer_kwargs:
        warnings.warn("chat_template is already set in tokenizer_kwargs, ignoring it")
    tokenizer_kwargs["chat_template"] = chat_template
    tokenizer_kwargs["return_dict"] = True
    tokenizer_kwargs["return_assistant_tokens_mask"] = True
    tok_dict = tokenizer.apply_chat_template(convs, **tokenizer_kwargs)
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
