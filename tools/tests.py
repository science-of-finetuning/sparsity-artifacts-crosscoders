import pytest
from transformers import AutoTokenizer
from .tokenization_utils import custom_chat_template


@pytest.mark.parametrize("ctrl_tokens", [True, False])
def test_custom_chat_template(ctrl_tokens):

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    custom_chat_template(
        tokenizer,
        ctrl_tokens=ctrl_tokens,
        start_of_turn_token="<blockquote>",
        end_of_turn_token="</blockquote>",
        user_token="dog",
        assistant_token="cat",
    )
