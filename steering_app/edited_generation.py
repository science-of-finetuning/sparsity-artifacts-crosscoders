from pathlib import Path
import sys
import torch as th
from nnterp.nnsight_utils import get_layer, get_layer_output
from nnsight import LanguageModel
import nnsight as nns

sys.path.append(str(Path(__file__).parent.parent))


from tools.tokenization_utils import tokenize_with_ctrl_ids


def _generate_with_model(
    model: LanguageModel,
    tokenizer,
    layer: int,
    conversation: list,
    continuations: list,
    cont_idx: list,
    base_acts: th.Tensor,
    chat_acts: th.Tensor,
    max_new_tokens: int,
    is_chat_model: bool,
    only_patch_context: bool,
) -> dict:
    """Helper function to generate outputs with either chat or base model."""
    if not continuations:
        return {}

    model_input = tokenizer.apply_chat_template(
        [conversation] * len(continuations),
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if only_patch_context:
        patched_acts = []
        for inter in continuations:
            base_cont, chat_cont = inter.preprocess(base_acts, chat_acts)
            cont = chat_cont if is_chat_model else base_cont
            assert (base_cont if is_chat_model else chat_cont) is None
            patched_acts.append(cont)

        patched_acts = th.cat(patched_acts, dim=0)

    with model.generate(
        model_input,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stop_strings=["<end_of_turn>"],
        tokenizer=tokenizer,
    ):
        if only_patch_context:
            get_layer(model, layer).output = (patched_acts,)
        else:
            with model.model.all():
                acts = get_layer_output(model, layer)
                for i, inter in enumerate(continuations):
                    patched_acts = inter.process_single_activation(acts[i])
                    get_layer_output(model, layer)[i] = patched_acts

        out = model.generator.output.save()

    return {
        i: model.tokenizer.decode(o, skip_special_tokens=False).replace(
            tokenizer.pad_token, ""
        )
        for o, i in zip(out, cont_idx)
    }


def generate_with_interventions(
    base_model: LanguageModel,
    chat_model: LanguageModel,
    layer,
    crosscoder_device,
    interventions,
    input_text,
    max_new_tokens=512,
    only_patch_context=False,
):
    """
    Generate text with interventions applied to model activations.

    This function applies a list of interventions to the activations of a specified layer
    in either a base model or a chat model, then generates text using the modified activations.

    Args:
        base_model (LanguageModel): The base language model.
        chat_model (LanguageModel): The chat-tuned language model.
        layer: The layer where interventions will be applied.
        crosscoder_device: The device where cross-model operations will be performed.
        interventions: List of intervention objects that modify activations.
        input_text (str): The input text prompt.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.

    Returns:
        dict: A list of generated outputs for each intervention.
    """

    outputs = {}
    tokenizer = chat_model.tokenizer
    chat_continuations = [
        inter for inter in interventions if inter.continue_with == "chat"
    ]
    base_continuations = [
        inter for inter in interventions if inter.continue_with == "base"
    ]
    assert len(chat_continuations) + len(base_continuations) == len(
        interventions
    ), "All interventions must have a continue_with value"
    cont_chat_idx = []
    cont_base_idx = []
    for i, inter in enumerate(interventions):
        if inter.continue_with == "chat":
            cont_chat_idx.append(i)
        else:
            cont_base_idx.append(i)
    assert len(cont_chat_idx) == len(chat_continuations)
    assert len(cont_base_idx) == len(base_continuations)
    conversation = [{"role": "user", "content": input_text}]
    input = tokenize_with_ctrl_ids(conversation, tokenizer, add_generation_prompt=True)
    # TODO: Add an option to use crosscoder reconstruction (needs for loop of traces with caches)
    with th.no_grad():
        chat_acts = None
        base_acts = None
        if only_patch_context:
            with base_model.trace(input):
                base_acts = (
                    get_layer_output(base_model, layer).to(crosscoder_device).save()
                )
                get_layer(base_model, layer).output.stop()
            with chat_model.trace(input):
                chat_acts = (
                    get_layer_output(chat_model, layer).to(crosscoder_device).save()
                )
                get_layer(chat_model, layer).output.stop()

        # Generate with chat model
        outputs = _generate_with_model(
            chat_model,
            tokenizer,
            layer,
            conversation,
            chat_continuations,
            cont_chat_idx,
            base_acts,
            chat_acts,
            max_new_tokens,
            is_chat_model=True,
            only_patch_context=only_patch_context,
        )

        # Generate with base model
        outputs.update(
            _generate_with_model(
                base_model,
                tokenizer,
                layer,
                conversation,
                base_continuations,
                cont_base_idx,
                base_acts,
                chat_acts,
                max_new_tokens,
                is_chat_model=False,
                only_patch_context=only_patch_context,
            )
        )

        out_list = [outputs[i] for i in range(len(interventions))]
        return out_list
