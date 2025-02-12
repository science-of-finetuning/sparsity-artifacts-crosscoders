import torch
from typing import Optional, Union, Tuple, List
from transformers.models.gemma.modeling_gemma import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    GemmaForCausalLM,
)
from transformers.cache_utils import Cache as HybridCache
from loguru import logger
from torch.nn import CrossEntropyLoss
from types import MethodType
from typing import Callable


def model_first_half_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    layer_idx: int = 13,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    # embed positions
    hidden_states = inputs_embeds

    # normalized
    # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
    hidden_states = hidden_states * normalizer

    # Add position_embeddings computation before the layer loop
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for i, decoder_layer in enumerate(self.layers):
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                **kwargs,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = layer_outputs[0]
        if i == layer_idx:
            break
    return hidden_states, causal_mask, position_ids, cache_position


def model_second_half_forward(
    self,
    hidden_states,
    causal_mask,
    position_ids,
    layer_idx: int = 13,
    return_dict: bool = False,
    all_layers_process_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # Add position_embeddings computation
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for i, decoder_layer in enumerate(self.layers[layer_idx + 1 :]):
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                None,
                False,
                None,
                position_embeddings,
                cache_position,
                **kwargs,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=None,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = layer_outputs[0]
        if all_layers_process_fn is not None:
            hidden_states = all_layers_process_fn(hidden_states)

    hidden_states = self.norm(hidden_states)

    if not return_dict:
        return tuple(v for v in [hidden_states, None, None, None] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    )


def causal_lm_first_half_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[HybridCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    layer_idx: int = 13,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, GemmaForCausalLM

    >>> model = GemmaForCausalLM.from_pretrained("google/gemma-2-9b")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""

    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma2 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.first_half_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        layer_idx=layer_idx,
        **kwargs,
    )
    return outputs


def causal_lm_second_half_forward(
    self,
    hidden_states,
    causal_mask,
    position_ids,
    labels: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    return_dict: bool = False,
    layer_idx: int = 13,
    all_layers_process_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    **kwargs,
):
    outputs = self.model.second_half_forward(
        hidden_states=hidden_states,
        causal_mask=causal_mask,
        position_ids=position_ids,
        layer_idx=layer_idx,
        return_dict=return_dict,
        all_layers_process_fn=all_layers_process_fn,
        **kwargs,
    )
    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    if self.config.final_logit_softcapping is not None:
        logits = logits / self.config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.final_logit_softcapping

    # TODO: remove the float() operation in v4.46
    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=None,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def compute_loss(self, logits, labels, already_shifted=False):
    """
    Compute the loss for a given logits and labels.
    If already_shifted is True, it is assumed that at an index i the target index is at index i in the labels.
    """
    if not already_shifted:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits
        shift_labels = labels
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    return loss_fct(shift_logits, shift_labels)


def split_gemma(model: GemmaForCausalLM):
    model.model.first_half_forward = MethodType(model_first_half_forward, model.model)
    model.model.second_half_forward = MethodType(model_second_half_forward, model.model)
    model.first_half_forward = MethodType(causal_lm_first_half_forward, model)
    model.second_half_forward = MethodType(causal_lm_second_half_forward, model)
    model.compute_loss = MethodType(compute_loss, model)
    return model
