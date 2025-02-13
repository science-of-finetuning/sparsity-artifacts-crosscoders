import torch as th
from dictionary_learning.dictionary import CrossCoder
from abc import ABC, abstractmethod
import einops
from typing import Literal


def ensure_model(arg: str):
    if arg not in ["base", "chat"]:
        raise ValueError(f"arg must be one of 'base', 'instruct', 'it', got {arg}")
    return arg


class HalfStepPreprocessFn(ABC):
    """Base class for preprocessing functions that modify activations during model execution."""

    @abstractmethod
    def preprocess(
        self, base_activations, chat_activations, **kwargs
    ) -> (
        tuple[th.Tensor, None]
        | tuple[None, th.Tensor]
        | tuple[None, th.Tensor, th.Tensor | None]
        | tuple[th.Tensor, None, th.Tensor | None]
        | tuple[None, None, None]
    ):
        """
        Preprocess the activations before the last half forward.
        Returns activations, None if the base model should finish the forward pass.
        Returns None, activations if the instruct model should finish the forward pass.
        Can also return a third element which is the batch mask of activations that were selected.

        Accepts additional keyword arguments to support extra inputs (e.g. ctrl_mask, assistant_mask, k_first_pred_toks_mask, next_ass_toks_mask).
        """
        raise NotImplementedError

    def __call__(
        self,
        base_activations,
        chat_activations,
        **kwargs,
    ) -> (
        tuple[th.Tensor, None]
        | tuple[None, th.Tensor]
        | tuple[None, th.Tensor, th.Tensor | None]
        | tuple[th.Tensor, None, th.Tensor | None]
        | tuple[None, None, None]
    ):
        res = self.preprocess(base_activations, chat_activations, **kwargs)
        if len(res) == 2:  # no mask provided
            return res[0], res[1], None
        elif len(res) == 3:  # mask
            assert (isinstance(res[2], th.Tensor) and res[2].dtype == th.bool) or res[
                2
            ] is None, "Mask must be a boolean tensor"
            assert res[2] is None or res[2].shape[0] == base_activations.shape[0], (
                "Mask must have the same number of elements as the input activations"
                if res[2] is not None
                else "Mask can be None"
            )
            return res
        else:
            raise ValueError(f"Expected 2 or 3 elements, got {len(res)}")

    def result(self, base_activations, chat_activations, **kwargs):
        """
        Returns the result of the preprocess function. Use this if you don't care about which model should continue the forward pass.
        """
        res = self(base_activations, chat_activations, **kwargs)
        return res[0] if res[0] is not None else res[1]


class IdentityPreprocessFn(HalfStepPreprocessFn):
    """Simple preprocessing function that returns activations unchanged."""

    def __init__(self, continue_with: Literal["base", "chat"]):
        self.continue_with = ensure_model(continue_with)

    def continue_with_model(self, result):
        return (result, None) if self.continue_with == "base" else (None, result)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        return (
            (base_activations, None)
            if self.continue_with == "base"
            else (None, chat_activations)
        )


class SwitchPreprocessFn(HalfStepPreprocessFn):
    """Preprocessing function that swaps activations between base and chat models."""

    def __init__(self, continue_with: Literal["base", "chat"]):
        self.continue_with = ensure_model(continue_with)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        return (
            (chat_activations, None)
            if self.continue_with == "base"
            else (None, base_activations)
        )


class TestMaskPreprocessFn(IdentityPreprocessFn):
    """Test preprocessing function that randomly masks some batch elements."""

    def preprocess(self, base_activations, chat_activations, **kwargs):
        mask = th.randint(
            0, 2, (base_activations.shape[0],), device=base_activations.device
        ).bool()
        if not mask.any():
            return None, None, None
        return (
            *super().preprocess(
                base_activations[mask], chat_activations[mask], **kwargs
            ),
            mask,
        )


class CrossCoderReconstruction(IdentityPreprocessFn):
    """Preprocessing function that reconstructs activations using a CrossCoder model."""

    def __init__(
        self,
        crosscoder: CrossCoder,
        reconstruct_with: Literal["base", "chat"],
        continue_with: Literal["base", "chat"],
    ):
        super().__init__(continue_with)
        self.crosscoder = crosscoder
        self.reconstruct_with = ensure_model(reconstruct_with)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        cc_input = th.stack(
            [base_activations, chat_activations], dim=2
        ).float()  # b, s, 2, d
        cc_input = einops.rearrange(cc_input, "b s m d -> (b s) m d")
        f = self.crosscoder.encode(cc_input)  # (b s) D
        reconstruction = th.einsum(
            "bD, Dd->bd",
            f,
            self.crosscoder.decoder.weight[0 if self.reconstruct_with == "base" else 1],
        )  # (b s) d
        reconstruction = (
            einops.rearrange(
                reconstruction, "(b s) d -> b s d", b=base_activations.shape[0]
            )
            + self.crosscoder.decoder.bias[0 if self.reconstruct_with == "base" else 1]
        )
        return self.continue_with_model(reconstruction.bfloat16())


class CrossCoderSteeringLatent(IdentityPreprocessFn):
    """Preprocessing function that steers activations using CrossCoder latent space.

    This class implements activation steering by:
    1. Encoding activations into the CrossCoder latent space
    2. Computing steering based on differences in decoded latents
    3. Applying the steering to the original activations

    Args:
        crosscoder: The CrossCoder model to use for encoding/decoding
        steer_activations_of: Which model's activations to steer ("base" or "chat")
        steer_with_latents_from: Which model's latents to use for steering ("base" or "chat")
        latents_to_steer: List of latent dimensions to use for steering, or None for all
        continue_with: Which model should continue generation after steering
        monitored_latents: List of latent dimensions to monitor, defaults to latents_to_steer
        filter_treshold: Optional threshold for filtering based on latent magnitudes
        scale_steering_latent: Scale factor for steering magnitude
        ignore_encoder: If True, uses only decoder weights for steering without encoding
    """

    def __init__(
        self,
        crosscoder: CrossCoder,
        steer_activations_of: Literal["base", "chat"],
        steer_with_latents_from: Literal["base", "chat"],
        latents_to_steer: list[int] | None,
        continue_with: Literal["base", "chat"],
        monitored_latents: list[int] | None = None,
        filter_treshold: float | None = None,
        scale_steering_latent: float = 1.0,
        ignore_encoder: bool = False,
    ):
        if filter_treshold is not None and ignore_encoder:
            raise ValueError(
                "Cannot filter latents and ignore encoder at the same time"
            )
        super().__init__(continue_with)
        if latents_to_steer is None:
            latents_to_steer = list(range(crosscoder.decoder.weight.shape[1]))
        self.decoder_weight = crosscoder.decoder.weight[:, latents_to_steer]  # lfd
        self.crosscoder = crosscoder
        self.steer_with_latents_from = ensure_model(steer_with_latents_from)
        self.steer_activations_of = ensure_model(steer_activations_of)
        self.filter_treshold = filter_treshold
        self.latents_to_steer = latents_to_steer
        self.scale_steering_latent = scale_steering_latent
        self.ignore_encoder = ignore_encoder
        if monitored_latents is None:
            monitored_latents = latents_to_steer
        elif len(monitored_latents) != len(latents_to_steer):
            raise ValueError(
                "Monitored latents and latents to steer must have the same length"
            )
        self.monitored_latents = monitored_latents

    def preprocess(self, base_activations, chat_activations, **kwargs):
        act = (
            base_activations
            if self.steer_activations_of == "base"
            else chat_activations
        )
        if self.ignore_encoder:
            steering = (self.decoder_weight[1] - self.decoder_weight[0]).sum(
                dim=0
            ) * self.scale_steering_latent
            if self.steer_with_latents_from == "base":
                steering = -steering
            return self.continue_with_model((act + steering).bfloat16())
        cc_input = th.stack(
            [base_activations, chat_activations], dim=-2
        ).float()  # b, seq, 2, d
        if base_activations.dim() == 3:
            cc_input = einops.rearrange(cc_input, "b s m d -> (b s) m d")
            assert cc_input.shape == (
                base_activations.shape[0] * base_activations.shape[1],
                2,
                base_activations.shape[2],
            ), f"cc_input.shape: {cc_input.shape}, base_activations.shape: {base_activations.shape}"
        else:
            assert cc_input.shape == (
                base_activations.shape[0],
                2,
                base_activations.shape[1],
            ), f"cc_input.shape: {cc_input.shape}, base_activations.shape: {base_activations.shape}"
            assert (
                base_activations.dim() == 2
            ), f"base_activations.dim(): {base_activations.dim()}"
        f = self.crosscoder.encode(
            cc_input, select_features=self.monitored_latents
        )  # (b s) f
        if base_activations.dim() == 3:
            f = einops.rearrange(f, "(b s) f -> b s f", b=base_activations.shape[0])
        mask = None
        if self.filter_treshold is not None:
            mask = f.sum(dim=-1).max(dim=-1).values > self.filter_treshold
            if not mask.any():
                return None, None, None
            f = f[mask]
        f = f * self.scale_steering_latent
        self.last_f = f
        if base_activations.dim() == 3:
            decoded = th.einsum("bsf, lfd -> bsld", f, self.decoder_weight)
            steering_latent = decoded[:, :, 1, :] - decoded[:, :, 0, :]
        else:
            decoded = th.einsum("Bf, lfd -> Bld", f, self.decoder_weight)
            steering_latent = decoded[:, 1, :] - decoded[:, 0, :]
        if self.steer_with_latents_from == "base":
            steering_latent = -steering_latent
        if self.filter_treshold is not None:
            act = act[mask]
        res = act + steering_latent
        # TODO: Improve this, this is a hack to get the steering latent for all layers
        self.last_steering_latent = steering_latent
        return *self.continue_with_model(res.bfloat16()), mask

    def all_layers(self, act):
        assert (
            self.last_steering_latent is not None
        ), "last_steering_latent is not set, call preprocess first"
        return act + self.last_steering_latent


class CrossCoderOutProjection(IdentityPreprocessFn):
    """Preprocessing function that projects out certain directions from activations using CrossCoder.

    This class removes components of the activations that align with specified CrossCoder latent dimensions.

    Args:
        crosscoder: The CrossCoder model to use
        steer_activations_of: Which model's activations to modify ("base" or "chat")
        steer_with_latents_from: Which model's latents to use ("base" or "chat")
        latents_to_steer: List of latent dimensions to project out, or None for all
        continue_with: Which model should continue generation after modification
        scale_steering_latent: Scale factor for the projection strength
    """

    def __init__(
        self,
        crosscoder: CrossCoder,
        steer_activations_of: Literal["base", "chat"],
        steer_with_latents_from: Literal["base", "chat"],
        latents_to_steer: list[int] | None,
        continue_with: Literal["base", "chat"],
        scale_steering_latent: float = 1.0,
    ):
        super().__init__(continue_with)
        self.crosscoder = crosscoder
        if latents_to_steer is None:
            latents_to_steer = list(range(crosscoder.decoder.weight.shape[1]))
        self.steer_with_latents_from = ensure_model(steer_with_latents_from)
        self.projection_matrices = []
        self.layer_idx = int(self.steer_with_latents_from == "chat")
        for i in latents_to_steer:
            normalized_decoder_vector = (
                crosscoder.decoder.weight[self.layer_idx, i]
                / crosscoder.decoder.weight[self.layer_idx, i].norm()
            )
            self.projection_matrices.append(
                th.outer(normalized_decoder_vector, normalized_decoder_vector)
            )
        self.steer_activations_of = ensure_model(steer_activations_of)
        self.latents_to_steer = latents_to_steer
        self.scale_steering_latent = scale_steering_latent

    def project_out(self, act):
        # act: bd
        batch_size = act.shape[0]
        original_dtype = act.dtype
        act = einops.rearrange(act, "b s d -> (b s) d").to(
            self.crosscoder.decoder.weight.dtype
        )
        for projection_matrix in self.projection_matrices:
            act = act - act @ projection_matrix * self.scale_steering_latent
        out = einops.rearrange(act, "(b s) d -> b s d", b=batch_size).to(original_dtype)
        return out

    def preprocess(self, base_activations, chat_activations, **kwargs):
        act = (
            base_activations
            if self.steer_activations_of == "base"
            else chat_activations
        )

        return self.continue_with_model(self.project_out(act))

    def all_layers(self, act):
        return self.project_out(act)


class BasePatchIntervention(IdentityPreprocessFn):
    """Base class for interventions that patch activations from one model to another.

    This class provides common functionality for patching activations based on a mask.
    Derived classes should implement get_patch_mask() to specify which tokens to patch.

    Args:
        continue_with: Which model should complete generation ("base" or "chat")
        patch_target: Which model activations to patch ("base" or "chat")
        activation_processor: Optional preprocess function to apply to the source activations before patching. If not provided, the other model (not patch_target) will be used as the source.
    """

    def __init__(
        self,
        continue_with: Literal["base", "chat"],
        patch_target: Literal["base", "chat"],
        activation_processor: IdentityPreprocessFn | None = None,
    ):
        super().__init__(continue_with)
        self.patch_target = ensure_model(patch_target)
        if activation_processor is None:
            self.activation_processor = IdentityPreprocessFn(
                "base" if self.patch_target == "chat" else "chat"
            )
        else:
            self.activation_processor = activation_processor

    def _validate_mask(self, mask: th.Tensor, activations: th.Tensor) -> None:
        """Validates that a mask has correct shape and type if provided."""
        if not isinstance(mask, th.Tensor):
            raise TypeError(f"Mask must be a tensor, got {type(mask)}")
        if mask.dtype != th.bool:
            raise TypeError(f"Mask must be boolean, got {mask.dtype}")
        if mask.shape != activations.shape[:2]:  # batch x seq
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match activation shape {activations.shape[:2]}"
            )

    def get_patch_mask(self, **kwargs) -> th.Tensor:
        """Returns boolean mask indicating which tokens to patch. Override in derived classes."""
        raise NotImplementedError

    def preprocess(
        self, base_activations: th.Tensor, chat_activations: th.Tensor, **kwargs
    ):
        mask = self.get_patch_mask(**kwargs)
        self._validate_mask(mask, base_activations)

        # Get source and target based on configuration
        destination = (
            base_activations if self.patch_target == "base" else chat_activations
        )

        patch = self.activation_processor.result(
            base_activations[mask], chat_activations[mask], **kwargs
        )
        # Apply patch
        modified = destination.clone()
        modified[mask] = patch

        return self.continue_with_model(modified)


class PatchCtrl(BasePatchIntervention):
    """Patches control tokens from one model to another.

    This intervention copies control token activations (specified by ctrl_mask)
    from the source model to the target model.

    Args:
        continue_with: Which model should complete generation ("base" or "chat")
        patch_target: Which model activations to patch ("base" or "chat")
        activation_processor: Optional preprocess function to apply to the source activations before patching. If not provided, the other model (not patch_target) will be used as the source.
    Kwargs for preprocess:
        ctrl_mask: Boolean tensor of shape [batch_size, seq_len] indicating control tokens
    """

    def get_patch_mask(self, *, ctrl_mask: th.Tensor, **kwargs) -> th.Tensor:
        return ctrl_mask


class PatchKFirstPredictions(BasePatchIntervention):
    """Patches the first k assistant tokens from one model to another.

    This intervention copies the first k assistant token activations
    (specified by k_first_pred_toks_mask) from the source model to the target model.

    Args:
        continue_with: Which model should complete generation ("base" or "chat")
        patch_target: Which model activations to patch ("base" or "chat")
        activation_processor: Optional preprocess function to apply to the source activations before patching. If not provided, the other model (not patch_target) will be used as the source.
    Kwargs for preprocess:
        k_first_pred_toks_mask: Boolean tensor of shape [batch_size, seq_len]
                              indicating first k assistant tokens
    """

    def get_patch_mask(
        self, *, k_first_pred_toks_mask: th.Tensor, **kwargs
    ) -> th.Tensor:
        return k_first_pred_toks_mask


class PatchKFirstAndCtrl(BasePatchIntervention):
    """Patches tokens specified by combining control and first k assistant token masks.

    This intervention combines ctrl_mask and k_first_pred_toks_mask using OR operation
    to patch tokens from the source model to the target model.

    Args:
        continue_with: Which model should complete generation ("base" or "chat")
        patch_target: Which model activations to patch ("base" or "chat")
        activation_processor: Optional preprocess function to apply to the source activations before patching. If not provided, the other model (not patch_target) will be used as the source.
    Kwargs for preprocess:
        ctrl_mask: Boolean tensor of shape [batch_size, seq_len] indicating control tokens
        k_first_pred_toks_mask: Boolean tensor of shape [batch_size, seq_len]
                              indicating first k assistant tokens
    """

    def get_patch_mask(
        self, *, ctrl_mask: th.Tensor, k_first_pred_toks_mask: th.Tensor, **kwargs
    ) -> th.Tensor:
        return ctrl_mask | k_first_pred_toks_mask
