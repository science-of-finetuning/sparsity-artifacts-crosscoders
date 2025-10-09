import torch as th
from dictionary_learning.dictionary import CrossCoder, Dictionary
from abc import ABC, abstractmethod
import einops
from typing import Literal
import pandas as pd


def ensure_model(arg: str):
    if arg not in ["base", "chat"]:
        raise ValueError(f"arg must be one of 'base', 'instruct', 'it', got {arg}")
    return arg


class HalfStepPreprocessFn(ABC):
    """Base class for preprocessing functions that modify activations during model execution."""

    can_edit_single_activation = False

    def __init__(self, model_dtype: th.dtype = th.bfloat16):
        self.model_dtype = model_dtype

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

    def _preprocess_single_activation(self, activation, **kwargs):
        """
        Edit a single activation
        """
        raise NotImplementedError

    def process_single_activation(self, activation, **kwargs):
        """
        Edit a single activation
        """
        if self.can_edit_single_activation:
            return self._preprocess_single_activation(activation, **kwargs)
        else:
            raise ValueError(
                "This preprocess function does not support editing single activations. You need to use preprocess instead."
            )


class IdentityPreprocessFn(HalfStepPreprocessFn):
    """Simple preprocessing function that returns activations unchanged."""

    can_edit_single_activation = True

    def __init__(
        self,
        continue_with: Literal["base", "chat"],
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(model_dtype)
        self.continue_with = ensure_model(continue_with)

    def continue_with_model(self, result):
        return (result, None) if self.continue_with == "base" else (None, result)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        return (
            (base_activations, None)
            if self.continue_with == "base"
            else (None, chat_activations)
        )

    def _process_single_activation(self, activation, **kwargs):
        return activation


class SwitchPreprocessFn(HalfStepPreprocessFn):
    """Preprocessing function that swaps activations between base and chat models."""

    def __init__(
        self,
        continue_with: Literal["base", "chat"],
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(model_dtype)
        self.continue_with = ensure_model(continue_with)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        return (
            (chat_activations, None)
            if self.continue_with == "base"
            else (None, base_activations)
        )


class TestMaskPreprocessFn(IdentityPreprocessFn):
    """Test preprocessing function that randomly masks some batch elements."""

    def _process_single_activation(self, activation, **kwargs):
        mask = th.randint(0, 2, (activation.shape[0],), device=activation.device).bool()
        if not mask.any():
            return None
        return activation[mask]

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


class PatchProjectionFromDiff(IdentityPreprocessFn):
    """Preprocessing function that projects the difference between two activations to another activation.

    Args:
        continue_with: Which model should complete generation ("base" or "chat")
        patch_target: Which model activations to patch ("base" or "chat")
        vectors_to_project: Tensor of shape (n,d) where n is the number of vectors to project the difference to.
        scale_steering_latent: Scale factor for the projection strength
    """

    can_edit_single_activation = False

    def __init__(
        self,
        continue_with: Literal["base", "chat"],
        patch_target: Literal["base", "chat"],
        vectors_to_project: th.Tensor,
        scale_steering_latent: float = 1.0,
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
        if vectors_to_project.ndim == 1:
            vectors_to_project = vectors_to_project.unsqueeze(0)
        self.patch_target = ensure_model(patch_target)
        self.vectors_to_project = vectors_to_project
        self.scale_steering_latent = scale_steering_latent

    def get_patch_steering_vector(self, base_activations, chat_activations):
        diff = chat_activations - base_activations
        scale = self.scale_steering_latent
        if self.patch_target == "chat":
            scale = -scale
        return (
            th.einsum("bsd, nd -> bsn", diff, self.vectors_to_project).unsqueeze(-1)
            * self.vectors_to_project
            * scale
        ).sum(dim=-2)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        steering_vector = self.get_patch_steering_vector(
            base_activations, chat_activations
        )
        target_acts = (
            base_activations if self.patch_target == "base" else chat_activations
        )
        return self.continue_with_model(target_acts + steering_vector)


class SteeringVector(IdentityPreprocessFn):
    """Preprocessing function that steers activations using a steering vector."""

    can_edit_single_activation = True

    def __init__(
        self,
        continue_with: Literal["base", "chat"],
        steer_activations_of: Literal["base", "chat"],
        vector: th.Tensor,
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
        self.steer_activations_of = ensure_model(steer_activations_of)
        self.vector = vector.to(model_dtype)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        acts = (
            base_activations
            if self.steer_activations_of == "base"
            else chat_activations
        )
        return self.continue_with_model(acts + self.vector)

    def _process_single_activation(self, activation, **kwargs):
        return activation + self.vector


class SteerWithCrosscoderLatent(SteeringVector):
    """Preprocessing function that steers activations using a steering vector."""

    can_edit_single_activation = True

    def __init__(
        self,
        crosscoder: CrossCoder,
        latent_df: pd.DataFrame,
        steer_activations_of: Literal["base", "chat"],
        steer_with_latents_from: Literal["base", "chat"],
        latents_to_steer: list[int] | int | None = None,
        continue_with: Literal["base", "chat"] = "chat",
        scale_steering_latent: float = 1.0,
        model_dtype: th.dtype = th.bfloat16,
    ):
        """Preprocessing function that steers activations using latents from a CrossCoder model.

        Args:
            crosscoder: The CrossCoder model to use for steering
            latent_df: DataFrame containing latent information, including max activation values
            steer_activations_of: Which model's activations to steer ("base" or "chat")
            steer_with_latents_from: Which decoder's latents to use for steering ("base" or "chat")
            latents_to_steer: List of latent indices to use for steering, or a single latent index,
                              or None to use all latents
            continue_with: Which model should complete generation ("base" or "chat")
            scale_steering_latent: Scaling factor to apply to the steering vector
        """
        if ensure_model(steer_with_latents_from) == "chat":
            weight_idx = 1
        else:
            weight_idx = 0
        if latents_to_steer is None:
            latents_to_steer = list(range(crosscoder.decoder.weight.shape[1]))
        elif isinstance(latents_to_steer, int):
            latents_to_steer = [latents_to_steer]
        latents = crosscoder.decoder.weight[weight_idx, latents_to_steer]
        column = (
            "max_act_train" if "max_act_train" in latent_df.columns else "max_act_val"
        )
        max_acts = th.from_numpy(latent_df[column].values)[latents_to_steer].to(
            latents.device
        )
        assert max_acts.shape == (latents.shape[0],)
        vector = (latents / latents.norm(p=2, dim=1) * max_acts).sum(
            dim=0
        ) * scale_steering_latent
        assert vector.shape == (
            crosscoder.activation_dim,
        ), f"vector.shape: {vector.shape}, crosscoder.activation_dim: {crosscoder.activation_dim}"
        super().__init__(continue_with, steer_activations_of, vector, model_dtype)


class CrossCoderReconstruction(IdentityPreprocessFn):
    """Preprocessing function that reconstructs activations using a CrossCoder model."""

    can_edit_single_activation = False

    def __init__(
        self,
        crosscoder: CrossCoder,
        reconstruct_with: Literal["base", "chat"],
        continue_with: Literal["base", "chat"],
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
        self.crosscoder = crosscoder
        self.reconstruct_with = ensure_model(reconstruct_with)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        cc_input = th.stack(
            [base_activations, chat_activations], dim=2
        ).float()  # b, s, 2, d
        cc_input = einops.rearrange(cc_input, "b s m d -> (b s) m d")
        f = self.crosscoder.encode(cc_input)  # (b s) D
        reconstruction = self.crosscoder.decode(f)  # (b s) 2 d
        if self.reconstruct_with == "base":
            reconstruction = reconstruction[:, 0, :]
        else:
            reconstruction = reconstruction[:, 1, :]
        reconstruction = einops.rearrange(
            reconstruction, "(b s) d -> b s d", b=base_activations.shape[0]
        )
        return self.continue_with_model(reconstruction.to(self.model_dtype))


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

    can_edit_single_activation = False

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
        model_dtype: th.dtype = th.bfloat16,
    ):
        if filter_treshold is not None and ignore_encoder:
            raise ValueError(
                "Cannot filter latents and ignore encoder at the same time"
            )
        super().__init__(continue_with, model_dtype)
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
            return self.continue_with_model((act + steering).to(self.model_dtype))
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
        if self.crosscoder.decoupled_code:
            if self.steer_with_latents_from == "base":
                f = f[:, 0, :]
            else:
                f = f[:, 1, :]
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
        return *self.continue_with_model(res.to(self.model_dtype)), mask

    def all_layers(self, act):
        assert (
            self.last_steering_latent is not None
        ), "last_steering_latent is not set, call preprocess first"
        return act + self.last_steering_latent


class CrossCoderAdditiveSteering(CrossCoderSteeringLatent):
    """Preprocessing function that adds activations using CrossCoder latent space.

    This class implements activation steering by:
    1. Encoding activations into the CrossCoder latent space
    2. Computing steering based on the latents from the steer_with_latents_from model
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_weight = self.decoder_weight.clone()
        if self.steer_with_latents_from == "chat":
            self.decoder_weight[0] = 0
        else:
            self.decoder_weight[1] = 0


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
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
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

    def _process_single_activation(self, activation, **kwargs):
        return self.project_out(activation)


class SAEReconstruction(IdentityPreprocessFn):
    """Use reconstruction of a model + reconstruction error of the other.

    Args:
        sae: The SAE model to use
        use_reconstruction_of: Which model's activations reconstruction to use
        continue_with: Which model should continue generation after steering
        model_dtype: The dtype of the model
    """

    can_edit_single_activation = False

    def __init__(
        self,
        sae: Dictionary,
        use_reconstruction_of: Literal["base", "chat"],
        continue_with: Literal["base", "chat"],
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
        self.sae = sae
        self.use_reconstruction_of = ensure_model(use_reconstruction_of)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        # Choose which activations to reconstruct
        if self.use_reconstruction_of == "base":
            activations_to_reconstruct = base_activations
            other_activations = chat_activations
        else:
            activations_to_reconstruct = chat_activations
            other_activations = base_activations

        # Reshape for SAE processing
        b, s, d = activations_to_reconstruct.shape
        flat_activations = einops.rearrange(
            activations_to_reconstruct, "b s d -> (b s) d"
        ).float()
        reconstructed = self.sae(flat_activations)
        reconstructed = einops.rearrange(reconstructed, "(b s) d -> b s d", b=b)

        other_flat = einops.rearrange(other_activations, "b s d -> (b s) d").float()
        other_reconstructed = self.sae(other_flat)
        other_reconstructed = einops.rearrange(
            other_reconstructed, "(b s) d -> b s d", b=b
        )
        reconstruction_error = other_activations - other_reconstructed

        # Combine reconstruction + reconstruction error
        result = reconstructed + reconstruction_error

        return self.continue_with_model(result.to(self.model_dtype))


class SAEAdditiveSteering(IdentityPreprocessFn):
    """Preprocessing function that adds activations using SAE activations and decoder weights.

    Args:
        sae: The SAE model to use
        steer_activations_of: Which model's activations to steer ("base" or "chat")
        steer_with_latents_from: Which model's latents to use for steering ("base" or "chat")
        continue_with: Which model should continue generation after steering
    """

    can_edit_single_activation = False  # note: could be implemented

    def __init__(
        self,
        sae: Dictionary,
        steer_activations_of: Literal["base", "chat"],
        latents_to_steer: list[int] | None,
        continue_with: Literal["base", "chat"],
        sae_model: Literal["base", "chat"],
        is_difference_sae: bool = False,
        # monitored_latents: list[int] | None = None,
        # filter_treshold: float | None = None,
        scale_steering_latent: float = 1.0,
        model_dtype: th.dtype = th.bfloat16,
        add_decoder_bias: bool = False,
    ):
        super().__init__(continue_with, model_dtype)
        self.sae = sae
        self.sae_model = ensure_model(sae_model)
        self.steer_activations_of = ensure_model(steer_activations_of)
        self.latents_to_steer = latents_to_steer
        self.scale_steering_latent = scale_steering_latent
        self.is_difference_sae = is_difference_sae
        assert (
            self.sae.decoder.weight.shape[0] == self.sae.activation_dim
            and self.sae.decoder.weight.shape[1] == self.sae.dict_size
        ), "sae decoder has shape (dict_size, d_model), while assumed to be (d_model, dict_size)"
        self.decoder_weights = self.sae.decoder.weight[
            :, self.latents_to_steer
        ].T  # L, d
        assert self.decoder_weights.shape == (
            len(self.latents_to_steer),
            self.sae.activation_dim,
        )
        self.add_decoder_bias = add_decoder_bias

    def preprocess(self, base_activations, chat_activations, **kwargs):
        if self.is_difference_sae:
            if self.sae_model == "chat":
                sae_input = chat_activations - base_activations
            else:
                sae_input = base_activations - chat_activations
        else:
            sae_input = (
                base_activations if self.sae_model == "base" else chat_activations
            )
        if sae_input.dim() == 3:
            sae_input = einops.rearrange(sae_input, "b s d -> (b s) d")
        sae_input = sae_input.float()
        sae_acts = self.sae.encode(sae_input)  # bs, D
        assert sae_acts.shape == (sae_input.shape[0], self.sae.dict_size)
        steering_act = sae_acts[:, self.latents_to_steer]  # bs, L
        assert steering_act.shape == (sae_input.shape[0], len(self.latents_to_steer))
        steering_vects = th.einsum(
            "BL, Ld -> Bd", steering_act, self.decoder_weights
        )  # bs, d

        if self.add_decoder_bias:
            steering_vects = steering_vects + self.sae.b_dec

        assert steering_vects.shape == (sae_input.shape[0], sae_input.shape[1])
        # unfold steering_vects to b, s, d
        steering_vects = einops.rearrange(
            steering_vects, "(b s) d -> b s d", b=base_activations.shape[0]
        )
        steering_vects = steering_vects * self.scale_steering_latent
        if self.steer_activations_of == "base":
            steered = base_activations + steering_vects
        else:
            steered = chat_activations + steering_vects
        return self.continue_with_model(steered.to(self.model_dtype))


class SAEDifferenceReconstruction(IdentityPreprocessFn):
    """Preprocessing function that reconstructs difference activations using a difference SAE and adds to base activations."""

    can_edit_single_activation = False

    def __init__(
        self,
        sae: Dictionary,
        sae_model: Literal["base", "chat"],
        steer_activations_of: Literal["base", "chat"],
        continue_with: Literal["base", "chat"],
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
        self.sae_model = ensure_model(sae_model)
        self.sae = sae
        self.steer_activations_of = ensure_model(steer_activations_of)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        # Compute activation difference
        if self.sae_model == "chat":
            sae_input = chat_activations - base_activations  # (b, s, d)
        else:
            sae_input = base_activations - chat_activations
        b, s, d = sae_input.shape
        sae_input = einops.rearrange(sae_input, "b s d -> (b s) d").float()
        # Encode and decode to reconstruct difference
        chat_minus_base_recon = self.sae(sae_input)
        if self.sae_model == "base":
            chat_minus_base_recon = -chat_minus_base_recon
        chat_minus_base_recon = einops.rearrange(
            chat_minus_base_recon, "(b s) d -> b s d", b=b
        )
        # Add reconstructed difference to base activations
        if self.steer_activations_of == "base":
            # Approximate the chat activations by adding the reconstructed difference
            approx = base_activations + chat_minus_base_recon.to(self.model_dtype)
        else:
            # Approximate the base activations by subtracting the reconstructed difference
            approx = chat_activations - chat_minus_base_recon.to(self.model_dtype)
        return self.continue_with_model(approx)


class SAEDifferenceBiasSteering(IdentityPreprocessFn):
    """Preprocessing function that steers activations using only the bias term from a difference SAE."""

    can_edit_single_activation = False  # note: could be implemented

    def __init__(
        self,
        sae: Dictionary,
        sae_model: Literal["base", "chat"],
        steer_activations_of: Literal["base", "chat"],
        continue_with: Literal["base", "chat"],
        scale_steering: float = 1.0,
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
        self.sae_model = ensure_model(sae_model)
        self.sae = sae
        self.steer_activations_of = ensure_model(steer_activations_of)
        self.scale_steering = scale_steering

    def preprocess(self, base_activations, chat_activations, **kwargs):
        # Get the decoder bias from the SAE
        scaled_bias_chat_minus_base = self.sae.b_dec * self.scale_steering

        # Apply sign correction based on sae_model
        if self.sae_model == "base":
            scaled_bias_chat_minus_base = -scaled_bias_chat_minus_base

        # Add bias to the specified activations
        if self.steer_activations_of == "base":
            steered = base_activations + scaled_bias_chat_minus_base
        else:
            steered = chat_activations - scaled_bias_chat_minus_base

        return self.continue_with_model(steered.to(self.model_dtype))


class SAEDifferenceReconstructionError(IdentityPreprocessFn):
    """Preprocessing function that reconstructs the error between actual and reconstructed difference activations."""

    can_edit_single_activation = False

    def __init__(
        self,
        sae: Dictionary,
        sae_model: Literal["base", "chat"],
        steer_activations_of: Literal["base", "chat"],
        continue_with: Literal["base", "chat"],
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
        self.sae_model = ensure_model(sae_model)
        self.sae = sae
        self.steer_activations_of = ensure_model(steer_activations_of)

    def preprocess(self, base_activations, chat_activations, **kwargs):
        # Compute activation difference
        chat_minus_base = chat_activations - base_activations
        if self.sae_model == "chat":
            sae_input = chat_minus_base  # (b, s, d)
        else:
            sae_input = base_activations - chat_activations
        b, s, d = sae_input.shape
        sae_input = einops.rearrange(sae_input, "b s d -> (b s) d").float()

        # Encode and decode to reconstruct difference
        chat_minus_base_recon = self.sae(sae_input)
        if self.sae_model == "base":
            chat_minus_base_recon = -chat_minus_base_recon
        chat_minus_base_recon = einops.rearrange(
            chat_minus_base_recon, "(b s) d -> b s d", b=b
        )

        # Compute reconstruction error
        chat_minus_base_recon_error = chat_minus_base - chat_minus_base_recon

        # Add reconstruction error to the specified activations
        if self.steer_activations_of == "base":
            steered = base_activations + chat_minus_base_recon_error
        else:
            steered = chat_activations - chat_minus_base_recon_error

        return self.continue_with_model(steered.to(self.model_dtype))


class BasePatchIntervention(IdentityPreprocessFn):
    """Base class for interventions that patch activations from one model to another.

    This class provides common functionality for patching activations based on a mask.
    Derived classes should implement get_patch_mask() to specify which tokens to patch.

    Args:
        continue_with: Which model should complete generation ("base" or "chat")
        patch_target: Which model activations to patch ("base" or "chat")
        activation_processor: Optional preprocess function to apply to the source activations before patching. If not provided, the other model (not patch_target) will be used as the source.
    """

    can_edit_single_activation = False  # note: could be implemented

    def __init__(
        self,
        continue_with: Literal["base", "chat"],
        patch_target: Literal["base", "chat"],
        activation_processor: IdentityPreprocessFn | None = None,
        model_dtype: th.dtype = th.bfloat16,
    ):
        super().__init__(continue_with, model_dtype)
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
