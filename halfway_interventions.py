import torch as th
from dictionary_learning.dictionary import CrossCoder
from abc import ABC, abstractmethod
import einops
from typing import Literal


def ensure_model(arg: str):
    if arg not in ["base", "instruct", "it"]:
        raise ValueError(f"arg must be one of 'base', 'instruct', 'it', got {arg}")
    return arg


class HalfStepPreprocessFn(ABC):
    @abstractmethod
    def preprocess(
        self, base_activations, instruct_activations
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
        """
        raise NotImplementedError

    def __call__(self, base_activations, instruct_activations):
        res = self.preprocess(base_activations, instruct_activations)
        if len(res) == 2:  # no mask
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


class IdentityPreprocessFn(HalfStepPreprocessFn):
    def __init__(self, continue_with: Literal["base", "instruct", "it"]):
        self.continue_with = ensure_model(continue_with)

    def continue_with_model(self, result):
        return (result, None) if self.continue_with == "base" else (None, result)

    def preprocess(self, base_activations, instruct_activations):
        return (
            (base_activations, None)
            if self.continue_with == "base"
            else (None, instruct_activations)
        )


class SwitchPreprocessFn(HalfStepPreprocessFn):
    def __init__(self, continue_with: Literal["base", "instruct", "it"]):
        self.continue_with = ensure_model(continue_with)

    def preprocess(self, base_activations, instruct_activations):
        return (
            (instruct_activations, None)
            if self.continue_with == "base"
            else (None, base_activations)
        )


class TestMaskPreprocessFn(IdentityPreprocessFn):
    def preprocess(self, base_activations, instruct_activations):
        mask = th.randint(
            0, 2, (base_activations.shape[0],), device=base_activations.device
        ).bool()
        if not mask.any():
            return None, None, None
        return (
            *super().preprocess(base_activations[mask], instruct_activations[mask]),
            mask,
        )


class CrossCoderReconstruction(IdentityPreprocessFn):
    def __init__(
        self,
        crosscoder: CrossCoder,
        reconstruct_with: Literal["base", "instruct", "it"],
        continue_with: Literal["base", "instruct", "it"],
    ):
        super().__init__(continue_with)
        self.crosscoder = crosscoder
        self.reconstruct_with = ensure_model(reconstruct_with)

    def preprocess(self, base_activations, instruct_activations):
        cc_input = th.stack(
            [base_activations, instruct_activations], dim=2
        ).float()  # b, seq, 2, d
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


class CrossCoderSteeringFeature(IdentityPreprocessFn):
    def __init__(
        self,
        crosscoder: CrossCoder,
        steer_activations_of: Literal["base", "instruct", "it"],
        steer_with_features_from: Literal["base", "instruct", "it"],
        features_to_steer: list[int] | None,
        continue_with: Literal["base", "instruct", "it"],
        monitored_features: list[int] | None = None,
        filter_treshold: float | None = None,
        scale_steering_feature: float = 1.0,
        ignore_encoder: bool = False,
    ):
        if filter_treshold is not None and ignore_encoder:
            raise ValueError(
                "Cannot filter features and ignore encoder at the same time"
            )
        super().__init__(continue_with)
        if features_to_steer is None:
            features_to_steer = list(range(crosscoder.decoder.weight.shape[1]))
        self.decoder_weight = crosscoder.decoder.weight[:, features_to_steer]  # lfd
        self.crosscoder = crosscoder
        self.steer_with_features_from = ensure_model(steer_with_features_from)
        self.steer_activations_of = ensure_model(steer_activations_of)
        self.filter_treshold = filter_treshold
        self.features_to_steer = features_to_steer
        self.scale_steering_feature = scale_steering_feature
        self.ignore_encoder = ignore_encoder
        if monitored_features is None:
            monitored_features = features_to_steer
        elif len(monitored_features) != len(features_to_steer):
            raise ValueError(
                "Monitored features and features to steer must have the same length"
            )
        self.monitored_features = monitored_features

    def preprocess(self, base_activations, instruct_activations):
        act = (
            base_activations
            if self.steer_activations_of == "base"
            else instruct_activations
        )
        if self.ignore_encoder:
            steering = (self.decoder_weight[1] - self.decoder_weight[0]).sum(
                dim=0
            ) * self.scale_steering_feature
            if self.steer_with_features_from == "base":
                steering = -steering
            return *self.continue_with_model((act + steering).bfloat16()), None
        cc_input = th.stack(
            [base_activations, instruct_activations], dim=2
        ).float()  # b, seq, 2, d
        cc_input = einops.rearrange(cc_input, "b s m d -> (b s) m d")

        f = self.crosscoder.encode(
            cc_input, select_features=self.monitored_features
        )  # (b s) f
        f = einops.rearrange(f, "(b s) f -> b s f", b=base_activations.shape[0])
        mask = None
        if self.filter_treshold is not None:
            mask = f.sum(dim=-1).max(dim=-1).values > self.filter_treshold
            if not mask.any():
                return None, None, None
            f = f[mask]
        f = f * self.scale_steering_feature
        self.last_f = f
        decoded = th.einsum("Bsf, lfd -> Bsld", f, self.decoder_weight)
        steering_feature = decoded[:, :, 1, :] - decoded[:, :, 0, :]
        if self.steer_with_features_from == "base":
            steering_feature = -steering_feature
        if self.filter_treshold is not None:
            act = act[mask]
        res = act + steering_feature
        # TODO: Improve this, this is a hack to get the steering feature for all layers
        self.last_steering_feature = steering_feature
        return *self.continue_with_model(res.bfloat16()), mask

    def all_layers(self, act):
        assert (
            self.last_steering_feature is not None
        ), "last_steering_feature is not set, call preprocess first"
        return act + self.last_steering_feature


class CrossCoderOutProjection(IdentityPreprocessFn):
    def __init__(
        self,
        crosscoder: CrossCoder,
        steer_activations_of: Literal["base", "instruct", "it"],
        steer_with_features_from: Literal["base", "instruct", "it"],
        features_to_steer: list[int] | None,
        continue_with: Literal["base", "instruct", "it"],
        scale_steering_feature: float = 1.0,
    ):
        super().__init__(continue_with)
        self.crosscoder = crosscoder
        if features_to_steer is None:
            features_to_steer = list(range(crosscoder.decoder.weight.shape[1]))

        self.steer_with_features_from = ensure_model(steer_with_features_from)
        self.projection_matrices = []
        self.layer_idx = int(self.steer_with_features_from == "instruct")
        for i in features_to_steer:
            normalized_decoder_vector = (
                crosscoder.decoder.weight[self.layer_idx, i]
                / crosscoder.decoder.weight[self.layer_idx, i].norm()
            )
            self.projection_matrices.append(
                th.outer(normalized_decoder_vector, normalized_decoder_vector)
            )
        self.steer_activations_of = ensure_model(steer_activations_of)
        self.features_to_steer = features_to_steer
        self.scale_steering_feature = scale_steering_feature

    def project_out(self, act):
        # act: bd
        batch_size = act.shape[0]
        original_dtype = act.dtype
        act = einops.rearrange(act, "b s d -> (b s) d").to(
            self.crosscoder.decoder.weight.dtype
        )
        for projection_matrix in self.projection_matrices:
            act = act - act @ projection_matrix * self.scale_steering_feature

        out = einops.rearrange(act, "(b s) d -> b s d", b=batch_size).to(original_dtype)
        return out

    def preprocess(self, base_activations, instruct_activations):
        act = (
            base_activations
            if self.steer_activations_of == "base"
            else instruct_activations
        )

        return *self.continue_with_model(self.project_out(act)), None

    def all_layers(self, act):
        return self.project_out(act)
