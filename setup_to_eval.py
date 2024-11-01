import torch as th
from dictionary_learning.dictionary import CrossCoder
from abc import ABC, abstractmethod
import einops
from torch.nn.functional import relu
from typing import Literal

INTERESTING_FEATURES = [72073, 46325, 51408, 31726, 10833, 39938, 1045]


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
        filter_treshold: float | None = None,
        scale_steering_feature: float = 1.0,
        ignore_encoder: bool = False,
    ):
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
            return *self.continue_with_model(act + steering), None
        cc_input = th.stack(
            [base_activations, instruct_activations], dim=2
        ).float()  # b, seq, 2, d
        cc_input = einops.rearrange(cc_input, "b s m d -> (b s) m d")

        f = self.crosscoder.encode(
            cc_input, select_features=self.features_to_steer
        )  # (b s) f
        f = (
            einops.rearrange(f, "(b s) f -> b s f", b=base_activations.shape[0])
            * self.scale_steering_feature
        )
        mask = None
        if self.filter_treshold is not None:
            mask = f.sum(dim=-1).max(dim=-1).values > self.filter_treshold
            if not mask.any():
                return None, None, None
            f = f[mask]
        decoded = th.einsum("Bsf, lfd -> Bsld", f, self.decoder_weight)
        steering_feature = decoded[:, :, 1, :] - decoded[:, :, 0, :]
        if self.steer_with_features_from == "base":
            steering_feature = -steering_feature
        if self.filter_treshold is not None:
            act = act[mask]
        res = act + steering_feature
        return *self.continue_with_model(res.bfloat16()), mask


def create_half_fn_dict_main(
    crosscoder: CrossCoder,
    it_only_features: list[int],
    base_only_features: list[int],
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}

    half_fns["1-instruct"] = IdentityPreprocessFn(continue_with="instruct")

    # steer the base activations with all features from IT decoder and continue with instruct   ===   replace it error with base error
    half_fns["2-steer_all"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="instruct",
        continue_with="instruct",
        features_to_steer=None,
    )

    half_fns["3-base_to_instruct"] = SwitchPreprocessFn(continue_with="instruct")

    # steer the base activations with the IT only features from instruct decoder and continue with instruct
    half_fns["4-steer_it_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="instruct",
        continue_with="instruct",
        features_to_steer=it_only_features,
    )
    # 5/ RANDOM

    # instruct reconstruction
    half_fns["6-instruct_reconstruct"] = CrossCoderReconstruction(
        crosscoder, reconstruct_with="instruct", continue_with="instruct"
    )
    # Take the instruct activations and remove the IT only features
    half_fns["7-remove_it_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="instruct",
        steer_with_features_from="base",
        continue_with="instruct",
        features_to_steer=it_only_features,
    )

    # steer the base activations with the IT only & base only features from instruct decoder and continue with instruct
    half_fns["?-steer_it_and_base_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="instruct",
        continue_with="instruct",
        features_to_steer=it_only_features + base_only_features,
    )

    # reconstruct instruct and finish with instruct

    return half_fns


def create_half_fn_dict_secondary(
    crosscoder: CrossCoder, it_only_features: list[int], base_only_features: list[int]
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    half_fns["base_reconstruct"] = CrossCoderReconstruction(
        crosscoder, reconstruct_with="base", continue_with="base"
    )
    # steer the base activations with the IT only features from instruct decoder and continue with base
    half_fns["steer_it_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="instruct",
        continue_with="base",
        features_to_steer=it_only_features,
    )
    # steer the base activations with the IT only & base only features from instruct decoder and continue with base
    half_fns["steer_it_and_base_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="instruct",
        continue_with="base",
        features_to_steer=it_only_features + base_only_features,
    )
    # steer the base activations with the base only features from instruct decoder and continue with base
    half_fns["steer_base_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="instruct",
        continue_with="base",
        features_to_steer=base_only_features,
    )
    # steer the base activations with all features from IT decoder and continue with base   ===   replace it error with base error
    half_fns["steer_all_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="instruct",
        continue_with="base",
        features_to_steer=None,
    )
    # steer the base activations with the base only features from instruct decoder and continue with instruct
    half_fns["steer_base_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="instruct",
        continue_with="instruct",
        features_to_steer=base_only_features,
    )

    return half_fns


def create_half_fn_dict_steer_seeds(
    crosscoder: CrossCoder,
    seeds: list[int],
    num_features: int,
    threshold: float | None = None,
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    for seed in seeds:
        th.manual_seed(seed)
        features_to_steer = th.randperm(crosscoder.decoder.weight.shape[1])[
            :num_features
        ]
        name = f"5-steer_random_s{seed}_n{num_features}"
        if threshold is not None:
            name += f"_t{threshold}"
        half_fns[name] = CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="instruct",
            continue_with="instruct",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
        )
    return half_fns


def create_half_fn_dict_remove_seeds(
    crosscoder: CrossCoder,
    seeds: list[int],
    num_features: int,
    threshold: float | None = None,
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    for seed in seeds:
        th.manual_seed(seed)
        features_to_steer = th.randperm(crosscoder.decoder.weight.shape[1])[
            :num_features
        ]
        name = f"?-remove_random_s{seed}_n{num_features}"
        if threshold is not None:
            name += f"_t{threshold}"
        half_fns[name] = CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="instruct",
            steer_with_features_from="base",
            continue_with="instruct",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
        )
    return half_fns


def create_half_fn_dict_no_cross() -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    half_fns["base"] = IdentityPreprocessFn(continue_with="base")
    half_fns["instruct - debugging"] = IdentityPreprocessFn(continue_with="instruct")
    half_fns["instruct_to_base"] = SwitchPreprocessFn(continue_with="base")
    return half_fns


def create_it_only_ft_fn_dict(
    crosscoder: CrossCoder, it_only_features: list[int], base_only_features: list[int]
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {
        "remove_it_only_custom": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="instruct",
            steer_with_features_from="base",
            continue_with="instruct",
            features_to_steer=it_only_features,
        ),
        "remove_it_and_base_only_custom": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="instruct",
            steer_with_features_from="base",
            continue_with="instruct",
            features_to_steer=it_only_features + base_only_features,
        ),
        "steer_it_only_custom_to_base": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="instruct",
            continue_with="base",
            features_to_steer=it_only_features,
        ),
        "steer_it_only_custom": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="instruct",
            continue_with="instruct",
            features_to_steer=it_only_features,
        ),
        "steer_it_and_base_only_custom": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="instruct",
            continue_with="instruct",
            features_to_steer=it_only_features + base_only_features,
        ),
    }
    return half_fns


def create_half_fn_thresholded_features(
    crosscoder: CrossCoder,
    threshold: float = 10,
    features_to_steer: list[int] | None = None,
):
    if features_to_steer is None:
        features_to_steer = INTERESTING_FEATURES
    half_fns = {
        f"steer_t{threshold}_all": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="instruct",
            continue_with="instruct",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
        ),
        f"rm_t{threshold}_all": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="instruct",
            steer_with_features_from="base",
            continue_with="instruct",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
        ),
    }
    for feature in features_to_steer:
        half_fns[f"steer_t{threshold}_{feature}"] = CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="instruct",
            continue_with="instruct",
            features_to_steer=[feature],
            filter_treshold=threshold,
        )
        half_fns[f"rm_t{threshold}_{feature}"] = CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="instruct",
            steer_with_features_from="base",
            continue_with="instruct",
            features_to_steer=[feature],
            filter_treshold=threshold,
        )
    return half_fns
