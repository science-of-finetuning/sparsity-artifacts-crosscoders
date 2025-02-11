import torch as th
from dictionary_learning.dictionary import CrossCoder
from abc import ABC, abstractmethod
import einops
from typing import Literal
from halfway_interventions import (
    HalfStepPreprocessFn,
    IdentityPreprocessFn,
    SwitchPreprocessFn,
    CrossCoderSteeringFeature,
    CrossCoderReconstruction,
)

INTERESTING_FEATURES = [72073, 46325, 51408, 31726, 10833, 39938, 1045]


def create_half_fn_dict_main(
    crosscoder: CrossCoder,
    it_only_features: list[int],
    base_only_features: list[int],
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}

    half_fns["1-chat"] = IdentityPreprocessFn(continue_with="chat")

    # steer the base activations with all features from IT decoder and continue with chat   ===   replace it error with base error
    half_fns["2-steer_all"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="chat",
        continue_with="chat",
        features_to_steer=None,
    )

    half_fns["3-base_to_chat"] = SwitchPreprocessFn(continue_with="chat")

    # steer the base activations with the IT only features from chat decoder and continue with chat
    half_fns["4-steer_it_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="chat",
        continue_with="chat",
        features_to_steer=it_only_features,
    )
    # 5/ RANDOM

    # chat reconstruction
    half_fns["6-chat_reconstruct"] = CrossCoderReconstruction(
        crosscoder, reconstruct_with="chat", continue_with="chat"
    )
    # Take the chat activations and remove the IT only features
    half_fns["7-remove_it_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="chat",
        steer_with_features_from="base",
        continue_with="chat",
        features_to_steer=it_only_features,
    )

    # steer the base activations with the IT only & base only features from chat decoder and continue with chat
    half_fns["?-steer_it_and_base_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="chat",
        continue_with="chat",
        features_to_steer=it_only_features + base_only_features,
    )

    # reconstruct chat and finish with chat

    return half_fns


def create_half_fn_dict_secondary(
    crosscoder: CrossCoder, it_only_features: list[int], base_only_features: list[int]
) -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    half_fns["base_reconstruct"] = CrossCoderReconstruction(
        crosscoder, reconstruct_with="base", continue_with="base"
    )
    # steer the base activations with the IT only features from chat decoder and continue with base
    half_fns["steer_it_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="chat",
        continue_with="base",
        features_to_steer=it_only_features,
    )
    # steer the base activations with the IT only & base only features from chat decoder and continue with base
    half_fns["steer_it_and_base_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="chat",
        continue_with="base",
        features_to_steer=it_only_features + base_only_features,
    )
    # steer the base activations with the base only features from chat decoder and continue with base
    half_fns["steer_base_only_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="chat",
        continue_with="base",
        features_to_steer=base_only_features,
    )
    # steer the base activations with all features from IT decoder and continue with base   ===   replace it error with base error
    half_fns["steer_all_to_base"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="chat",
        continue_with="base",
        features_to_steer=None,
    )
    # steer the base activations with the base only features from chat decoder and continue with chat
    half_fns["steer_base_only"] = CrossCoderSteeringFeature(
        crosscoder,
        steer_activations_of="base",
        steer_with_features_from="chat",
        continue_with="chat",
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
            steer_with_features_from="chat",
            continue_with="chat",
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
            steer_activations_of="chat",
            steer_with_features_from="base",
            continue_with="chat",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
        )
    return half_fns


def create_half_fn_dict_no_cross() -> dict[str, HalfStepPreprocessFn]:
    half_fns = {}
    half_fns["base"] = IdentityPreprocessFn(continue_with="base")
    half_fns["chat - debugging"] = IdentityPreprocessFn(continue_with="chat")
    half_fns["chat_to_base"] = SwitchPreprocessFn(continue_with="base")
    return half_fns


# def create_it_only_ft_fn_dict(
#     crosscoder: CrossCoder, it_only_features: list[int], base_only_features: list[int]
# ) -> dict[str, HalfStepPreprocessFn]:
#     half_fns = {
#         "remove_it_only_custom": CrossCoderSteeringFeature(
#             crosscoder,
#             steer_activations_of="chat",
#             steer_with_features_from="base",
#             continue_with="chat",
#             features_to_steer=it_only_features,
#         ),
#         "remove_it_and_base_only_custom": CrossCoderSteeringFeature(
#             crosscoder,
#             steer_activations_of="chat",
#             steer_with_features_from="base",
#             continue_with="chat",
#             features_to_steer=it_only_features + base_only_features,
#         ),
#         "steer_it_only_custom_to_base": CrossCoderSteeringFeature(
#             crosscoder,
#             steer_activations_of="base",
#             steer_with_features_from="chat",
#             continue_with="base",
#             features_to_steer=it_only_features,
#         ),
#         "steer_it_only_custom": CrossCoderSteeringFeature(
#             crosscoder,
#             steer_activations_of="base",
#             steer_with_features_from="chat",
#             continue_with="chat",
#             features_to_steer=it_only_features,
#         ),
#         "steer_it_and_base_only_custom": CrossCoderSteeringFeature(
#             crosscoder,
#             steer_activations_of="base",
#             steer_with_features_from="chat",
#             continue_with="chat",
#             features_to_steer=it_only_features + base_only_features,
#         ),
#     }
#     return half_fns


def create_half_fn_thresholded_features(
    crosscoder: CrossCoder,
    threshold: float = 10,
    features_to_steer: list[int] | None = None,
    steering_factor: float = 1,
):
    sf_name = f"_sf{steering_factor}" if steering_factor != 1 else ""
    if features_to_steer is None:
        features_to_steer = INTERESTING_FEATURES
    half_fns = {
        f"steer_t{threshold}{sf_name}_all": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="chat",
            continue_with="chat",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
            scale_steering_feature=steering_factor,
        ),
        f"rm_t{threshold}{sf_name}_all": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="chat",
            steer_with_features_from="base",
            continue_with="chat",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
            scale_steering_feature=steering_factor,
        ),
    }
    for feature in features_to_steer:
        half_fns[f"steer_t{threshold}{sf_name}_{feature}"] = CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="chat",
            continue_with="chat",
            features_to_steer=[feature],
            filter_treshold=threshold,
            scale_steering_feature=steering_factor,
        )
        half_fns[f"rm_t{threshold}{sf_name}_{feature}"] = CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="chat",
            steer_with_features_from="base",
            continue_with="chat",
            features_to_steer=[feature],
            filter_treshold=threshold,
            scale_steering_feature=steering_factor,
        )
    return half_fns


def create_tresholded_baseline_half_fns(
    crosscoder: CrossCoder,
    threshold: float = 10,
    features_to_steer: list[int] | None = None,
):
    if features_to_steer is None:
        features_to_steer = INTERESTING_FEATURES
    half_fns = {
        f"baseline_t{threshold}_all": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="chat",
            continue_with="chat",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
            ignore_encoder=False,
            scale_steering_feature=0,
        )
    }
    for feature in features_to_steer:
        half_fns[f"baseline_t{threshold}_{feature}"] = CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="base",
            steer_with_features_from="chat",
            continue_with="chat",
            features_to_steer=[feature],
            filter_treshold=threshold,
            ignore_encoder=False,
            scale_steering_feature=0,
        )
    return half_fns


def create_tresholded_it_baseline_half_fns(
    crosscoder: CrossCoder,
    threshold: float = 10,
    features_to_steer: list[int] | None = None,
):
    if features_to_steer is None:
        features_to_steer = INTERESTING_FEATURES
    half_fns = {
        f"it_baseline_t{threshold}_all": CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="chat",
            steer_with_features_from="chat",
            continue_with="chat",
            features_to_steer=features_to_steer,
            filter_treshold=threshold,
            ignore_encoder=False,
            scale_steering_feature=0,
        )
    }
    for feature in features_to_steer:
        half_fns[f"it_baseline_t{threshold}_{feature}"] = CrossCoderSteeringFeature(
            crosscoder,
            steer_activations_of="chat",
            steer_with_features_from="chat",
            continue_with="chat",
            features_to_steer=[feature],
            filter_treshold=threshold,
            ignore_encoder=False,
            scale_steering_feature=0,
        )
    return half_fns


def create_tresholded_baseline_random_half_fns(
    crosscoder: CrossCoder,
    seeds: list[int],
    num_features: int,
    threshold: float = 10,
):
    half_fns = {}
    for seed in seeds:
        th.manual_seed(seed)
        features_to_steer = th.randperm(crosscoder.decoder.weight.shape[1])[
            :num_features
        ]
        half_fns[f"baseline_t{threshold}_s{seed}_n{num_features}"] = (
            CrossCoderSteeringFeature(
                crosscoder,
                steer_activations_of="base",
                steer_with_features_from="chat",
                continue_with="chat",
                features_to_steer=features_to_steer,
                filter_treshold=threshold,
                ignore_encoder=False,
                scale_steering_feature=0,
            )
        )
    return half_fns


def create_tresholded_baseline_random_steering_half_fns(
    crosscoder: CrossCoder,
    seeds: list[int],
    threshold: float = 10,
    features_to_monitor: list[int] | None = None,
):
    # todo properly implement this
    raise NotImplementedError
    if features_to_monitor is None:
        features_to_monitor = INTERESTING_FEATURES
    half_fns = {}
    for seed in seeds:
        th.manual_seed(seed)
        features_to_steer = th.randperm(crosscoder.decoder.weight.shape[1])[
            : len(features_to_monitor)
        ]
        half_fns[f"steer_t{threshold}_s{seed}_n{len(features_to_monitor)}"] = (
            CrossCoderSteeringFeature(
                crosscoder,
                steer_activations_of="base",
                steer_with_features_from="chat",
                continue_with="chat",
                monitored_features=features_to_monitor,
                features_to_steer=features_to_steer,
                filter_treshold=threshold,
                ignore_encoder=True,
            )
        )
        half_fns[f"rm_t{threshold}_s{seed}_n{len(features_to_monitor)}"] = (
            CrossCoderSteeringFeature(
                crosscoder,
                steer_activations_of="chat",
                steer_with_features_from="base",
                continue_with="chat",
                monitored_features=features_to_monitor,
                features_to_steer=features_to_steer,
                filter_treshold=threshold,
            )
        )
    return half_fns
