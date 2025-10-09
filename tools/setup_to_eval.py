from typing import Literal
import torch as th
import numpy as np
import pandas as pd
from loguru import logger
from dictionary_learning.dictionary import CrossCoder, Dictionary
from tools.halfway_interventions import (
    HalfStepPreprocessFn,
    IdentityPreprocessFn,
    SwitchPreprocessFn,
    CrossCoderSteeringLatent,
    CrossCoderReconstruction,
    CrossCoderAdditiveSteering,
    PatchCtrl,
    PatchKFirstPredictions,
    PatchKFirstAndCtrl,
    SteeringVector,
    PatchProjectionFromDiff,
    SAEAdditiveSteering,
    SAEDifferenceReconstruction,
    SAEDifferenceReconstructionError,
    SAEDifferenceBiasSteering,
    SAEReconstruction,
)
from tools.cc_utils import load_latent_df

INTERESTING_LATENTS = [72073, 46325, 51408, 31726, 10833, 39938, 1045]


def create_acl_vanilla_half_fns() -> dict[str, HalfStepPreprocessFn]:
    """
    - Base
    - Base -> Chat
    - Chat -> Base
    - Chat
    """
    return {
        "vanilla_base": IdentityPreprocessFn(continue_with="base"),
        "vanilla_base2chat": SwitchPreprocessFn(continue_with="chat"),
        "vanilla_chat2base": SwitchPreprocessFn(continue_with="base"),
        "vanilla_chat": IdentityPreprocessFn(continue_with="chat"),
    }


def create_acl_patching_half_fns() -> dict[str, HalfStepPreprocessFn]:
    """
    - Base -> Chat, Chat -> Base
        - Patch the control tokens
        - Patch the 5 first predicted tokens
        - Patch the control tokens and the 5 first predicted tokens
    """
    half_fns = {}
    for name, patch_class in zip(
        ["ctrl", "first5", "ctrlfirst5"],
        [PatchCtrl, PatchKFirstPredictions, PatchKFirstAndCtrl],
    ):
        for patch_target in ["base", "chat"]:
            for continue_with in ["base", "chat"]:
                # Control token patching
                half_fns[f"patch_{patch_target}-{name}_c{continue_with}"] = patch_class(
                    continue_with=continue_with,
                    patch_target=patch_target,
                )
        half_fns[f"test_{name}_isbase"] = patch_class(
            continue_with="base",
            patch_target="base",
            activation_processor=IdentityPreprocessFn("base"),
        )
        half_fns[f"test_{name}_ischat"] = patch_class(
            continue_with="chat",
            patch_target="chat",
            activation_processor=IdentityPreprocessFn("chat"),
        )
    return half_fns


def create_acl_half_fns(
    crosscoder: CrossCoder,
    seeds: list[int],
    crosscoder_name: str | None = None,
    percentages: list[int] = [5, 10, 50, 100],
    columns: list[str] = [
        "rank_sum",
        "beta_ratio_reconstruction",
        "beta_ratio_error",
        "base uselessness score",
    ],
    skip_target_patch=False,
    skip_vanilla=False,
    skip_patching=False,
    add_base_only_latents=False,
) -> dict[str, HalfStepPreprocessFn]:
    """
    datasets:
    - Lmsys
    - Generated

    Creates the half functions used in the ACL paper:
    # Vanilla
    - Base
    - Base -> Chat
    - Chat -> Base
    - Chat

    # Patching
    - Base -> Chat, Chat -> Base
        - Patch the control tokens
        - Patch the 5 first predicted tokens
        - Patch the control tokens and the 5 first predicted tokens
    """
    half_fns = {}
    # --- Vanilla configurations ---
    if not skip_vanilla:
        half_fns.update(create_acl_vanilla_half_fns())

    # --- Patching interventions ---
    # Create configurations for different patching strategies
    if not skip_patching:
        half_fns.update(create_acl_patching_half_fns())

    # --- Crosscoder interventions ---
    half_fns_crosscoder, infos = create_acl_crosscoder_half_fns(
        crosscoder,
        seeds,
        crosscoder_name,
        percentages,
        columns,
        skip_target_patch,
        add_base_only_latents,
    )
    half_fns.update(half_fns_crosscoder)

    return half_fns, infos


def create_acl_crosscoder_half_fns(
    crosscoder: CrossCoder,
    seeds: list[int],
    crosscoder_name: str | None = None,
    percentages: list[int] = [5, 10, 50, 100],
    columns: list[str] = [
        "rank_sum",
        "beta_ratio_reconstruction",
        "beta_ratio_error",
        "base uselessness score",
    ],
    add_base_only_latents: bool = False,
    skip_target_patch=False,
):
    half_fns = {}
    # --- CrossCoder integrations ---
    """
    # CrossCoder
    - Continue with Chat, Base:
        - Patch all, patch control tokens, patch 5 first predicted tokens, patch control tokens and 5 first predicted tokens:
            - Base reconstruction + Chat Error
            - 5%, 10%, all:
                - Add x% chat only at the pareto frontier
                - Add x% random chat only * 5
                - Add x% chat only Anti-Pareto
    """
    for error_from in ["base", "chat"]:
        add_reconstruction_of = "base" if error_from == "chat" else "chat"
        for continue_with in ["base", "chat"]:
            preprocess_fn = CrossCoderSteeringLatent(
                crosscoder,
                steer_activations_of=error_from,
                steer_with_latents_from=add_reconstruction_of,
                continue_with=continue_with,
                latents_to_steer=None,  # all latents
            )
            half_fns[f"patch_{error_from}-error_c{continue_with}"] = preprocess_fn
            if skip_target_patch:
                continue
            for patch_target in ["base", "chat"]:
                for patch_name, patch_class in zip(
                    ["ctrl", "first5", "ctrlfirst5"],
                    [PatchCtrl, PatchKFirstPredictions, PatchKFirstAndCtrl],
                ):
                    half_fns[
                        f"patch_error_{patch_target}-{patch_name}_c{continue_with}"
                    ] = patch_class(
                        continue_with=continue_with,
                        patch_target=patch_target,
                        activation_processor=preprocess_fn,
                    )
    full_df = load_latent_df(crosscoder_name).query("lmsys_dead == False")
    base_only_latents = full_df.query("tag == 'Base only'").index.values
    assert len(base_only_latents) > 0, "No base only latents found"
    df = (
        full_df[
            [
                "beta_ratio_reconstruction",
                "beta_ratio_error",
                "tag",
                "base uselessness score",
                "dec_norm_diff",
                "lmsys_freq",
                "lmsys_ctrl_%",
                "lmsys_ctrl_freq",
                "lmsys_avg_act",
                "beta_activation_ratio",
                "beta_activation_chat",
                "beta_activation_base",
                "beta_error_chat",
                "beta_error_base",
            ]
        ]
        .dropna()
        .query("tag == 'IT only'")
        .query("-0.1 <= beta_ratio_reconstruction <= 2")
        .query("-0.1 <= beta_ratio_error <= 2")
    )
    assert (
        df.iloc[0]["dec_norm_diff"] < 0.2
    ), "Dec norm diff is reverted for this one so pareto and antipareto would be inverted, check code"
    # big is better for these columns
    bigger_is_better_cols = [
        "base uselessness score",
        "lmsys_freq",
        "lmsys_ctrl_%",
        "lmsys_ctrl_freq",
        "lmsys_avg_act",
        "beta_activation_chat",
        "beta_error_chat",
    ]
    rank_sum = df["beta_ratio_reconstruction"].rank() + df["beta_ratio_error"].rank()
    df["rank_sum"] = rank_sum
    full_df["rank_sum"] = np.nan
    full_df.loc[df.index.values, "rank_sum"] = rank_sum
    rnd_latents_dict = {}
    infos = {"rnd latents": rnd_latents_dict}
    for perc in percentages:
        random_latents_list = []
        random_latents_types = []
        infos[f"{perc}%"] = {}
        num_latents = int(len(df) * (perc / 100) + 0.5)
        print(f"{perc}%: {num_latents}")
        for seed in seeds:
            np.random.seed(seed)
            random_chat_latents = np.random.permutation(df.index.values)[:num_latents]
            random_latents_list.append(random_chat_latents)
            random_latents_types.append(f"random-chat{seed}")
        for seed in seeds:
            np.random.seed(seed)
            perm = np.random.permutation(full_df.index.values)
            random_latents = perm[:num_latents]
            random_latents_list.append(random_latents)
            random_latents_types.append(f"random{seed}")
        rnd_latents_dict[perc] = {
            rnd_type: rl.tolist()
            for rl, rnd_type in zip(random_latents_list, random_latents_types)
        }
        for column in columns:
            print(f"===== {perc} =====")
            bigger_is_better = column in bigger_is_better_cols
            sorted_index = df.sort_values(
                by=column, ascending=not bigger_is_better
            ).index.values
            pareto_latents = sorted_index[:num_latents]
            antipareto_latents = sorted_index[len(df) - num_latents :]
            threshold_low = sorted_index[num_latents - 1]
            threshold_high = sorted_index[len(df) - num_latents]
            infos[f"{perc}%"][column] = {
                "bigger_is_better": bigger_is_better,
                "threshold_pareto": float(threshold_low),
                "threshold_antipareto": float(threshold_high),
                "pareto_latents": pareto_latents.tolist(),
                "antipareto_latents": antipareto_latents.tolist(),
                "pareto_values": df.loc[pareto_latents][column].values.tolist(),
                "antipareto_values": df.loc[antipareto_latents][column].values.tolist(),
            }
            latents_setups = [
                pareto_latents,
                antipareto_latents,
            ]
            latents_types = ["pareto", "antipareto"]

            if column == columns[0]:
                latents_setups.extend(random_latents_list)
                latents_types.extend(random_latents_types)
                for random_latents, random_latents_type in zip(
                    random_latents_list, random_latents_types
                ):
                    infos[f"{perc}%"][column][random_latents_type] = {
                        "values": full_df.loc[random_latents][column].values.tolist(),
                    }
            print(f"len pareto: {len(pareto_latents)}")
            print(f"len antipareto: {len(antipareto_latents)}")
            print("================\n")
            for latents, latents_type in zip(latents_setups, latents_types):
                if latents_type != "pareto" and perc == 100:
                    if not ("random" in latents_type and "chat" not in latents_type):
                        continue
                if add_base_only_latents:
                    latents = np.concatenate([latents, base_only_latents])
                preprocess_fn = CrossCoderSteeringLatent(
                    crosscoder,
                    steer_activations_of="base",
                    steer_with_latents_from="chat",
                    continue_with="chat",  # doesn't matter as we use .result in mask
                    latents_to_steer=latents,
                )
                name = f"{column}-{latents_type}-{perc}pct"
                if add_base_only_latents:
                    name += "+base_only"
                for continue_with in ["base", "chat"]:
                    half_fns[f"patch_all_{name}_c{continue_with}"] = (
                        CrossCoderSteeringLatent(
                            crosscoder,
                            steer_activations_of="base",
                            steer_with_latents_from="chat",
                            continue_with=continue_with,
                            latents_to_steer=latents,
                        )
                    )
                    if skip_target_patch:
                        continue

                    for patch_target in ["base", "chat"]:
                        for patch_name, patch_class in zip(
                            ["ctrl", "first5", "ctrlfirst5"],
                            [PatchCtrl, PatchKFirstPredictions, PatchKFirstAndCtrl],
                        ):
                            half_fns[
                                f"patch_{name}->{patch_name}-{patch_target}_c{continue_with}"
                            ] = patch_class(
                                continue_with=continue_with,
                                patch_target=patch_target,
                                activation_processor=preprocess_fn,
                            )

    return half_fns, infos


def baseline_diffs_half_fns(
    mean_diff: th.Tensor, pca_vectors: list[th.Tensor]
) -> dict[str, HalfStepPreprocessFn]:

    half_fns = {
        "steer_mean_diff": SteeringVector(
            continue_with="chat",
            steer_activations_of="base",
            vector=mean_diff,
        )
    }
    for i, vector in enumerate(pca_vectors):
        half_fns[f"add_diff-pca_proj_{i}"] = PatchProjectionFromDiff(
            continue_with="chat",
            patch_target="base",
            vector=vector,
        )
        half_fns[f"rm_diff-pca_proj_{i}"] = PatchProjectionFromDiff(
            continue_with="chat",
            patch_target="chat",
            vector=vector,
        )
        half_fns[f"add_diff-pca_proj_:{i}"]
    return half_fns


def threshold_half_fns(
    full_df: pd.DataFrame,
    threshold: float,
    column: str,
    dictionary: Dictionary,
    is_crosscoder: bool,
    take_below: bool = True,
    sae_model: Literal["base", "chat"] | None = None,
    steering_scale: float = 1,
    is_difference_sae: bool = False,
) -> dict[str, HalfStepPreprocessFn]:
    """Creates a half function for steering using latents below a threshold of a given column.

    Args:
        full_df: Latent dataframe
        threshold: Value to filter latents by
        column: Column name to apply threshold to
        dictionary: Dictionary object containing the latent vectors
        is_crosscoder: Whether dictionary is a crosscoder
        take_below: If True, take latents below threshold, otherwise take above

    Returns:
        Dictionary with a single half function that steers using latents filtered by threshold and column.
    """
    comparison = "<=" if take_below else ">="
    latents = full_df.query(f"{column} {comparison} {threshold}").index.tolist()

    direction = "blw" if take_below else "abv"

    if is_crosscoder:
        op_name = "add"
        interv = CrossCoderAdditiveSteering(
            dictionary,
            steer_activations_of="base",
            steer_with_latents_from="chat",
            continue_with="chat",
            latents_to_steer=latents,
            scale_steering_latent=steering_scale,
        )
    else:
        if sae_model is None:
            raise ValueError("sae_model must be provided for sae steering")
        op_name = "sae"
        interv = SAEAdditiveSteering(
            dictionary,
            steer_activations_of="base",
            continue_with="chat",
            latents_to_steer=latents,
            sae_model=sae_model,
            scale_steering_latent=steering_scale,
            is_difference_sae=is_difference_sae,
        )
    infos = {
        f"{column}-{direction}{threshold}-latents": latents,
    }
    return {f"patch_all_{op_name}_{column}-{direction}{threshold}-cchat": interv}, infos


def baselines_half_fns():
    half_fns = {
        "vanilla_base": IdentityPreprocessFn(continue_with="base"),
        "vanilla_chat": IdentityPreprocessFn(continue_with="chat"),
        "vanilla_base2chat": SwitchPreprocessFn(continue_with="chat"),
        "vanilla_chat2base": SwitchPreprocessFn(continue_with="base"),
    }
    for continue_with in ["base", "chat"]:
        half_fns[f"patch_base-ctrl_c{continue_with}"] = PatchCtrl(
            continue_with=continue_with,
            patch_target="base",
            activation_processor=None,
        )
    return half_fns


def arxiv_paper_half_fns(
    crosscoder: CrossCoder,
    full_df: pd.DataFrame,
    crosscoder_name: str | None = None,
    add_base_only_latents: bool = False,
) -> dict[str, HalfStepPreprocessFn]:
    """Creates the half functions used in the arxiv paper:
    - Top 50% and flop 50% using norm diff and latent scaling
    - Configurations: base -> chat, base, chat, chat/base error patching
    """
    if add_base_only_latents:
        raise NotImplementedError("Base only latents not implemented for arxiv paper")
    half_fns = {}
    infos = {}
    # --- Vanilla configurations ---
    half_fns.update(baselines_half_fns())
    # --- Error and all ----
    for error_from in ["base", "chat"]:
        add_reconstruction_of = "base" if error_from == "chat" else "chat"
        preprocess_fn = CrossCoderSteeringLatent(
            crosscoder,
            steer_activations_of=error_from,
            steer_with_latents_from=add_reconstruction_of,
            continue_with="chat",
            latents_to_steer=None,  # all latents
        )
        half_fns[f"patch_{error_from}-error_cchat"] = preprocess_fn

    # Load and prepare data
    if "lmsys_dead" in full_df.columns:
        full_df = full_df.query("lmsys_dead == False")
    df = (
        full_df[
            [
                "beta_ratio_reconstruction",
                "beta_ratio_error",
                "tag",
                "dec_norm_diff",
            ]
        ]
        .dropna()
        .query("tag in ['Chat only', 'IT only']")
        .query("-0.1 <= beta_ratio_reconstruction <= 2")
        .query("-0.1 <= beta_ratio_error <= 2")
    )
    print(f"len df: {len(df)}")
    base_only_latents = full_df.query("tag == 'Base only'").index.values

    # Get top/flop 50% based on norm diff
    sorted_by_norm = df.sort_values(by="dec_norm_diff", ascending=True)
    num_latents = len(df) // 2
    top_norm_latents = sorted_by_norm.index.values[:num_latents]
    flop_norm_latents = sorted_by_norm.index.values[-num_latents:]

    # Get top/flop 50% based on rank sum
    rank_sum = df["beta_ratio_reconstruction"].rank() + df["beta_ratio_error"].rank()
    df["rank_sum"] = rank_sum
    sorted_by_ratios = df.sort_values(by="rank_sum", ascending=True)
    full_df["rank_sum"] = np.nan
    full_df.loc[df.index.values, "rank_sum"] = rank_sum
    top_ratios_latents = sorted_by_ratios.index.values[:num_latents]
    flop_ratios_latents = sorted_by_ratios.index.values[-num_latents:]
    rnd_latents_dict = {}
    infos = {
        "rnd latents": rnd_latents_dict,
        "base only latents": base_only_latents.tolist(),
    }

    for column, pareto_50, antipareto_50 in [
        ("dec_norm_diff", top_norm_latents, flop_norm_latents),
        ("rank_sum", top_ratios_latents, flop_ratios_latents),
    ]:
        infos[column] = {
            "pareto_latents": pareto_50.tolist(),
            "antipareto_latents": antipareto_50.tolist(),
            "pareto_values": df.loc[pareto_50][column].values.tolist(),
            "antipareto_values": df.loc[antipareto_50][column].values.tolist(),
        }
        latents_setups = [
            pareto_50,
            antipareto_50,
        ]
        latents_types = ["pareto", "antipareto"]

        print(f"len pareto: {len(pareto_50)}")
        print(f"len antipareto: {len(antipareto_50)}")
        print("================\n")
        for latents, latents_type in zip(latents_setups, latents_types):
            name = f"{column}-{latents_type}-50pct-cchat"
            half_fns[f"patch_all_{name}"] = CrossCoderSteeringLatent(
                crosscoder,
                steer_activations_of="base",
                steer_with_latents_from="chat",
                continue_with="chat",
                latents_to_steer=latents,
            )
            half_fns[f"patch_all_add_{name}"] = CrossCoderAdditiveSteering(
                crosscoder,
                steer_activations_of="base",
                steer_with_latents_from="chat",
                continue_with="chat",
                latents_to_steer=latents,
            )
    full_df["beta_activation_ratio_abs"] = full_df["beta_activation_ratio"].abs()
    for column, threshold, take_below in [
        ("dec_norm_diff", 0.3, True),
        ("beta_activation_ratio_abs", 0.6, True),
        ("beta_activation_ratio_abs", 0.3, True),
    ]:
        fn, info = threshold_half_fns(
            full_df,
            threshold,
            column,
            crosscoder,
            is_crosscoder=True,
            take_below=take_below,
        )
        half_fns.update(fn)
        infos[next(iter(fn.keys()))] = info
    return half_fns, infos


def sae_steering_half_fns(
    sae: Dictionary,
    seeds: list[int],
    full_df: pd.DataFrame,
    num_latents: int,
    sae_model: Literal["base", "chat"],
    is_difference_sae: bool = False,
) -> dict[str, HalfStepPreprocessFn]:
    """
    Creates the half functions for SAE steering.

    Args:
        sae: The SAE model to use
        seeds: The seeds to use for random steering
        full_df: The full dataframe to use
        num_latents: The number of latents to steer, should be equal to the number of latents you steer in your crosscoder
        is_difference_sae: Whether the SAE is a difference SAE
    Returns:
        A dictionary of half functions for SAE steering containing:
        - patch_all_sae_random{seed}_cchat: Half function for random steering
        - patch_all_sae_(anti)pareto_(nofilter_)cchat: Half function for steering using the top num_latents or the top 2*num_latents without the top num_latents. No filter means that the top num_latents are not filtered for -0.1 <= beta_activation_ratio <= 2 and abs(beta_activation_ratio) is used instead.
        - patch_diff_sae_recon_cchat: Half function for steering using the difference SAE reconstruction if the SAE is a difference SAE.

    """
    half_fns = {}
    infos = {"num_latents": num_latents, "random latents": {}}
    full_df["beta_activation_ratio_abs"] = full_df["beta_activation_ratio"].abs()
    steering_scale = 1
    if sae_model == "base" and is_difference_sae:
        logger.info(
            "Using negative steering as you're evaluating a base - chat SAE, therefore chat specific latents will have be negated"
        )
        steering_scale = -1
    for seed in seeds:
        np.random.seed(seed)
        random_latents = np.random.permutation(full_df.index.values)[:num_latents]
        # Sort random latents by beta_activation_ratio
        random_latents = random_latents[
            np.argsort(full_df.loc[random_latents]["beta_activation_ratio_abs"].values)
        ]
        half_fns[f"patch_all_sae_random{seed}_cchat"] = SAEAdditiveSteering(
            sae,
            steer_activations_of="base",
            continue_with="chat",
            latents_to_steer=random_latents,
            sae_model=sae_model,
            scale_steering_latent=steering_scale,
            is_difference_sae=is_difference_sae,
        )
        if is_difference_sae:
            half_fns[f"patch_all_sae_random{seed}_wbias_cchat"] = SAEAdditiveSteering(
                sae,
                steer_activations_of="base",
                continue_with="chat",
                latents_to_steer=random_latents,
                sae_model=sae_model,
                scale_steering_latent=steering_scale,
                is_difference_sae=is_difference_sae,
                add_decoder_bias=True,
            )

        infos["random latents"][f"random{seed}"] = {
            "latents": random_latents.tolist(),
            "values": full_df.loc[random_latents][
                "beta_activation_ratio"
            ].values.tolist(),
        }
    # note: No filter in the name is for retrocompatibility with the old runs that filtered for latents with -0.1 <= beta_activation_ratio <= 2
    best_nofilter_latents = full_df.sort_values(
        by="beta_activation_ratio_abs", ascending=True
    ).index.values[:num_latents]
    half_fns["patch_all_sae_pareto_nofilter_cchat"] = SAEAdditiveSteering(
        sae,
        steer_activations_of="base",
        continue_with="chat",
        latents_to_steer=best_nofilter_latents,
        sae_model=sae_model,
        scale_steering_latent=steering_scale,
        is_difference_sae=is_difference_sae,
    )
    if is_difference_sae:
        half_fns["patch_all_sae_pareto_nofilter_wbias_cchat"] = SAEAdditiveSteering(
            sae,
            steer_activations_of="base",
            continue_with="chat",
            latents_to_steer=best_nofilter_latents,
            sae_model=sae_model,
            scale_steering_latent=steering_scale,
            is_difference_sae=is_difference_sae,
            add_decoder_bias=True,
        )
    infos["best nofilter latents"] = {
        "latents": best_nofilter_latents.tolist(),
        "values": full_df.loc[best_nofilter_latents][
            "beta_activation_ratio"
        ].values.tolist(),
    }

    worst_nofilter_latents = full_df.sort_values(
        by="beta_activation_ratio_abs", ascending=True
    ).index.values[num_latents : num_latents * 2]
    half_fns["patch_all_sae_antipareto_nofilter_cchat"] = SAEAdditiveSteering(
        sae,
        steer_activations_of="base",
        continue_with="chat",
        latents_to_steer=worst_nofilter_latents,
        sae_model=sae_model,
        scale_steering_latent=steering_scale,
        is_difference_sae=is_difference_sae,
    )
    if is_difference_sae:
        half_fns["patch_all_sae_antipareto_nofilter_wbias_cchat"] = SAEAdditiveSteering(
            sae,
            steer_activations_of="base",
            continue_with="chat",
            latents_to_steer=worst_nofilter_latents,
            sae_model=sae_model,
            scale_steering_latent=steering_scale,
            is_difference_sae=is_difference_sae,
            add_decoder_bias=True,
        )
    infos["worst nofilter latents"] = {
        "latents": worst_nofilter_latents.tolist(),
        "values": full_df.loc[worst_nofilter_latents][
            "beta_activation_ratio"
        ].values.tolist(),
    }
    # Apply threshold for beta_activation_ratio at 0.6 and 0.3
    for threshold in [0.6, 0.3, 0.1]:
        fn, info = threshold_half_fns(
            full_df,
            threshold,
            "beta_activation_ratio_abs",
            sae,
            is_crosscoder=False,
            take_below=True,
            sae_model=sae_model,
            steering_scale=steering_scale,
            is_difference_sae=is_difference_sae,
        )
        half_fns.update(fn)
        infos[f"threshold_{threshold}"] = info
    # Add special reconstruction steering for difference SAEs
    if is_difference_sae:
        half_fns["patch_diff_sae_recon_cchat"] = SAEDifferenceReconstruction(
            sae,
            continue_with="chat",
            steer_activations_of="base",
            sae_model=sae_model,
        )
        half_fns["patch_diff_sae_recon_error_cchat"] = SAEDifferenceReconstructionError(
            sae,
            continue_with="chat",
            steer_activations_of="base",
            sae_model=sae_model,
        )
        half_fns["patch_diff_sae_bias_steering_cchat"] = SAEDifferenceBiasSteering(
            sae,
            continue_with="chat",
            steer_activations_of="base",
            sae_model=sae_model,
        )
    else:
        half_fns["patch_chat_recon_cchat"] = SAEReconstruction(
            sae,
            use_reconstruction_of="chat",
            continue_with="chat",
        )
        half_fns["patch_base_recon_cchat"] = SAEReconstruction(
            sae,
            use_reconstruction_of="base",
            continue_with="chat",
        )
    infos["is_difference_sae"] = is_difference_sae
    return half_fns, infos
