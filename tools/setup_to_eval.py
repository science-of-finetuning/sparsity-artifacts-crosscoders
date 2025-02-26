import torch as th
import numpy as np
from dictionary_learning.dictionary import CrossCoder
from tools.halfway_interventions import (
    HalfStepPreprocessFn,
    IdentityPreprocessFn,
    SwitchPreprocessFn,
    CrossCoderSteeringLatent,
    CrossCoderReconstruction,
    PatchCtrl,
    PatchKFirstPredictions,
    PatchKFirstAndCtrl,
    SteeringVector,
    PatchProjectionFromDiff,
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
            patch_target="base",
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
