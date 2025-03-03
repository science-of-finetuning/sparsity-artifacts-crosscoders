import json
from pathlib import Path
import re
import pandas as pd

LATENT_TYPE_NAMES = {
    "pareto": "Best",
    "antipareto": "Worst",
    "random-chat": "Random chat only",
    "random": "Random",
    "sae_pareto": "SAE Best",
    "sae_antipareto": "SAE Worst",
    "sae_pareto_nofilter": "SAE Best (no filter)",
    "sae_antipareto_nofilter": "SAE Worst (no filter)",
}
# Constants for setup name formatting
VANILLA_NAMES = {
    "base": "Base model only",
    "chat": "Chat model only",
    "base2chat": "Base → Chat switch",
    "chat2base": "Chat → Base switch",
}

PATCH_TYPE_NAMES = {
    "ctrl": "Chat template tokens",
    "first5": "First 5 predicted tokens",
    "ctrlfirst5": "Template & first 5 tokens",
}

COLUMN_NAMES = {
    "beta ratio reconstruction": "Reconstruction ratio",
    "beta ratio error": "Error ratio",
    "rank sum": "Both ratios",
    "base uselessness score": "Joint scalars",
    "dec norm diff": "Norm difference",
    "lmsys freq": "LMSys freq",
    "lmsys ctrl %": "LMSys ctrl %",
    "lmsys ctrl freq": "LMSys ctrl freq",
    "lmsys avg act": "LMSys avg act",
    "beta activation ratio": "Beta activation ratio",
    "beta activation chat": "Beta activation chat",
    "beta activation base": "Beta activation base",
    "beta error chat": "Beta error chat",
    "beta error base": "Beta error base",
    "beta activation ratio abs": "Absolute Beta activation ratio",
}


def normalize_key(key: str) -> str:
    """Normalize a key by replacing underscores and dashes with spaces."""
    return key.replace("_", " ").replace("-", " ")


def parse_key(key: str) -> dict:
    """Parses an intervention key to extract intuitive components."""
    # Try matching patch keys with detailed pattern
    pattern1 = r"^patch (?P<column>.*?) (?P<latents_type>[^ ]+) (?P<perc>[0-9.]+)pct (?P<patch_name>[^ ]+) (?P<patch_target>[^ ]+) c(?P<continue_with>.+)$"
    m = re.match(pattern1, key)
    if m:
        d = m.groupdict()
        d["kind"] = "patch"
        return d

    # Match patch_all keys
    pattern2 = r"^patch all(?P<add> add)? (?P<column>.*?) (?P<latents_type>[^ ]+) (?P<perc>[0-9.]+)pct(?:\+base only)? c(?P<continue_with>.+)$"
    m = re.match(pattern2, key)
    if m:
        d = m.groupdict()
        d["kind"] = "patch_all" if not d["add"] else "patch_all_add"
        d["patch_name"] = None
        d["patch_target"] = None
        d["has_base_only"] = "+base only" in key
        return d

    # Match SAE patch keys
    pattern_sae = r"^patch all sae (?P<latents_type>[^ ]+) c(?P<continue_with>.+)$"
    m = re.match(pattern_sae, key)
    if m:
        d = m.groupdict()
        d["kind"] = "sae"
        return d

    # Match patch_all_add keys
    pattern2b = r"^patch all add (?P<column>[^-]+) (?P<latents_type>[^-]+) (?P<perc>[0-9.]+)pct$"
    m = re.match(pattern2b, key)
    if m:
        d = m.groupdict()
        d["kind"] = "patch_all_add"
        d["patch_name"] = None
        d["patch_target"] = None
        d["has_base_only"] = False
        return d

    # Match error patching keys
    pattern3 = r"^patch (?P<model>base|chat) error c(?P<continue_with>.+)$"
    m = re.match(pattern3, key)
    if m:
        d = m.groupdict()
        d["kind"] = "error"
        return d

    # Match control token patching keys
    pattern4 = r"^patch (?P<model>base|chat) (?P<patch_type>ctrl|first5|ctrlfirst5) c(?P<continue_with>.+)$"
    m = re.match(pattern4, key)
    if m:
        d = m.groupdict()
        d["kind"] = "token_patch"
        return d

    # For vanilla interventions
    if key.startswith("vanilla "):
        variant = key[len("vanilla ") :]
        continue_with = None
        if variant in ["base", "chat"]:
            continue_with = variant
        elif "base2chat" in variant:
            continue_with = "base2chat"
        elif "chat2base" in variant:
            continue_with = "chat2base"
        return {"kind": "vanilla", "variant": variant, "continue_with": continue_with}

    print(f"Could not parse key: {key}")
    return {"kind": "unknown", "raw": key}


def format_setup_name(setup: str) -> str:
    """Convert internal setup name to readable format"""
    parsed = parse_key(setup)
    if parsed["kind"] == "vanilla":
        return f"Vanilla: {VANILLA_NAMES[parsed['variant']]}"
    elif parsed["kind"] == "patch":
        patch_desc = PATCH_TYPE_NAMES[parsed["patch_name"]]
        return f"Patch {patch_desc} from {parsed['patch_target']} model, continue with {parsed['continue_with']}"
    elif parsed["kind"] == "patch_all":
        latent_type = parsed["latents_type"]
        # Handle the case where we want to show mean across all random seeds
        if latent_type.startswith("random") and latent_type != "random":
            seed = latent_type[len("random") :]
            latent_type = f"Random (seed {seed})"
        elif latent_type == "random":
            latent_type = "Random (mean across seeds)"
        else:
            latent_type = LATENT_TYPE_NAMES.get(latent_type, latent_type)

        column_desc = COLUMN_NAMES.get(parsed["column"], parsed["column"])
        base_only_suffix = (
            " (with base only latents)" if parsed.get("has_base_only", False) else ""
        )
        return f"CrossCoder: Steer {latent_type} latents ({parsed['perc']}%) using {column_desc}{base_only_suffix}, continue with {parsed['continue_with']}"
    elif parsed["kind"] == "patch_all_add":
        latent_type = parsed["latents_type"]
        # Handle the case where we want to show mean across all random seeds
        if latent_type.startswith("random") and latent_type != "random":
            seed = latent_type[len("random") :]
            latent_type = f"Random (seed {seed})"
        elif latent_type == "random":
            latent_type = "Random (mean across seeds)"
        else:
            latent_type = LATENT_TYPE_NAMES.get(latent_type, latent_type)

        column_desc = COLUMN_NAMES.get(parsed["column"], parsed["column"])
        return f"CrossCoder: Steer by adding {latent_type} latents ({parsed['perc']}%) using {column_desc}"
    elif parsed["kind"] == "error":
        other_model = "chat" if parsed["model"] == "base" else "base"
        return f"{parsed['model'].capitalize()} error + {other_model} reconstruction, continue with {parsed['continue_with']}"
    elif parsed["kind"] == "token_patch":
        other_model = "chat" if parsed["model"] == "base" else "base"
        patch_type_desc = {
            "ctrl": "control tokens",
            "first5": "first 5 predicted tokens",
            "ctrlfirst5": "control & first 5 predicted tokens",
        }[parsed["patch_type"]]
        return f"Use {parsed['model']} but replace {patch_type_desc} with {other_model}, continue with {parsed['continue_with']}"
    elif parsed["kind"] == "sae":
        latent_type = parsed["latents_type"]
        if latent_type.startswith("random"):
            seed = latent_type[len("random"):]
            return f"SAE: Random latents (seed {seed}), continue with {parsed['continue_with']}"
        else:
            latent_desc = LATENT_TYPE_NAMES.get(f"sae_{latent_type}", latent_type)
            return f"SAE: {latent_desc} latents, continue with {parsed['continue_with']}"
    else:
        print(f"Unknown kind: {parsed['kind']}")
    return setup


def load_metrics(json_input) -> dict:
    """Load JSON metrics from a file path or file-like object and normalize the inner keys."""
    if isinstance(json_input, (str, Path)):
        with open(json_input, "r") as f:
            data = json.load(f)
    else:
        data = json.load(json_input)
    normalized_data = {}
    for category, metrics in data.items():
        normalized_data[category] = {}
        for metric, setups in metrics.items():
            normalized_data[category][metric] = {
                normalize_key(k): v for k, v in setups.items()
            }
    return normalized_data


def build_complete_dataframe(data: dict) -> pd.DataFrame:
    """Build a complete DataFrame containing all metrics across all categories."""
    all_data = {}
    for category in data:
        for metric_type in data[category]:
            category_data = data[category][metric_type]
            rows = []
            setups = []  # Track setups for index
            for setup in category_data:
                stats = category_data[setup]
                mean = stats.get("mean", None)
                var = stats.get("var", None)
                count = stats.get("count", None)
                if mean is not None and var is not None and count is not None:
                    rows.append(
                        {
                            "mean": mean,
                            "variance": var,
                            "n": count,
                        }
                    )
                    setups.append(setup)  # Add setup to index only when we add a row
            if rows:
                df = pd.DataFrame(rows, index=setups)  # Use collected setups as index
                # Add category and metric type to column names
                df.columns = pd.MultiIndex.from_product(
                    [[category], [metric_type], df.columns]
                )
                all_data[f"{category}_{metric_type}"] = df

    if all_data:
        return pd.concat(all_data.values(), axis=1)
    return None
