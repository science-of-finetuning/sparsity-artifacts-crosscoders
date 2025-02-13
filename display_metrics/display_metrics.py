import streamlit as st
import json
import re
import pandas as pd
import argparse
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import os

# Constants for state management
CACHE_DIR = Path("display_metrics/cache")
STATE_FILE = CACHE_DIR / "state.json"

PATCH_TYPE_NAMES = {
    "ctrl": "Chat template tokens",
    "first5": "First 5 predicted tokens",
    "ctrlfirst5": "Template & first 5 tokens",
}

VANILLA_NAMES = {
    "base": "Base model only",
    "chat": "Chat model only",
    "base2chat": "Base → Chat switch",
    "chat2base": "Chat → Base switch",
}

DEFAULT_METRIC_ORDER = ["kl-instruct", "loss", "kl-base", "low_wrt_instruct_pred"]

RESULTS_DIR = "results/interv_effects"


def load_state():
    """Load UI state from cache"""
    default_state = {
        "selected_setups": [],
        "metric_order": [],
        "expanded_metrics": set(),
        "current_file": None,
        "hidden_setups": set(),
        "selected_categories": None,
    }

    if not STATE_FILE.exists():
        return default_state

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

        # Validate state structure
        if not isinstance(state, dict):
            return default_state

        # Ensure all required keys exist
        for key in default_state:
            if key not in state:
                state[key] = default_state[key]

        # Convert expanded_metrics to set if it exists
        if "expanded_metrics" in state:
            try:
                state["expanded_metrics"] = set(state["expanded_metrics"])
            except (TypeError, ValueError):
                state["expanded_metrics"] = set()

        return state
    except (json.JSONDecodeError, OSError):
        # If there's any error reading or parsing the file, return default state
        if STATE_FILE.exists():
            try:
                STATE_FILE.unlink()  # Delete corrupted file
            except OSError:
                pass
        return default_state


def save_state(state):
    """Save UI state to cache"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Create a sanitized copy for saving
        state_to_save = {
            "selected_setups": list(state.get("selected_setups", [])),
            "metric_order": list(state.get("metric_order", [])),
            "expanded_metrics": list(state.get("expanded_metrics", set())),
            "current_file": state.get("current_file"),
            "selected_categories": list(state.get("selected_categories", [])),
        }

        with open(STATE_FILE, "w") as f:
            json.dump(state_to_save, f)
    except (OSError, TypeError) as e:
        # If we can't save the state, log the error but don't crash
        print(f"Error saving state: {e}")
        if STATE_FILE.exists():
            try:
                STATE_FILE.unlink()  # Delete potentially corrupted file
            except OSError:
                pass


def parse_key(key: str) -> dict:
    """Parses an intervention key to extract intuitive components."""
    # Try matching patch keys with detailed pattern
    pattern1 = r"^patch_(?P<latents_type>[^-]+)-(?P<perc>[0-9.]+)pct->(?P<patch_name>[^-]+)-(?P<patch_target>[^_]+)_c(?P<continue_with>.+)$"
    m = re.match(pattern1, key)
    if m:
        d = m.groupdict()
        d["kind"] = "patch"
        return d

    # Match patch_all keys
    pattern2 = r"^patch_all-(?P<latents_type>[^-]+)-(?P<perc>[0-9.]+)pct-c(?P<continue_with>.+)$"
    m = re.match(pattern2, key)
    if m:
        d = m.groupdict()
        d["kind"] = "patch_all"
        d["patch_name"] = None
        d["patch_target"] = None
        return d

    # For vanilla interventions
    if key.startswith("vanilla_"):
        variant = key[len("vanilla_") :]
        # Attempt to derive continue_with from known variants
        continue_with = None
        if variant in ["base", "chat"]:
            continue_with = variant
        elif "base2chat" in variant:
            continue_with = "base2chat"
        elif "chat2base" in variant:
            continue_with = "chat2base"
        return {"kind": "vanilla", "variant": variant, "continue_with": continue_with}

    # If not matched, return raw key
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
            latent_type = latent_type.capitalize()
        return f"CrossCoder: Steer {latent_type} latents ({parsed['perc']}%), continue with {parsed['continue_with']}"
    return setup


def setup_selector():
    """Create the hierarchical setup selector"""
    setup_type = st.selectbox("Setup Type", ["Vanilla", "Patching", "CrossCoder"])

    if setup_type == "Vanilla":
        model = st.selectbox(
            "Model", list(VANILLA_NAMES.keys()), format_func=lambda x: VANILLA_NAMES[x]
        )
        return f"vanilla_{model}"

    elif setup_type == "Patching":
        patch_type = st.selectbox(
            "What to patch",
            list(PATCH_TYPE_NAMES.keys()),
            format_func=lambda x: PATCH_TYPE_NAMES[x],
        )
        patch_target = st.selectbox("Target model", ["base", "chat"])
        continue_with = st.selectbox("Continue generation with", ["base", "chat"])
        return f"patch_{patch_target}-{patch_type}_c{continue_with}"

    elif setup_type == "CrossCoder":
        percentage = st.selectbox("Percentage", ["5", "10", "100"])
        latent_type = st.selectbox("Latent Type", ["pareto", "random", "antipareto"])
        patch_option = st.selectbox(
            "What to patch",
            ["all"] + list(PATCH_TYPE_NAMES.keys()),
            format_func=lambda x: (
                "All activations" if x == "all" else PATCH_TYPE_NAMES[x]
            ),
        )
        continue_with = st.selectbox("Continue generation with", ["base", "chat"])
        patch_target = (
            None
            if patch_option == "all"
            else st.selectbox("Target model", ["base", "chat"])
        )

        # Add seed selector if random is selected
        if latent_type == "random":
            seed = st.selectbox("Random Seed", ["all", "0", "1", "2", "3", "4"])
            if seed != "all":
                latent_type = f"random{seed}"

        if patch_option == "all":
            return f"patch_all-{latent_type}-{percentage}pct-c{continue_with}"
        else:
            return f"patch_{latent_type}-{percentage}pct->{patch_option}-{patch_target}_c{continue_with}"


def load_metrics(json_file) -> dict:
    """Loads JSON metrics from a file-like object or path."""
    if isinstance(json_file, (str, Path)):
        with open(json_file, "r") as f:
            return json.load(f)
    return json.load(json_file)


def create_metric_plot(df, metric_name, categories):
    """Create a plotly histogram with error bars for multiple categories"""
    fig = go.Figure()

    # Calculate bar positions
    n_setups = len(df.index.unique())
    bar_width = 0.8 / n_setups  # Adjust total width of group
    offsets = np.linspace(
        -(n_setups - 1) * bar_width / 2, (n_setups - 1) * bar_width / 2, n_setups
    )

    # For each setup, add bars for all categories
    for setup_idx, setup in enumerate(df.index.unique()):
        x_positions = []
        y_values = []
        error_values = []

        for cat_idx, category in enumerate(categories):
            if category not in df.columns:
                continue

            category_data = df[category]
            if setup in category_data.index:
                mean = category_data.loc[setup, "mean"]
                var = category_data.loc[setup, "variance"]
                n = category_data.loc[setup, "n"]
                ci = 1.96 * (var / n) ** 0.5

                x_positions.append(cat_idx + offsets[setup_idx])
                y_values.append(mean)
                error_values.append(ci)

        if y_values:  # Only add trace if we have data
            fig.add_trace(
                go.Bar(
                    name=format_setup_name(setup),
                    x=x_positions,
                    y=y_values,
                    width=bar_width,
                    error_y=dict(type="data", array=error_values, visible=True),
                    hovertemplate=f"Internal name: {setup}<br>Value: %{{y:.3f}}<br>Error: %{{error_y.array:.3f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=metric_name,
        showlegend=True,
        height=400,
        xaxis=dict(
            ticktext=categories, tickvals=list(range(len(categories))), title="Category"
        ),
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        bargap=0,  # Remove gap between bar groups
    )

    return fig


def display_metrics(data, current_file=None):
    """Main display logic for the metrics data."""
    # Display current file if provided
    if current_file:
        st.markdown(f"**Current file:** `{current_file}`")
        st.markdown("---")

    # Add CSS for tooltips and buttons
    st.markdown(
        """
        <style>
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #666;
        }
        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 5px;
            background-color: black;
            color: white;
            border-radius: 4px;
            font-size: 14px;
            white-space: nowrap;
            z-index: 1;
        }
        .stButton > button {
            margin: 0 4px;  /* Add horizontal spacing between buttons */
            padding: 0.25rem 0.5rem;  /* Make buttons more compact */
            height: auto;  /* Allow button height to be determined by content */
            line-height: 1.2;  /* Adjust line height for better text alignment */
        }
        .setup-container {
            display: flex;
            align-items: center;
            gap: 1rem;  /* Add consistent spacing between elements */
        }
        .setup-name {
            flex-grow: 1;
        }
        .setup-buttons {
            display: flex;
            gap: 0.5rem;  /* Space between buttons */
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Load/initialize state
    if "state" not in st.session_state:
        st.session_state.state = load_state()
    if "expanded_metrics" not in st.session_state.state:
        st.session_state.state["expanded_metrics"] = set()

    st.markdown("### Setup Selection")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_setup = setup_selector()
    with col2:
        if st.button("Add Setup"):
            if new_setup not in st.session_state.state["selected_setups"]:
                st.session_state.state["selected_setups"].append(new_setup)
                save_state(st.session_state.state)
                st.rerun()

    st.markdown("### Selected Setups")
    need_rerun = False
    for i, setup in enumerate(st.session_state.state["selected_setups"]):
        with st.container():
            col1, col2 = st.columns([6, 2])
            with col1:
                st.markdown(
                    f'<div class="tooltip" data-tooltip="{setup}">{format_setup_name(setup)}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                # Initialize hidden_setups if it doesn't exist
                if "hidden_setups" not in st.session_state.state:
                    st.session_state.state["hidden_setups"] = set()
                is_hidden = setup in st.session_state.state["hidden_setups"]
                
                # Place buttons side by side with minimal spacing
                c1, c2 = st.columns([1, 1])
                with c1:
                    button_label = "Hidden" if is_hidden else "Shown"
                    if st.button(
                        button_label,
                        key=f"toggle_{i}",
                        use_container_width=True,
                        type="primary" if is_hidden else "secondary",
                        help=f"Click to {'show' if is_hidden else 'hide'} this setup",
                    ):
                        if is_hidden:
                            st.session_state.state["hidden_setups"].remove(setup)
                        else:
                            st.session_state.state["hidden_setups"].add(setup)
                        save_state(st.session_state.state)
                        need_rerun = True
                with c2:
                    if st.button("Remove", key=f"remove_{i}", use_container_width=True):
                        st.session_state.state["selected_setups"].pop(i)
                        if setup in st.session_state.state["hidden_setups"]:
                            st.session_state.state["hidden_setups"].remove(setup)
                        save_state(st.session_state.state)
                        need_rerun = True

    if need_rerun:
        st.rerun()

    if not st.session_state.state["selected_setups"]:
        st.info("Please add some setups to compare")
        return

    st.markdown("### Metrics")
    # Replace the category selection code with:
    st.markdown("Select categories to display:")
    categories = list(data.keys())
    
    # Initialize selected categories in state
    if "selected_categories" not in st.session_state.state:
        st.session_state.state["selected_categories"] = categories  # Default all selected
    
    # Create checkboxes and update state
    cols = st.columns(3)
    selected_categories = []
    for i, category in enumerate(categories):
        with cols[i % 3]:
            # Default to True if not in state yet
            is_checked = category in st.session_state.state["selected_categories"]
            checkbox = st.checkbox(category, value=is_checked, key=f"cat_{category}")
            if checkbox:
                selected_categories.append(category)
    
    # Update state with new selections
    st.session_state.state["selected_categories"] = selected_categories
    save_state(st.session_state.state)

    if not selected_categories:
        st.info("Please select at least one category")
        return

    # Sort metrics according to default order
    metrics = set()
    for category in selected_categories:
        metrics.update(data[category].keys())
    metrics = sorted(
        list(metrics),
        key=lambda x: (
            DEFAULT_METRIC_ORDER.index(x)
            if x in DEFAULT_METRIC_ORDER
            else len(DEFAULT_METRIC_ORDER)
        ),
    )

    # For each metric, create a plot combining all selected categories
    for metric_type in metrics:
        # Check if this metric should be expanded
        is_expanded = metric_type in st.session_state.state["expanded_metrics"]
        with st.expander(metric_type, expanded=is_expanded) as exp:
            # Update expanded state
            if exp:
                st.session_state.state["expanded_metrics"].add(metric_type)
            else:
                st.session_state.state["expanded_metrics"].discard(metric_type)
            save_state(st.session_state.state)

            # Build a dataframe for selected setups across all categories
            all_data = {}
            for category in selected_categories:
                if metric_type not in data[category]:
                    continue

                category_data = data[category][metric_type]
                visible_setups = [s for s in st.session_state.state["selected_setups"] if s not in st.session_state.state.get("hidden_setups", set())]
                rows = []
                for setup in visible_setups:
                    if setup in category_data:
                        stats = category_data[setup]
                        rows.append({
                            "mean": stats.get("mean", None),
                            "variance": stats.get("var", None),
                            "n": stats.get("count", None),
                        })
                if rows:
                    all_data[category] = pd.DataFrame(rows, index=visible_setups)

            if all_data:
                # Combine all categories into one dataframe
                df = pd.concat(all_data, axis=1)
                fig = create_metric_plot(df, metric_type, selected_categories)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for selected setups and categories")


def get_available_result_files():
    """Get all available JSON result files from the results/interv_effects directory."""
    result_files = []
    for root, _, files in os.walk(RESULTS_DIR):
        for file in files:
            if file.endswith('_result.json') or file.endswith('_result_fixed.json'):
                rel_path = os.path.relpath(os.path.join(root, file), start='.')
                result_files.append(rel_path)
    return sorted(result_files)


def filter_files(files, search_pattern):
    """Filter files based on search pattern with implicit wildcards."""
    if not search_pattern:
        return files
    
    # Escape special regex characters in the search pattern
    escaped_pattern = re.escape(search_pattern)
    # Add .* before and after the pattern
    full_pattern = f".*{escaped_pattern}.*"
    try:
        regex = re.compile(full_pattern, re.IGNORECASE)
        return [f for f in files if regex.search(f)]
    except re.error:
        # If regex compilation fails, return all files
        return files


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Display evaluation metrics from JSON file"
    )
    parser.add_argument(
        "--path",
        "-p",
        help="Path to the JSON file containing evaluation results",
        default=None,
    )
    args = parser.parse_args()

    st.title("Evaluation Metrics Display")

    # Initialize session state
    if 'state' not in st.session_state:
        st.session_state.state = load_state()
        # Only use args.path if there's no saved current_file
        if st.session_state.state["current_file"] is None:
            st.session_state.state["current_file"] = args.path
            save_state(st.session_state.state)
    if 'should_load_file' not in st.session_state:
        st.session_state.should_load_file = None

    def load_and_display_file(file_path):
        if not file_path:
            return False
        try:
            data = load_metrics(file_path)
            display_metrics(data, file_path)
            st.session_state.state["current_file"] = file_path
            save_state(st.session_state.state)
            return True
        except Exception as e:
            st.error(f"Error loading JSON file from path: {e}")
            return False

    # Handle file changes from previous run
    if st.session_state.should_load_file:
        file_to_load = st.session_state.should_load_file
        st.session_state.should_load_file = None  # Reset the flag
        if load_and_display_file(file_to_load):
            return

    # Get available result files
    result_files = get_available_result_files()
    
    # If we have a current file, try to load and display it
    current_file = st.session_state.state.get("current_file")
    if current_file:
        load_and_display_file(current_file)

    # Add file selector with search at the bottom
    st.markdown("---")
    st.markdown("### Change Result File")
    
    if result_files:
        search_pattern = st.text_input("Search files (wildcards added automatically)", "", key="bottom_search")
        filtered_files = filter_files(result_files, search_pattern)
        if filtered_files:
            selected_file = st.selectbox(
                "Choose a result file",
                filtered_files,
                format_func=lambda x: x.replace('results/interv_effects/', ''),
                key="bottom_select"
            )
            if selected_file and selected_file != current_file:
                if st.button("Load Selected File"):
                    st.session_state.should_load_file = selected_file
                    st.rerun()
        else:
            st.info("No files match your search pattern")
    else:
        st.info("No result files found in results/interv_effects directory")
    
    # Keep the upload option as fallback
    st.markdown("### Or upload a file")
    uploaded_file = st.file_uploader("Choose a JSON file", type=["json"])
    if uploaded_file is not None and (not current_file or not current_file.endswith(uploaded_file.name)):
        st.session_state.should_load_file = uploaded_file
        st.rerun()


if __name__ == "__main__":
    main()
