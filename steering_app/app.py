import streamlit as st
import torch as th
import sys
from pathlib import Path
import json
from argparse import ArgumentParser

from nnterp import load_model
from nnterp.nnsight_utils import get_num_layers

# Add the parent directory to the path to import tools
sys.path.append(str(Path(__file__).parent.parent))
from tools.utils import auto_device, get_available_models, patch_tokenizer

# Import the intervention factory
from intervention_factory import create_intervention
from edited_generation import generate_with_interventions

# Constants for state management
CACHE_DIR = Path(__file__).parent / "cache"
INTERVENTIONS_FILE = CACHE_DIR / "interventions.json"
OUTPUTS_FILE = CACHE_DIR / "outputs.json"
INPUT_FILE = CACHE_DIR / "input.json"
ENABLED_INTERVENTIONS_FILE = CACHE_DIR / "enabled_interventions.json"

# Set page config
st.set_page_config(
    page_title="Halfway Interventions App",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS to improve the appearance
st.markdown(
    """
<style>
    .stTextArea textarea {
        font-family: monospace;
        font-size: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_from_json(file_path, default_value):
    """Generic function to load data from a JSON file"""
    if not file_path.exists():
        return default_value

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError):
        # If there's any error reading or parsing the file, return default value
        if file_path.exists():
            try:
                file_path.unlink()  # Delete corrupted file
            except OSError:
                pass
        return default_value


def save_to_json(file_path, data):
    """Generic function to save data to a JSON file"""
    try:
        # Ensure the cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f)
        return True
    except (OSError, TypeError) as e:
        # If we can't save the data, log the error but don't crash
        print(f"Error saving to {file_path}: {e}")
        if file_path.exists():
            try:
                file_path.unlink()  # Delete potentially corrupted file
            except OSError:
                pass
        return False


@st.cache_data
def load_interventions():
    """Load interventions from cache file"""
    return load_from_json(INTERVENTIONS_FILE, [])


def save_interventions(interventions):
    """Save interventions to cache file"""
    success = save_to_json(INTERVENTIONS_FILE, interventions)
    if success:
        # Clear the cache for load_interventions to ensure fresh data next time
        load_interventions.clear()
    return success


@st.cache_data
def load_outputs():
    """Load generated outputs from cache file"""
    return load_from_json(OUTPUTS_FILE, [])


def save_outputs(outputs):
    """Save generated outputs to cache file"""
    success = save_to_json(OUTPUTS_FILE, outputs)
    if success:
        # Clear the cache for load_outputs to ensure fresh data next time
        load_outputs.clear()
    return success


@st.cache_data
def load_input():
    """Load input text from cache file"""
    data = load_from_json(INPUT_FILE, {"text": "Enter your text here..."})
    return data.get("text", "Enter your text here...")


def save_input(text):
    """Save input text to cache file"""
    success = save_to_json(INPUT_FILE, {"text": text})
    if success:
        # Clear the cache for load_input to ensure fresh data next time
        load_input.clear()
    return success


@st.cache_data
def load_enabled_interventions():
    """Load enabled interventions from cache file"""
    return load_from_json(ENABLED_INTERVENTIONS_FILE, [])


def save_enabled_interventions(enabled_interventions):
    """Save enabled interventions to cache file"""
    success = save_to_json(ENABLED_INTERVENTIONS_FILE, enabled_interventions)
    if success:
        # Clear the cache for load_enabled_interventions to ensure fresh data next time
        load_enabled_interventions.clear()
    return success


# Initialize session state
if "interventions" not in st.session_state:
    st.session_state.interventions = load_interventions()

if "generated_outputs" not in st.session_state:
    st.session_state.generated_outputs = load_outputs()

if "input_text" not in st.session_state:
    st.session_state.input_text = load_input()

if "enabled_interventions" not in st.session_state:
    st.session_state.enabled_interventions = load_enabled_interventions()


@st.cache_data
def create_cached_intervention(intervention_type, params, crosscoder_dict, cc_device):
    """Cache intervention creation to avoid recreating the same interventions repeatedly"""
    return create_intervention(intervention_type, params, crosscoder_dict, cc_device)


def intervention_ui(sidebar):
    with sidebar:
        st.header("Add Intervention")

        # Dropdown to select intervention type
        intervention_type = st.selectbox(
            "Select Intervention Type",
            [
                "IdentityPreprocessFn",
                "SwitchPreprocessFn",
                "TestMaskPreprocessFn",
                "PatchProjectionFromDiff",
                "SteeringVector",
                "SteerWithCrosscoderLatent",
                "CrossCoderReconstruction",
                "CrossCoderSteeringLatent",
                "CrossCoderAdditiveSteering",
                "CrossCoderOutProjection",
                "SAEAdditiveSteering",
                "PatchCtrl",
                "PatchKFirstPredictions",
                "PatchKFirstAndCtrl",
            ],
        )

        # Add a field for intervention name
        intervention_name = st.text_input("Intervention Name", f"{intervention_type}")

        # Parameters section - will change based on selected intervention
        st.subheader("Parameters")

        # Common parameters
        continue_with = st.selectbox("continue_with", ["base", "chat"])

        # Specific parameters based on intervention type
        intervention_params = {}

        if intervention_type == "IdentityPreprocessFn":
            intervention_params = {"continue_with": continue_with}

        elif intervention_type == "SwitchPreprocessFn":
            intervention_params = {"continue_with": continue_with}

        elif intervention_type == "TestMaskPreprocessFn":
            intervention_params = {"continue_with": continue_with}

        elif intervention_type in ["PatchProjectionFromDiff", "SteeringVector"]:
            patch_target = st.selectbox("patch_target", ["base", "chat"])

            # For demonstration, use a dummy tensor
            # In a real app, you'd need a way to input or select tensors
            dummy_tensor = th.randn(768)

            if intervention_type == "PatchProjectionFromDiff":
                scale = st.slider("scale_steering_latent", 0.0, 5.0, 1.0, 0.1)
                intervention_params = {
                    "continue_with": continue_with,
                    "patch_target": patch_target,
                    "vectors_to_project": dummy_tensor,
                    "scale_steering_latent": scale,
                }
            else:  # SteeringVector
                intervention_params = {
                    "continue_with": continue_with,
                    "patch_target": patch_target,
                    "vector": dummy_tensor,
                }

        elif intervention_type in [
            "CrossCoderReconstruction",
            "CrossCoderSteeringLatent",
            "CrossCoderAdditiveSteering",
            "CrossCoderOutProjection",
            "SteerWithCrosscoderLatent",
        ]:
            # Get available CrossCoder models using the cached function
            crosscoder_models = get_available_cached_models()

            # Add a dropdown to select a CrossCoder model
            selected_crosscoder = st.selectbox(
                "Select CrossCoder Model",
                crosscoder_models,
                help="Select a CrossCoder model from Hugging Face",
            )

            if intervention_type == "CrossCoderReconstruction":
                reconstruct_with = st.selectbox("reconstruct_with", ["base", "chat"])
                intervention_params = {
                    "crosscoder": selected_crosscoder,
                    "reconstruct_with": reconstruct_with,
                    "continue_with": continue_with,
                }

            elif intervention_type == "SteerWithCrosscoderLatent":
                steer_activations_of = st.selectbox(
                    "steer_activations_of", ["base", "chat"]
                )
                steer_with_latents_from = st.selectbox(
                    "steer_with_latents_from", ["base", "chat"]
                )

                # For latents_to_steer, we'll use a text input that can be parsed as a list
                latents_str = st.text_input(
                    "latents_to_steer (comma-separated indices)", "0,1,2"
                )
                latents_to_steer = (
                    [int(x.strip()) for x in latents_str.split(",")]
                    if latents_str
                    else None
                )

                scale = st.slider("scale_steering_latent", 0.0, 5.0, 1.0, 0.1)

                intervention_params = {
                    "crosscoder": selected_crosscoder,
                    "steer_activations_of": steer_activations_of,
                    "steer_with_latents_from": steer_with_latents_from,
                    "latents_to_steer": latents_to_steer,
                    "continue_with": continue_with,
                    "scale_steering_latent": scale,
                }

            elif intervention_type in [
                "CrossCoderSteeringLatent",
                "CrossCoderAdditiveSteering",
                "CrossCoderOutProjection",
            ]:
                steer_activations_of = st.selectbox(
                    "steer_activations_of", ["base", "chat"]
                )
                steer_with_latents_from = st.selectbox(
                    "steer_with_latents_from", ["base", "chat"]
                )

                # For latents_to_steer, we'll use a text input that can be parsed as a list
                latents_str = st.text_input(
                    "latents_to_steer (comma-separated indices)", "0,1,2"
                )
                latents_to_steer = (
                    [int(x.strip()) for x in latents_str.split(",")]
                    if latents_str
                    else None
                )

                scale = st.slider("scale_steering_latent", 0.0, 5.0, 1.0, 0.1)

                intervention_params = {
                    "crosscoder": selected_crosscoder,
                    "steer_activations_of": steer_activations_of,
                    "steer_with_latents_from": steer_with_latents_from,
                    "latents_to_steer": latents_to_steer,
                    "continue_with": continue_with,
                    "scale_steering_latent": scale,
                }

                if intervention_type == "CrossCoderSteeringLatent":
                    monitored_latents_str = st.text_input(
                        "monitored_latents (comma-separated indices)", ""
                    )
                    monitored_latents = (
                        [int(x.strip()) for x in monitored_latents_str.split(",")]
                        if monitored_latents_str
                        else None
                    )

                    filter_threshold = st.number_input("filter_threshold", value=None)
                    ignore_encoder = st.checkbox("ignore_encoder", value=False)

                    intervention_params.update(
                        {
                            "monitored_latents": monitored_latents,
                            "filter_treshold": (
                                filter_threshold if filter_threshold else None
                            ),
                            "ignore_encoder": ignore_encoder,
                        }
                    )

        elif intervention_type == "SAEAdditiveSteering":
            steer_activations_of = st.selectbox(
                "steer_activations_of", ["base", "chat"]
            )
            steer_with_latents_from = st.selectbox(
                "steer_with_latents_from", ["base", "chat"]
            )

            latents_str = st.text_input(
                "latents_to_steer (comma-separated indices)", "0,1,2"
            )
            latents_to_steer = (
                [int(x.strip()) for x in latents_str.split(",")]
                if latents_str
                else None
            )

            scale = st.slider("scale_steering_latent", 0.0, 5.0, 1.0, 0.1)

            intervention_params = {
                "sae": "dummy_sae",
                "steer_activations_of": steer_activations_of,
                "steer_with_latents_from": steer_with_latents_from,
                "latents_to_steer": latents_to_steer,
                "continue_with": continue_with,
                "scale_steering_latent": scale,
            }

        elif intervention_type in [
            "PatchCtrl",
            "PatchKFirstPredictions",
            "PatchKFirstAndCtrl",
        ]:
            patch_target = st.selectbox("patch_target", ["base", "chat"])

            intervention_params = {
                "continue_with": continue_with,
                "patch_target": patch_target,
                "activation_processor": None,  # For simplicity
            }

        # Add intervention button
        if st.button("Add Intervention"):
            # Create a description of the intervention for display
            intervention_desc = f"{intervention_params}"

            # Add to session state
            st.session_state.interventions.append(
                {
                    "type": intervention_type,
                    "name": intervention_name,
                    "params": intervention_params,
                    "description": intervention_desc,
                }
            )

            # Enable the newly added intervention (its index is len-1)
            new_intervention_index = len(st.session_state.interventions) - 1
            if new_intervention_index not in st.session_state.enabled_interventions:
                st.session_state.enabled_interventions.append(new_intervention_index)

            # Save interventions to file
            save_interventions(st.session_state.interventions)
            save_enabled_interventions(st.session_state.enabled_interventions)

            st.success(f"Added {intervention_name} to the list!")


@st.cache_resource
def load_cached_model(
    model_name, device_map, dtype=th.bfloat16, attn_implementation="eager"
):
    """Load and cache a model to avoid reloading it on each rerun"""
    print(f"Loading model {model_name} (this will only happen once per model)")
    model = load_model(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return model


@st.cache_data
def get_available_cached_models():
    """Cache the list of available models to avoid regenerating it on each rerun"""
    return get_available_models()


@st.cache_data
def generate_cached_text(
    _base_model,
    _chat_model,
    layer,
    cc_device,
    _active_interventions,
    input_text,
    max_new_tokens,
    only_patch_context,
):
    """Cache text generation results based on input parameters"""
    # Note: we don't directly hash the model and intervention objects
    # Instead, we rely on the input parameters to determine if the cache should be invalidated

    # Extract intervention types and parameters to use for caching
    intervention_specs = []
    for intervention in _active_interventions:
        # Get a hashable representation of the intervention
        intervention_type = type(intervention).__name__
        # We can't include the actual parameters since some might not be hashable
        # So we just include the intervention type in the cache key
        intervention_specs.append(intervention_type)

    # Actually perform the generation
    return generate_with_interventions(
        _base_model,
        _chat_model,
        layer,
        cc_device,
        _active_interventions,
        input_text,
        max_new_tokens=max_new_tokens,
        only_patch_context=only_patch_context,
    )


def main():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    parser = ArgumentParser()
    parser.add_argument("--base-device", type=str, default="auto")
    parser.add_argument("--chat-device", type=str, default="auto")
    parser.add_argument("--cc-device", type=auto_device, default="auto")
    args = parser.parse_args()
    base_device = args.base_device
    chat_device = args.chat_device
    cc_device = args.cc_device

    if cc_device == "auto":
        cc_device = "cuda" if th.cuda.is_available() else "cpu"

    st.title("Halfway Interventions App")
    crosscoder_dict = {}
    current_base_model = "google/gemma-2-2b"
    current_chat_model = "google/gemma-2-2b-it"

    # Use cached model loading
    base_model = load_cached_model(
        current_base_model,
        device_map=base_device,
        dtype=th.bfloat16,
        attn_implementation="eager",
    )
    chat_model = load_cached_model(
        current_chat_model,
        device_map=chat_device,
        dtype=th.bfloat16,
        attn_implementation="eager",
    )
    patch_tokenizer(chat_model.tokenizer, current_chat_model)

    # Create a sidebar for adding interventions
    with st.sidebar:
        st.header("Models")
        base_model_name = st.text_input("Base Model", value="google/gemma-2-2b")
        chat_model_name = st.text_input("Chat Model", value="google/gemma-2-2b-it")
        # Layer selection for interventions
        layer = st.number_input(
            "Layer Number",
            min_value=0,
            value=13,
            step=1,
            help="Select the layer number for interventions",
        )
        intervention_ui(st.sidebar)

    # Main content area - use two columns with different widths
    main_col1, main_col2 = st.columns([1, 2])

    with main_col1:
        st.header("Interventions List")

        if not st.session_state.interventions:
            st.info(
                "No custom interventions added yet. Use the sidebar to add interventions."
            )
        else:
            for i, intervention in enumerate(st.session_state.interventions):
                # Format the title consistently with the same format used in outputs
                is_enabled = i in st.session_state.enabled_interventions
                status_label = "🟢" if is_enabled else "🟥"
                formatted_title = f"{status_label} ({i+1}) {intervention['name']}"

                with st.expander(formatted_title):
                    st.write(f"Type: {intervention['type']}")
                    st.write(f"Parameters: {intervention['description']}")

                    # Create a row of buttons with equal width
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Enable/Disable button with the same style as shown/hidden
                        button_label = "Enabled" if is_enabled else "Disabled"
                        if st.button(
                            button_label,
                            key=f"toggle_enable_{i}",
                            use_container_width=True,
                            type="secondary" if is_enabled else "primary",
                            help=f"Click to {'disable' if is_enabled else 'enable'} this intervention",
                        ):
                            if is_enabled:
                                st.session_state.enabled_interventions.remove(i)
                            else:
                                st.session_state.enabled_interventions.append(i)
                            save_enabled_interventions(
                                st.session_state.enabled_interventions
                            )
                            st.rerun()

                    with col2:
                        if st.button(
                            "Remove", key=f"remove_{i}", use_container_width=True
                        ):
                            st.session_state.interventions.pop(i)

                            # Update enabled interventions list when removing an intervention
                            updated_enabled = []
                            for idx in st.session_state.enabled_interventions:
                                if idx < i:
                                    updated_enabled.append(idx)
                                elif idx > i:
                                    updated_enabled.append(idx - 1)
                            st.session_state.enabled_interventions = updated_enabled

                            # Save interventions to file after removal
                            save_interventions(st.session_state.interventions)
                            save_enabled_interventions(
                                st.session_state.enabled_interventions
                            )
                            st.rerun()

        if st.session_state.interventions:
            if st.button("Clear All Interventions"):
                st.session_state.interventions = []
                # Save empty interventions list to file
                save_interventions(st.session_state.interventions)
                # Also clear outputs
                st.session_state.generated_outputs = []
                save_outputs([])
                st.rerun()

    with main_col2:
        st.header("Text Generation")

        input_text = st.text_area("Input Text", st.session_state.input_text, height=150)
        # Update session state when input changes
        if input_text != st.session_state.input_text:
            st.session_state.input_text = input_text
            save_input(input_text)

        # Add max_new_tokens parameter
        max_new_tokens = st.slider(
            "Max New Tokens",
            min_value=1,
            max_value=2048,
            value=512,
            step=64,
            help="Maximum number of tokens to generate",
        )

        # Create a row with generate button and checkbox
        col1, col2 = st.columns([1, 1])
        with col1:
            generate_button = st.button("Generate", key="generate_button")
        with col2:
            only_patch_context = st.checkbox(
                "Only intervene on context",
                value=False,
                help="When checked, interventions will only be applied to the context tokens, not during generation",
            )

    # Full width section for outputs
    if generate_button or st.session_state.generated_outputs:
        st.markdown("## Generated Outputs")

        # Always include the default Identity intervention (chat model forward pass)
        default_intervention = {
            "type": "IdentityPreprocessFn",
            "name": "Default Chat Model",
            "params": {"continue_with": "chat"},
            "description": "{'continue_with': 'chat'}",
        }

        # Generate outputs if button was clicked
        if generate_button:
            st.session_state.generated_outputs = []
            active_interventions = [
                create_cached_intervention(
                    default_intervention["type"],
                    default_intervention["params"],
                    crosscoder_dict,
                    cc_device,
                )
            ]
            active_interventions_metadata = [
                {"idx": 0, "intervention": default_intervention}
            ]
            for intervention_idx in st.session_state.enabled_interventions:
                try:
                    intervention_data = st.session_state.interventions[intervention_idx]
                    # Create the actual intervention object
                    intervention_obj = create_cached_intervention(
                        intervention_data["type"],
                        intervention_data["params"],
                        crosscoder_dict,
                        cc_device,
                    )
                    active_interventions.append(intervention_obj)
                    active_interventions_metadata.append(
                        {"idx": intervention_idx + 1, "intervention": intervention_data}
                    )
                except Exception as e:
                    st.session_state.generated_outputs.append(
                        {
                            "intervention": intervention_data,
                            "error": str(e)
                            + "\nIntervention: "
                            + str(intervention_data),
                            "idx": intervention_idx,
                        }
                    )
                    raise

            # Load models if changed using the cached loading function
            if base_model_name != current_base_model:
                print(f"Loading base model {base_model_name}")
                base_model = load_cached_model(
                    base_model_name,
                    device_map=base_device,
                    dtype=th.bfloat16,
                    attn_implementation="eager",
                )
                current_base_model = base_model_name

            if chat_model_name != current_chat_model:
                print(f"Loading chat model {chat_model_name}")
                chat_model = load_cached_model(
                    chat_model_name,
                    device_map=chat_device,
                    dtype=th.bfloat16,
                    attn_implementation="eager",
                )
                patch_tokenizer(chat_model.tokenizer, current_chat_model)
                current_chat_model = chat_model_name

            # Use cached text generation
            generated_text = generate_cached_text(
                base_model,
                chat_model,
                layer,
                cc_device,
                active_interventions,
                input_text,
                max_new_tokens=max_new_tokens,
                only_patch_context=only_patch_context,
            )

            for int_text, int_data in zip(
                generated_text, active_interventions_metadata
            ):
                st.session_state.generated_outputs.append(
                    {
                        "idx": int_data["idx"],
                        "intervention": int_data["intervention"],
                        "text": int_text,
                    }
                )
            # Save outputs to file
            save_outputs(st.session_state.generated_outputs)

        # Use columns for the outputs
        output_cols = st.columns(2)

        for i, output in enumerate(st.session_state.generated_outputs):
            # Alternate between columns
            col_idx = i % 2
            intervention_data = output["intervention"]
            intervention_idx = output.get("idx", i)

            # Format the title consistently
            formatted_title = f"## ({intervention_idx}) {intervention_data['name']}"

            with output_cols[col_idx]:
                with st.expander(formatted_title, expanded=True):
                    if "error" in output:
                        st.error(
                            f"Error generating with {intervention_data['type']}: {output['error']}"
                        )
                    else:
                        st.markdown(
                            formatted_title
                            + "\n"
                            + output["text"].replace("\n", "<br>"),
                            unsafe_allow_html=True,
                        )

                        # Add a download button for the generated text
                        st.download_button(
                            label="Download this output",
                            data=output["text"],
                            file_name=f"{formatted_title.replace(' ', '_')}.md",
                            mime="text/markdown",
                            key=f"download_{i}",
                        )


if __name__ == "__main__":
    main()
