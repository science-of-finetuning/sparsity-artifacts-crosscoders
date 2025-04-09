# Halfway Interventions Streamlit App

This Streamlit application allows you to experiment with different halfway interventions for language models. You can create a list of interventions, configure their parameters, and generate text with each intervention applied to the model's activations.

## Features

- Create and configure various types of halfway interventions
- Add multiple interventions to a list and toggle them on/off
- Generate text using each enabled intervention
- Compare outputs from different interventions side-by-side
- Cache generated outputs and interventions between sessions
- Control whether interventions apply only to context or during generation
- Download generated outputs as markdown files

## Installation

1. Make sure you have the required dependencies installed:
   ```
   pip install streamlit torch nnterp
   ```

2. Clone this repository and navigate to the project directory.

## Usage

1. Run the Streamlit app:
   ```
   cd steering_app
   streamlit run app.py [--base-device DEVICE] [--chat-device DEVICE] [--cc-device DEVICE]
   ```

2. The app will open in your browser.

3. Use the sidebar to:
   - Configure base and chat models
   - Select the layer number for interventions
   - Choose an intervention type from the dropdown
   - Set the parameters for the selected intervention
   - Click "Add Intervention" to add it to the list

4. Enter text in the text area on the right side of the screen.

5. Set the maximum number of tokens to generate.

6. Optionally check "Only intervene on context" to apply interventions only to context tokens.

7. Click "Generate" to produce text with each enabled intervention.

## Model Loading and Caching

The app uses efficient caching mechanisms to:
- Load models only once and reuse them across sessions
- Cache generated text based on input parameters
- Store interventions and their configurations between sessions
- Preserve the enabled/disabled state of interventions

## Intervention Types

The app supports the following intervention types:

- **IdentityPreprocessFn**: Passes through activations unchanged
- **SwitchPreprocessFn**: Switches between base and chat model activations
- **TestMaskPreprocessFn**: Applies a test mask to activations
- **PatchProjectionFromDiff**: Projects activations based on differences
- **SteeringVector**: Steers activations using a vector
- **SteerWithCrosscoderLatent**: Steers activations using CrossCoder latent representations
- **CrossCoderReconstruction**: Reconstructs activations using a CrossCoder
- **CrossCoderSteeringLatent**: Steers activations using CrossCoder latents with monitoring
- **CrossCoderAdditiveSteering**: Applies additive steering with CrossCoder
- **CrossCoderOutProjection**: Projects out directions from activations
- **SAEAdditiveSteering**: Applies additive steering with Sparse Autoencoders
- **PatchCtrl**: Patches control tokens
- **PatchKFirstPredictions**: Patches first K predictions
- **PatchKFirstAndCtrl**: Patches both control tokens and first K predictions

## Managing Interventions

- **Add**: Configure and add interventions using the sidebar
- **Enable/Disable**: Toggle interventions on/off without removing them
- **Remove**: Delete individual interventions from the list
- **Clear All**: Remove all interventions at once

## Output Comparison

Generated outputs are displayed side-by-side for easy comparison, with:
- Clear labeling of which intervention produced each output
- Expandable sections for each output
- Download buttons for saving individual outputs

## Notes

- The app automatically selects appropriate devices for model loading
- CrossCoder models can be selected from available models on Hugging Face
- Interventions are cached to improve performance between runs
- The app maintains state between sessions using JSON files

## Command Line Arguments

- `--base-device`: Device to load the base model on (default: auto)
- `--chat-device`: Device to load the chat model on (default: auto)
- `--cc-device`: Device to use for CrossCoder operations (default: auto) 