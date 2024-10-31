from IPython.display import HTML, display
import numpy as np
import ipywidgets as widgets
import urllib.parse
from pathlib import Path
from nnsight import LanguageModel
from functools import cache
import torch as th
import ast

import traceback
from nnterp.nnsight_utils import get_layer, get_layer_output
import einops


def parse_list_str(s: str) -> list[int]:
    if not s.startswith("["):
        s = "[" + s
    if not s.endswith("]"):
        s = s + "]"
    return ast.literal_eval(s)


class FeatureCentricDashboard:
    """
    This Dashboard is composed of a feature selector and a feature viewer.
    The feature selector allows you to select a feature and view the max activating examples for that feature.
    The feature viewer displays the max activating examples for the selected feature. Text is highlighted with a gradient based on the activation value of each token.
    An hover text showing the activation value, token id is also shown. When the mouse passes over a token, the token is highlighted in light grey.
    By default, the text sample is not displayed entirely, but only a few tokens before and after the highest activating token. If the user clicks on the text sample, the entire text sample is displayed.
    """

    def __init__(
        self,
        max_activation_examples: dict[int, list[tuple[float, list[str], list[float]]]],
        tokenizer,
        window_size: int = 50,
        max_examples: int = 30,
    ):
        """
        Args:
            max_activation_examples: Dictionary mapping feature indices to lists of tuples
                (max_activation_value, list of tokens, list of activation values)
            tokenizer: HuggingFace tokenizer for the model
            window_size: Number of tokens to show before/after the max activation token
        """
        self.max_activation_examples = max_activation_examples
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_examples = max_examples
        # Load templates at initialization
        template_dir = Path(__file__).parent / "templates"
        with open(template_dir / "styles.css", "r") as f:
            self.styles = f.read()
        with open(template_dir / "tooltips.js", "r") as f:
            self.scripts = f.read()
        with open(template_dir / "base.html", "r") as f:
            self.base_template = f.read()
        self._setup_widgets()

    def _setup_widgets(self):
        """Initialize the dashboard widgets"""

        # Convert to list for easier validation
        self.available_features = sorted(self.max_activation_examples.keys())

        self.feature_selector = widgets.Text(
            placeholder="Type a feature number...",
            description="Feature:",
            continuous_update=False,  # Only trigger on Enter/loss of focus
            style={"description_width": "initial"},
        )

        self.examples_output = widgets.Output()
        self.feature_selector.observe(self._handle_feature_selection, names="value")

    def _handle_feature_selection(self, change):
        """Handle feature selection, including validation of typed input"""
        try:
            # Try to convert input to integer
            feature_idx = int(change["new"])

            # Validate feature exists
            if feature_idx in self.max_activation_examples:
                self._update_examples({"new": feature_idx})
            else:
                with self.examples_output:
                    self.examples_output.clear_output()
                    print(
                        f"Feature {feature_idx} not found. Available features: {self.available_features}"
                    )
        except ValueError:
            # Handle invalid input
            with self.examples_output:
                self.examples_output.clear_output()
                print("Please enter a valid feature number")

    def _create_html_highlight(
        self,
        tokens: list[str],
        activations: list[float],
        max_idx: int,
        show_full: bool = False,
    ) -> str:
        html_parts = [
            f"<style>{self.styles}</style>",
            f"<script>{self.scripts}</script>",
        ]
        # Determine window bounds
        if show_full:
            start_idx = 0
            end_idx = len(tokens)
        else:
            start_idx = max(0, max_idx - self.window_size)
            end_idx = min(len(tokens), max_idx + self.window_size + 1)

        # Normalize activations for color intensity
        act_array = np.array(activations)
        max_act = np.max(np.abs(act_array))
        norm_acts = act_array / max_act if max_act > 0 else act_array

        # Create HTML spans with activation values
        for i in range(start_idx, end_idx):
            act = activations[i]
            norm_act = norm_acts[i]
            # Double escape: First for HTML, then for JavaScript string
            token = (
                tokens[i]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("'", "&apos;")
                .replace('"', "&quot;")
                .replace("\n", "\\n\n")
            )

            color = f"rgba(255, 0, 0, {abs(norm_act):.3f})"

            # Create tooltip content with token ID, string, and activation
            tok_id = self.tokenizer.convert_tokens_to_ids(tokens[i])
            tooltip_content = f"Token {tok_id}: '{token}'\nActivation: {act:.3f}"
            tooltip_content = tooltip_content.replace('"', "&quot;")

            html_parts.append(
                f'<span class="token" style="background-color: {color};" '
                f'data-tooltip="{tooltip_content}">{token}</span>'
            )

        if end_idx < len(tokens):
            html_parts.append('<span style="color: gray;">...</span>')

        return "".join(html_parts)

    def _update_examples(self, change):
        """Update the examples display when a new feature is selected"""

        feature_idx = change["new"]
        examples = self.max_activation_examples[feature_idx]

        with self.examples_output:
            self.examples_output.clear_output()

            for max_act, tokens, token_acts in list(examples)[: self.max_examples]:
                # Find max activation index
                max_idx = np.argmax(token_acts)

                # Create both collapsed and full versions
                collapsed_html = self._create_html_highlight(
                    tokens, token_acts, max_idx, False
                )
                full_html = self._create_html_highlight(
                    tokens, token_acts, max_idx, True
                )

                # Escape full_html to prevent rendering issues
                # full_html = full_html.replace("<", "&lt;").replace(">", "&gt;")
                # print(f"full_html: {full_html}\nPartial: {collapsed_html}")
                display(
                    HTML(
                        f"""
                <div style="margin: 10px 0; padding: 10px; border: 1px solid #ccc;">
                    <p style="margin: 0 0 2px 0;"><b>Max Activation: {max_act:.2f}</b></p>
                    <div class="text-sample" style="white-space: pre-wrap;" 
                         onclick="this.innerHTML=decodeURIComponent(this.dataset.fullText); setupTokenTooltips();" 
                         data-full-text="{urllib.parse.quote(full_html)}">
                        {collapsed_html}
                    </div>
                    <p style="color: gray; font-size: 0.8em; margin: 5px 0 0 0;">Click to expand</p>
                </div>
                """
                    )
                )

    def display(self):
        """Display the dashboard"""

        dashboard = widgets.VBox([self.feature_selector, self.examples_output])
        display(dashboard)

    def export_to_html(self, output_path: str, features_to_export: list[int]):
        """
        Export the dashboard data to a static HTML file.
        Creates a single self-contained HTML file with embedded CSS and JavaScript.
        """
        # Generate content
        content_parts = []

        for feature_idx in features_to_export:
            examples = self.max_activation_examples[feature_idx]
            content_parts.append(
                f'<div class="feature-section"><h2>Feature {feature_idx}</h2>'
            )

            for max_act, tokens, token_acts in examples:
                max_idx = np.argmax(token_acts)
                full_html = self._create_html_highlight(
                    tokens, token_acts, max_idx, True
                )

                content_parts.append(
                    f"""
                    <div style="margin: 10px 0; padding: 10px; border: 1px solid #ccc;">
                        <p style="margin: 0 0 2px 0;"><b>Max Activation: {max_act:.2f}</b></p>
                        <div class="text-sample" style="white-space: pre-wrap;">
                            {full_html}
                        </div>
                    </div>
                """
                )

            content_parts.append("</div>")

        # Replace placeholders in base template
        html_content = (
            self.base_template.replace("{{content}}", "\n".join(content_parts))
            .replace("{{styles}}", self.styles)
            .replace("{{scripts}}", self.scripts)
        )

        # Create output directory and write file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


class OnlineFeatureCentricDashboard:
    """
    This Dashboard allows real-time analysis of text for specific features.
    Users can input text, select a feature, and see the activation patterns
    highlighted directly in the text with the same visualization style as
    the FeatureCentricDashboard.
    """

    def __init__(
        self,
        base_model: LanguageModel,
        instruct_model: LanguageModel,
        crosscoder,
        collect_layer: int,
        window_size: int = 50,
    ):
        """
        Args:
            model: nnsight LanguageModel
            window_size: Number of tokens to show before/after the max activation token
        """
        self.tokenizer = instruct_model.tokenizer
        self.instruct_model = instruct_model
        self.base_model = base_model
        self.crosscoder = crosscoder
        self.layer = collect_layer
        self.window_size = window_size

        # Load templates
        template_dir = Path(__file__).parent / "templates"
        with open(template_dir / "styles.css", "r") as f:
            self.styles = f.read()
        with open(template_dir / "tooltips.js", "r") as f:
            self.scripts = f.read()

        self._setup_widgets()

    def _setup_widgets(self):
        """Initialize the dashboard widgets"""
        self.text_input = widgets.Textarea(
            placeholder="Enter text to analyze...",
            description="Text:",
            layout=widgets.Layout(
                width="800px", height="100px", font_family="sans-serif"
            ),
            style={"description_width": "initial"},
        )

        # Widget for features to compute
        self.feature_input = widgets.Text(
            placeholder="Enter features to compute [1,2,3]",
            description="Features to compute:",
            continuous_update=False,
            style={"description_width": "initial"},
        )

        # New widgets for display control
        self.highlight_feature = widgets.Text(
            placeholder="Enter feature to highlight in red",
            description="Highlight feature:",
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.tooltip_features = widgets.Text(
            placeholder="Enter features to show in tooltip [1,2,3]",
            description="Tooltip features:",
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.analyze_button = widgets.Button(
            description="Analyze",
            button_style="primary",
        )

        self.output_area = widgets.Output()
        self.analyze_button.on_click(self._handle_analysis)

    @cache
    @th.no_grad
    def get_feature_activation(
        self, text: str, feature_indicies: tuple[int, ...]
    ) -> th.Tensor:
        """Get the activation values for a given feature"""
        with self.instruct_model.trace(text):
            instruct_activations = get_layer_output(self.instruct_model, self.layer)[
                0
            ].save()
            get_layer(self.instruct_model, self.layer).output.stop()
        with self.base_model.trace(text):
            base_activations = get_layer_output(self.base_model, self.layer)[0].save()
            get_layer(self.base_model, self.layer).output.stop()
        print(base_activations.shape)
        cc_input = th.stack(
            [base_activations, instruct_activations], dim=1
        ).float()  # seq, 2, d
        features_acts = self.crosscoder.get_activations(
            cc_input, select_features=feature_indicies
        )  # seq, f
        return features_acts

    def _create_html_highlight(
        self,
        tokens: list[str],
        activations: th.Tensor,
        all_feature_indicies: list[int],
        highlight_feature_idx: int,
        tooltip_features: list[int],
    ) -> str:
        """Create HTML with highlighted tokens based on activation values"""
        html_parts = [
            f"<style>{self.styles}</style>",
            f"<script>{self.scripts}</script>",
        ]

        # Find highlight feature index in the activation tensor
        highlight_idx = all_feature_indicies.index(highlight_feature_idx)
        # Normalize activations for color intensity (only for highlight feature)
        highlight_acts = activations[:, highlight_idx]
        max_highlight = highlight_acts.max()
        norm_acts = highlight_acts / (max_highlight + 1e-6)

        # Create HTML spans with activation values
        for i in range(len(tokens)):
            token = (
                tokens[i]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("'", "&apos;")
                .replace('"', "&quot;")
                .replace("\n", "\\n\n")
            )

            color = f"rgba(255, 0, 0, {norm_acts[i].item():.3f})"

            # Create tooltip content only for requested features
            tok_id = self.tokenizer.convert_tokens_to_ids(tokens[i])
            tooltip_lines = [f"Token {tok_id}: '{token}'"]
            for feat in tooltip_features:
                feat_idx = all_feature_indicies.index(feat)
                act_value = activations[i, feat_idx].item()
                tooltip_lines.append(f"Feature {feat}: {act_value:.3f}")

            tooltip_content = "\n".join(tooltip_lines).replace('"', "&quot;")

            html_parts.append(
                f'<span class="token" style="background-color: {color};" '
                f'data-tooltip="{tooltip_content}">{token}</span>'
            )

        return "".join(html_parts)

    def _handle_analysis(self, _):
        """Handle the analysis button click"""
        try:
            # Parse feature indices for computation
            f_idx_str = self.feature_input.value.strip()
            feature_indicies = parse_list_str(f_idx_str)
            if len(feature_indicies) == 0:
                raise ValueError("No feature indicies provided")
            # Parse display control features
            highlight_feature = int(self.highlight_feature.value.strip())
            tooltip_features = parse_list_str(self.tooltip_features.value.strip())
            text = self.text_input.value
            tokens = self.tokenizer.tokenize(text, add_special_tokens=True)
            all_features = (
                set(feature_indicies) | set(tooltip_features) | {highlight_feature}
            )
            activations = self.get_feature_activation(text, tuple(all_features))

            with self.output_area:
                self.output_area.clear_output()

                # Display max activations for tooltip features
                max_acts_html = []
                for feat in tooltip_features:
                    if feat in feature_indicies:
                        feat_idx = feature_indicies.index(feat)
                        max_act = activations[:, feat_idx].max().item()
                        max_acts_html.append(f"Feature {feat} max: {max_act:.3f}")

                max_acts_display = (
                    "<div style='margin-bottom: 10px'><b>"
                    + " | ".join(max_acts_html)
                    + "</b></div>"
                )

                # Create and display the highlighted text
                html_content = self._create_html_highlight(
                    tokens,
                    activations,
                    feature_indicies,
                    highlight_feature,
                    tooltip_features,
                )
                display(
                    HTML(
                        f"""
                    <div style="margin: 10px 0; padding: 10px; border: 1px solid #ccc;">
                        {max_acts_display}
                        <div class="text-sample" style="white-space: pre-wrap;">
                            {html_content}
                        </div>
                    </div>
                    """
                    )
                )

        except ValueError:
            with self.output_area:
                self.output_area.clear_output()
                print("Please enter a valid feature number")
        except Exception as e:
            with self.output_area:
                self.output_area.clear_output()
                traceback.print_exc()

    def display(self):
        """Display the dashboard"""
        dashboard = widgets.VBox(
            [
                widgets.HBox([self.text_input]),
                widgets.HBox(
                    [
                        self.feature_input,
                        self.highlight_feature,
                        self.tooltip_features,
                        self.analyze_button,
                    ]
                ),
                self.output_area,
            ]
        )
        display(dashboard)
