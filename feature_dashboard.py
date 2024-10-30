from IPython.display import HTML, display
import numpy as np
import ipywidgets as widgets
import urllib.parse
from pathlib import Path


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

            for max_act, tokens, token_acts in list(examples)[:self.max_examples]:
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
