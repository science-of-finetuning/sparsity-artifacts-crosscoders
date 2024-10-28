from IPython.display import HTML, display
import numpy as np
import ipywidgets as widgets
import urllib.parse


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
        max_activation_examples: dict[int, list[tuple[float, str, list[float]]]],
        tokenizer,
        window_size: int = 50,  # Number of tokens to show before/after max activation
    ):
        """
        Args:
            max_activation_examples: Dictionary mapping feature indices to lists of tuples
                (max_activation_value, text, activation_values_of_each_token)
            tokenizer: HuggingFace tokenizer for the model
            window_size: Number of tokens to show before/after the max activation token
        """
        self.max_activation_examples = max_activation_examples
        self.tokenizer = tokenizer
        self.window_size = window_size
        self._setup_widgets()

    def _setup_widgets(self):
        """Initialize the dashboard widgets"""
        
        # Convert to list for easier validation
        self.available_features = sorted(self.max_activation_examples.keys())
        
        self.feature_selector = widgets.Combobox(
            # Convert numbers to strings and create a tuple of options
            options=tuple(str(f) for f in self.available_features),
            placeholder='Type a feature number...',
            description="Feature:",
            ensure_option=False,  # Allow typing values not in dropdown
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
                    print(f"Feature {feature_idx} not found. Available features: {self.available_features}")
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
        # Update the CSS to blend the hover effect with the red background
        html_parts = ["""
            <style>
                .token { 
                    transition: background-color 0.1s;
                    position: relative;  /* For proper hover effect layering */
                }
                .token:hover { 
                    background-image: linear-gradient(rgba(128, 128, 128, 0.3), rgba(128, 128, 128, 0.3));
                }
                .token-tooltip {
                    position: absolute;
                    background: black;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    pointer-events: none;
                    z-index: 1000;
                    white-space: pre-wrap;
                }
            </style>
            <script>
                function setupTokenTooltips() {
                    // Create tooltip element if it doesn't exist
                    if (!document.querySelector('.token-tooltip')) {
                        const tooltip = document.createElement('div');
                        tooltip.className = 'token-tooltip';
                        tooltip.style.display = 'none';
                        document.body.appendChild(tooltip);
                    }
                    
                    // Add event listeners for all tokens
                    document.querySelectorAll('.token').forEach(token => {
                        token.addEventListener('mousemove', (e) => {
                            const tooltip = document.querySelector('.token-tooltip');
                            tooltip.textContent = token.dataset.tooltip;
                            tooltip.style.display = 'block';
                            tooltip.style.left = e.pageX + 10 + 'px';
                            tooltip.style.top = e.pageY + 10 + 'px';
                        });
                        
                        token.addEventListener('mouseleave', () => {
                            const tooltip = document.querySelector('.token-tooltip');
                            tooltip.style.display = 'none';
                        });
                    });
                }
                setupTokenTooltips();
            </script>
        """]
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
            )

            color = f"rgba(255, 0, 0, {abs(norm_act):.3f})"

            # Create tooltip content with token ID, string, and activation
            tok_id = self.tokenizer.convert_ids_to_tokens(i)
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

            for max_act, tokens, token_acts in examples:
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
                    <p style="margin: 0 0 5px 0;"><b>Max Activation: {max_act:.2f}</b></p>
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
