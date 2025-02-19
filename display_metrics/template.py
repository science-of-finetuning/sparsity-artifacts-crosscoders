import pandas as pd
import plotly.graph_objects as go
import numpy as np
import warnings
from display_metrics.shared import (
    load_metrics,
    format_setup_name,
    build_complete_dataframe,
)

data = load_metrics({{file_path}})

# Extract data for selected metric and categories
metric_type = {{metric_type}}
categories = {{categories}}
requested_setups = {{selected_setups}}

# Build dataframe for selected setups across categories
df = build_complete_dataframe(data)
if df is None:
    print("No valid data available for any setup")
else:
    # Filter for requested metric and categories
    df = df.loc[:, (categories, metric_type)]

    # Find which setups are available in the data
    valid_setups = [s for s in requested_setups if s in df.index]
    missing_setups = [s for s in requested_setups if s not in df.index]

    if missing_setups:
        print("Warning: Missing data for setups:", missing_setups)

    if not valid_setups:
        print("No data available for any of the requested setups")
    else:
        df = df.loc[valid_setups]
        df = df.dropna(axis=1, how="all")

        # Create plot
        fig = go.Figure()
        n_setups = len(valid_setups)
        bar_width = 0.8 / n_setups if n_setups > 0 else 0.8
        offsets = (
            np.linspace(
                -(n_setups - 1) * bar_width / 2,
                (n_setups - 1) * bar_width / 2,
                n_setups,
            )
            if n_setups > 0
            else [0]
        )

        # Add bars for each setup
        for setup_idx, setup in enumerate(valid_setups):
            x_positions = []
            y_values = []
            error_values = []

            for cat_idx, category in enumerate(categories):
                if (category, metric_type, "mean") in df.columns:
                    mean = df.loc[setup, (category, metric_type, "mean")]
                    var = df.loc[setup, (category, metric_type, "variance")]
                    n = df.loc[setup, (category, metric_type, "n")]

                    if not pd.isna(mean) and not pd.isna(var) and not pd.isna(n):
                        x_positions.append(cat_idx + offsets[setup_idx])
                        y_values.append(mean)
                        error_values.append(1.96 * (var / n) ** 0.5)

            if y_values:  # Only add trace if we have data
                setup_name = format_setup_name(setup)
                fig.add_trace(
                    go.Bar(
                        name=setup_name,
                        x=x_positions,
                        y=y_values,
                        width=bar_width,
                        error_y=dict(type="data", array=error_values, visible=True),
                        hovertemplate=f"Setup: {setup_name}<br>Internal name: {setup}<br>Value: %{{y:.3f}}<br>Error: %{{error_y.array:.3f}}<extra></extra>",
                    )
                )

        # Update layout
        fig.update_layout(
            title=metric_type,
            showlegend=True,
            height=400,
            xaxis=dict(
                ticktext=categories,
                tickvals=list(range(len(categories))),
                title="Category",
            ),
            yaxis_title="Value",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            bargap=0,
        )

        # Show plot
        fig.show()
