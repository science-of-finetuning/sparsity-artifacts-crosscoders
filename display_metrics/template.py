import pandas as pd
import plotly.graph_objects as go
import numpy as np
import warnings
from display_metrics.shared import load_metrics, format_setup_name

data = load_metrics({{file_path}})

# Extract data for selected metric and categories
metric_type = {{metric_type}}
categories = {{categories}}
requested_setups = {{selected_setups}}

# Track which setups have data
valid_setups = []
missing_setups = []

# Build dataframe for selected setups across categories
all_data = {}
for category in categories:
    if metric_type not in data.get(category, {}):
        warnings.warn(f"Metric '{metric_type}' not found in category '{category}'")
        continue

    category_data = data[category][metric_type]
    rows = []
    for setup in requested_setups:
        if setup in category_data:
            stats = category_data[setup]
            mean = stats.get("mean", None)
            var = stats.get("var", None)
            count = stats.get("count", None)
            if mean is not None and var is not None and count is not None:
                rows.append({"mean": mean, "variance": var, "n": count})
                if setup not in valid_setups:
                    valid_setups.append(setup)
        elif setup not in missing_setups:
            missing_setups.append(setup)

    if rows:
        all_data[category] = pd.DataFrame(
            rows, index=[s for s in requested_setups if s in valid_setups]
        )

if missing_setups:
    print("Warning: Missing data for setups:", missing_setups)

if all_data:
    # Combine all categories into one dataframe
    df = pd.concat(all_data, axis=1)
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
            if (category, "mean") in df.columns:
                mean = df.loc[setup, (category, "mean")]
                var = df.loc[setup, (category, "variance")]
                n = df.loc[setup, (category, "n")]

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
            ticktext=categories, tickvals=list(range(len(categories))), title="Category"
        ),
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        bargap=0,
    )

    # Show plot
    fig.show()
else:
    print("No valid data available for any setup")
