"""
Distribution (KDE) chart for xsam charts module.

Up to 3 KDE plots, no fill, optional quantile lines for each.
"""

from attrs import define, field
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from .colors import COLORS


@define(slots=True, frozen=True)
class DistributionChartConfig:
    columns: list[str]
    title: str = ""
    labels: dict[str, str] = field(factory=dict)
    quantiles: dict[str, list[float]] = field(factory=dict)  # column -> list of quantiles
    kde_points: int = 200
    xaxis_title: str | None = None
    yaxis_title: str | None = None
    line_names: list[str] | None = None


def plot_distribution_chart(
    df: pd.DataFrame,
    config: DistributionChartConfig,
) -> go.Figure:
    """
    Plot up to 3 KDE curves, optionally with quantile lines for each.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (DistributionChartConfig): Configuration for the chart.

    Returns:
        go.Figure: Plotly Figure object representing the KDE chart.
    """
    fig = go.Figure()
    for i, col in enumerate(config.columns[:3]):
        data = df[col].dropna()
        # Only plot KDE if there are at least 2 data points
        if len(data) < 2:
            continue
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min(), data.max(), config.kde_points)
        y_grid = kde(x_grid)
        name = config.line_names[i] if config.line_names and i < len(config.line_names) else col
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name=name,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                fill=None,
            )
        )
        # Quantile lines
        if col in config.quantiles:
            for q in config.quantiles[col]:
                q_val = data.quantile(q)
                fig.add_trace(
                    go.Scatter(
                        x=[q_val, q_val],
                        y=[0, kde(q_val)],
                        mode="lines",
                        name=f"{name} Q{int(q*100)}",
                        line=dict(color=COLORS[i % len(COLORS)], dash="dot", width=1),
                        showlegend=False,
                    )
                )
    fig.update_layout(
        title=config.title,
        xaxis_title=config.xaxis_title or config.labels.get("x") or "Value",
        yaxis_title=config.yaxis_title or config.labels.get("y") or "Density",
        legend_title=config.labels.get("legend", ""),
        template="plotly_white",
    )
    return fig
