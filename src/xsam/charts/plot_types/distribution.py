"""
Distribution (KDE) chart for xsam charts module.

Plots up to 5 KDE curves. Optional overlays (latest, median, quantiles) are shown as vertical lines.
"""

from attrs import define, field
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from typing import Sequence
from .colors import COLORS
from scipy.stats import gaussian_kde

@define(slots=True, frozen=True)
class DistributionChartConfig:
    columns: Sequence[str]
    title: str = ""
    labels: dict[str, str] = field(factory=dict)
    show_latest: bool = False
    show_median: bool = False
    quantiles: Sequence[float] | None = None
    xaxis_title: str | None = None
    yaxis_title: str | None = None
    kde_points: int = 200

def plot_distribution_chart(
    df: pd.DataFrame,
    config: DistributionChartConfig,
) -> go.Figure:
    """
    Plot up to 5 KDE curves with optional vertical overlays for latest, median, and quantiles.
    """
    fig = go.Figure()
    max_y = 0.0
    kde_results = []
    for i, col in enumerate(config.columns[:5]):
        data = df[col].dropna()
        if data.empty:
            continue
        kde = gaussian_kde(data)
        x_grid = pd.Series(data).sort_values()
        x_min, x_max = x_grid.iloc[0], x_grid.iloc[-1]
        x_vals = pd.Series(np.linspace(x_min, x_max, config.kde_points))
        y_vals = kde(x_vals)
        max_y = max(max_y, y_vals.max())
        kde_results.append((i, col, x_vals, y_vals))
    for i, col, x_vals, y_vals in kde_results:
        name = col
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=name,
                line=dict(color=COLORS[i % len(COLORS)]),
            )
        )
        overlays = []
        data = df[col].dropna()
        if config.show_latest:
            overlays.append((data.iloc[-1], f"{name} Latest", "solid"))
        if config.show_median:
            overlays.append((data.median(), f"{name} Median", "dash"))
        if config.quantiles:
            for q in config.quantiles:
                overlays.append((data.quantile(q), f"{name} Q{int(q*100)}", "dot"))
        for val, label, dash in overlays:
            fig.add_trace(
                go.Scatter(
                    x=[val, val],
                    y=[0, max_y * 1.05],
                    mode="lines",
                    name=label,
                    line=dict(color=COLORS[i % len(COLORS)], dash=dash, width=1),
                    showlegend=True,
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

