"""
Line chart plotting for xsam charts module.

Implements a flexible line chart with up to 5 lines, supporting overlays for latest, median, and quantiles.
"""

from attrs import define, field
import plotly.graph_objs as go
import pandas as pd
from typing import Sequence
from .colors import COLORS


@define(slots=True, frozen=True)
class LineChartConfig:
    columns: Sequence[str]
    title: str = ""
    labels: dict[str, str] = field(factory=dict)
    show_latest: bool = False
    show_median: bool = False
    quantiles: Sequence[float] | None = None
    xaxis_title: str | None = None
    yaxis_title: str | None = None


def plot_line_chart(
    df: pd.DataFrame,
    config: LineChartConfig,
) -> go.Figure:
    """
    Plot a line chart with up to 5 lines, with options for latest, median, and quantiles.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (LineChartConfig): Configuration for the line chart.

    Returns:
        go.Figure: Plotly Figure object representing the line chart.
    """
    fig = go.Figure()
    for i, col in enumerate(config.columns[:5]):
        name = col
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=name,
                line=dict(color=COLORS[i % len(COLORS)]),
            )
        )
        if config.show_latest:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[df[col].iloc[-1]],
                    mode="markers",
                    name=f"{name} Latest",
                    marker=dict(symbol="circle", size=10, color=COLORS[i % len(COLORS)]),
                )
            )
        if config.show_median:
            median_val = df[col].median()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=[median_val] * len(df),
                    mode="lines",
                    name=f"{name} Median",
                    line=dict(dash="dash", color=COLORS[i % len(COLORS)], width=1),
                )
            )
        if config.quantiles:
            for q in config.quantiles:
                q_val = df[col].quantile(q)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=[q_val] * len(df),
                        mode="lines",
                        name=f"{name} Q{int(q * 100)}",
                        line=dict(dash="dot", color=COLORS[i % len(COLORS)], width=1),
                    )
                )
    fig.update_layout(
        title=config.title,
        xaxis_title=config.xaxis_title or config.labels.get("x") or "Index",
        yaxis_title=config.yaxis_title or config.labels.get("y") or "Value",
        legend_title=config.labels.get("legend", ""),
        template="plotly_white",
    )
    return fig
