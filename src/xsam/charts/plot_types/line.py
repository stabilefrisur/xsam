"""
Line chart plotting for xsam charts module.

Implements a flexible line chart, supporting overlays for latest, median, and quantiles.
"""

from attrs import define, field
import plotly.graph_objs as go
import pandas as pd
from typing import Sequence
import plotly.express as px
from itertools import cycle


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
    Plot a line chart with options for latest, median, and quantiles.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (LineChartConfig): Configuration for the line chart.

    Returns:
        go.Figure: Plotly Figure object representing the line chart.
    """
    fig = go.Figure()
    for i, col in enumerate(config.columns):
        # Reset color cycle
        color_cycle = cycle(px.colors.qualitative.Plotly)
        name = col
        color = next(color_cycle)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=name,
                line=dict(dash="solid", width=2, color=color),
            )
        )
        if config.show_latest:
            color = next(color_cycle)
            fig.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[df[col].iloc[-1]],
                    mode="markers",
                    name=f"{name} Latest",
                    marker=dict(symbol="circle", size=10, color=color),
                )
            )
        if config.show_median:
            color = next(color_cycle)
            median_val = df[col].median()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=[median_val] * len(df),
                    mode="lines",
                    name=f"{name} Median",
                    line=dict(dash="dash", width=1, color=color),
                )
            )
        if config.quantiles:
            for q in config.quantiles:
                color = next(color_cycle)
                q_val = df[col].quantile(q)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=[q_val] * len(df),
                        mode="lines",
                        name=f"{name} Q{int(q * 100)}",
                        line=dict(dash="dot", width=1, color=color),
                    )
                )
    fig.update_layout(
        title=config.title,
        xaxis_title=config.xaxis_title or config.labels.get("x") or "Index",
        yaxis_title=config.yaxis_title or config.labels.get("y") or "Value",
        legend_title=config.labels.get("legend", ""),
    )
    return fig
