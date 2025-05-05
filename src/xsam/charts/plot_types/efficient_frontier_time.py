"""
Efficient frontier composition over time for xsam charts module.

Stacked area chart: time on x-axis, allocation on y-axis, multiple assets/strategies as areas.
"""

from attrs import define, field
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from itertools import cycle


@define(slots=True, frozen=True)
class EfficientFrontierTimeChartConfig:
    y_columns: list[str]
    x_column: str | None = None
    title: str = ""
    labels: dict[str, str] = field(factory=dict)
    xaxis_title: str | None = None
    yaxis_title: str | None = None
    legendgroup: str = "group"


def plot_efficient_frontier_time_chart(
    df: pd.DataFrame,
    config: EfficientFrontierTimeChartConfig,
) -> go.Figure:
    """
    Plot a stacked area chart for efficient frontier composition over time.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (EfficientFrontierTimeConfig): Configuration for the chart.

    Returns:
        go.Figure: Plotly Figure object representing the stacked area chart.
    """
    x = df[config.x_column] if config.x_column else df.index
    fig = go.Figure()
    color_cycle = cycle(px.colors.qualitative.Plotly)
    for i, col in enumerate(config.y_columns):
        color = next(color_cycle)
        name = config.y_columns[i] if config.y_columns and i < len(config.y_columns) else col
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                mode="lines",
                name=name,
                stackgroup="one",
                line=dict(dash="solid", width=2, color=color),
                legendgroup=config.legendgroup,
                legendgrouptitle_text=config.title,
            )
        )
    fig.update_layout(
        title=config.title,
        xaxis_title=config.xaxis_title or config.labels.get("x") or (config.x_column or "Time"),
        yaxis_title=config.yaxis_title or config.labels.get("y") or "Allocation",
    )
    return fig
