"""
Dual-axis line chart plotting for xsam charts module.

Implements a line chart with two y-axes (left and right), supporting custom titles and labels.
"""

from attrs import define, field
import plotly.graph_objs as go
import pandas as pd
from .colors import COLORS


@define(slots=True, frozen=True)
class DualAxisLineChartConfig:
    left_y_column: str
    right_y_column: str
    x_column: str | None = None
    title: str = ""
    left_y_name: str = "Left Y"
    right_y_name: str = "Right Y"
    xaxis_title: str | None = None
    left_yaxis_title: str | None = None
    right_yaxis_title: str | None = None
    labels: dict[str, str] = field(factory=dict)


def plot_dual_axis_line_chart(
    df: pd.DataFrame,
    config: DualAxisLineChartConfig,
) -> go.Figure:
    """
    Plot a line chart with two y-axes (left and right).

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (DualAxisLineChartConfig): Configuration for the dual-axis line chart.

    Returns:
        go.Figure: Plotly Figure object representing the dual-axis line chart.
    """
    x = df[config.x_column] if config.x_column else df.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[config.left_y_column],
            mode="lines",
            name=config.left_y_name,
            line=dict(color=COLORS[0]),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[config.right_y_column],
            mode="lines",
            name=config.right_y_name,
            line=dict(color=COLORS[1]),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=config.title,
        xaxis=dict(title=config.xaxis_title or config.labels.get("x") or (config.x_column or "Index")),
        yaxis=dict(
            title=dict(
                text=config.left_yaxis_title or config.labels.get("left_y") or config.left_y_name,
                font=dict(color=COLORS[0]),
            ),
            tickfont=dict(color=COLORS[0]),
        ),
        yaxis2=dict(
            title=dict(
                text=config.right_yaxis_title or config.labels.get("right_y") or config.right_y_name,
                font=dict(color=COLORS[1]),
            ),
            tickfont=dict(color=COLORS[1]),
            overlaying="y",
            side="right",
        ),
        legend_title=config.labels.get("legend", ""),
        template="plotly_white",
    )
    return fig
