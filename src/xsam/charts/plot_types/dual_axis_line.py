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
    fig = go.Figure()
    # Left Y axis
    if config.left_y_column in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[config.left_y_column],
            name=config.left_y_column,
            yaxis="y1",
            line=dict(color=COLORS[0])
        ))
    # Right Y axis
    if config.right_y_column in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[config.right_y_column],
            name=config.right_y_column,
            yaxis="y2",
            line=dict(color=COLORS[1])
        ))
    fig.update_layout(
        yaxis=dict(
            title=config.left_y_column,
            showgrid=True,
            zeroline=False,
            showline=True,
            showticklabels=True,
        ),
        yaxis2=dict(
            title=config.right_y_column,
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            anchor='x',
            showline=True,
            showticklabels=True,
            position=1
        ),
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
    )
    return fig
