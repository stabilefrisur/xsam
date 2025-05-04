"""
Efficient frontier composition over time for xsam charts module.

Stacked area chart: time on x-axis, allocation on y-axis, multiple assets/strategies as areas.
"""

from attrs import define, field
import plotly.graph_objs as go
import pandas as pd


@define(slots=True, frozen=True)
class EfficientFrontierTimeConfig:
    y_columns: list[str]
    x_column: str | None = None
    title: str = ""
    labels: dict[str, str] = field(factory=dict)
    area_names: list[str] | None = None
    xaxis_title: str | None = None
    yaxis_title: str | None = None


@define
class EfficientFrontierTimeChartConfig:
    area_columns: list[str]
    x_column: str | None = None


def plot_efficient_frontier_time(
    df: pd.DataFrame,
    config: EfficientFrontierTimeConfig,
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
    for i, col in enumerate(config.y_columns):
        name = config.area_names[i] if config.area_names and i < len(config.area_names) else col
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                mode="lines",
                name=name,
                stackgroup="one"
            )
        )
    fig.update_layout(
        title=config.title,
        xaxis_title=config.xaxis_title or config.labels.get("x") or (config.x_column or "Time"),
        yaxis_title=config.yaxis_title or config.labels.get("y") or "Allocation",
        legend_title=config.labels.get("legend", ""),
    )
    return fig


def plot_efficient_frontier_time_chart(df: pd.DataFrame, config: EfficientFrontierTimeChartConfig) -> go.Figure:
    fig = go.Figure()
    x = df[config.x_column] if config.x_column and config.x_column in df.columns else df.index
    for col in config.area_columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=x,
                y=df[col],
                mode='lines',
                stackgroup='one',
                name=col
            ))
    return fig
