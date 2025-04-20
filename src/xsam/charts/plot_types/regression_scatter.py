"""
Regression scatter fields for xsam charts module.

Scatter chart with optional regression line, standard error band, and latest observation highlight.
"""

from attrs import define, field
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy import stats
from .colors import COLORS


@define(slots=True, frozen=True)
class RegressionScatterConfig:
    x_column: str
    y_column: str
    title: str = ""
    labels: dict[str, str] = field(factory=dict)
    regression_line: bool = False
    show_std_err: bool = False
    highlight_latest: bool = False
    xaxis_title: str | None = None
    yaxis_title: str | None = None
    scatter_name: str = "Data"


def plot_regression_scatter(
    df: pd.DataFrame,
    config: RegressionScatterConfig,
) -> go.Figure:
    """
    Plot a scatter chart with optional regression line, standard error band, and latest observation highlight.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (RegressionScatterConfig): Configuration for the chart.

    Returns:
        go.Figure: Plotly Figure object representing the scatter chart.
    """
    x = df[config.x_column]
    y = df[config.y_column]
    fig = go.Figure()
    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=config.scatter_name,
            marker=dict(color=COLORS[0], opacity=0.6),
        )
    )
    # Regression line and std error
    if config.regression_line or config.show_std_err:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_sorted = np.sort(x)
        y_pred = slope * x_sorted + intercept
        if config.regression_line:
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_pred,
                    mode="lines",
                    name="Regression Line",
                    line=dict(color=COLORS[1], dash="dash"),
                )
            )
        if config.show_std_err:
            y_upper = y_pred + std_err
            y_lower = y_pred - std_err
            fillcolor = f"rgba({int(COLORS[1][1:3],16)},{int(COLORS[1][3:5],16)},{int(COLORS[1][5:7],16)},0.2)"
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_sorted, x_sorted[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Std Error",
                )
            )
    # Highlight latest observation
    if config.highlight_latest:
        fig.add_trace(
            go.Scatter(
                x=[x.iloc[-1]],
                y=[y.iloc[-1]],
                mode="markers",
                name="Latest",
                marker=dict(color=COLORS[2], size=12, symbol="star"),
            )
        )
    fig.update_layout(
        title=config.title,
        xaxis_title=config.xaxis_title or config.labels.get("x") or config.x_column,
        yaxis_title=config.yaxis_title or config.labels.get("y") or config.y_column,
        legend_title=config.labels.get("legend", ""),
        template="plotly_white",
    )
    return fig
