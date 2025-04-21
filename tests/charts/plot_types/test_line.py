"""
Tests for line chart plotting in xsam charts module.
"""

import pandas as pd
from plotly.graph_objs import Figure
from xsam.charts.plot_types.line import LineChartConfig, plot_line_chart


def test_line_chart_smoke() -> None:
    """Smoke test: should return a Figure for minimal valid input."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
    config = LineChartConfig(columns=["A", "B"])
    fig = plot_line_chart(df, config)
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


def test_line_chart_with_options() -> None:
    """Test median and quantile overlays."""
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    config = LineChartConfig(columns=["A"], show_median=True, quantiles=[0.25, 0.75])
    fig = plot_line_chart(df, config)
    # 1 line + 1 median + 2 quantiles = 4 traces
    assert len(fig.data) == 4


def test_line_chart_empty_df() -> None:
    """Should handle empty DataFrame gracefully."""
    df = pd.DataFrame({"A": []})
    config = LineChartConfig(columns=["A"])
    fig = plot_line_chart(df, config)
    assert isinstance(fig, Figure)
