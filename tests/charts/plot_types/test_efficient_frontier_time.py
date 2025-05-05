"""
Tests for efficient frontier composition over time chart in xsam charts module.
"""

import pandas as pd
from plotly.graph_objs import Figure
from xsam.charts.plot_types.efficient_frontier_time import EfficientFrontierTimeConfig, plot_efficient_frontier_time_chart


def test_efficient_frontier_time_smoke() -> None:
    """Smoke test: should return a Figure for minimal valid input."""
    df = pd.DataFrame({"t": [1, 2, 3], "A": [0.2, 0.3, 0.5], "B": [0.8, 0.7, 0.5]})
    config = EfficientFrontierTimeConfig(y_columns=["A", "B"], x_column="t")
    fig = plot_efficient_frontier_time_chart(df, config)
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


def test_efficient_frontier_time_area_names() -> None:
    """Test custom area names."""
    df = pd.DataFrame({"t": [1, 2, 3], "A": [0.2, 0.3, 0.5], "B": [0.8, 0.7, 0.5]})
    config = EfficientFrontierTimeConfig(y_columns=["A", "B"], x_column="t", y_columns=["foo", "bar"])
    fig = plot_efficient_frontier_time_chart(df, config)
    assert fig.data[0].name == "foo"
    assert fig.data[1].name == "bar"


def test_efficient_frontier_time_empty_df() -> None:
    """Should handle empty DataFrame gracefully."""
    df = pd.DataFrame({"t": [], "A": [], "B": []})
    config = EfficientFrontierTimeConfig(y_columns=["A", "B"], x_column="t")
    fig = plot_efficient_frontier_time_chart(df, config)
    assert isinstance(fig, Figure)
