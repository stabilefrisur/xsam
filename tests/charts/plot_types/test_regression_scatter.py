"""
Tests for regression scatter chart in xsam charts module.
"""

import pandas as pd
from plotly.graph_objs import Figure
from xsam.charts.plot_types.regression_scatter import RegressionScatterConfig, plot_regression_scatter


def test_regression_scatter_smoke() -> None:
    """Smoke test: should return a Figure for minimal valid input."""
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    config = RegressionScatterConfig(x_column="x", y_column="y")
    fig = plot_regression_scatter(df, config)
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1


def test_regression_scatter_with_regression() -> None:
    """Test regression line and std error band."""
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    config = RegressionScatterConfig(x_column="x", y_column="y", regression_line=True, show_std_err=True)
    fig = plot_regression_scatter(df, config)
    # 1 scatter + 1 regression + 1 std error = 3 traces
    assert len(fig.data) == 3


def test_regression_scatter_highlight_latest() -> None:
    """Test highlighting the latest observation."""
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    config = RegressionScatterConfig(x_column="x", y_column="y", highlight_latest=True)
    fig = plot_regression_scatter(df, config)
    assert any(trace.name == "Latest" for trace in fig.data)


def test_regression_scatter_empty_df() -> None:
    """Should handle empty DataFrame gracefully."""
    df = pd.DataFrame({"x": [], "y": []})
    config = RegressionScatterConfig(x_column="x", y_column="y")
    fig = plot_regression_scatter(df, config)
    assert isinstance(fig, Figure)
