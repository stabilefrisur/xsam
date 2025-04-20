"""
Tests for distribution (KDE) chart in xsam charts module.
"""

import pandas as pd
import numpy as np
from plotly.graph_objs import Figure
from xsam.charts.plot_types.distribution import DistributionChartConfig, plot_distribution_chart


def test_distribution_chart_smoke() -> None:
    """Smoke test: should return a Figure for minimal valid input."""
    df = pd.DataFrame({"A": np.random.normal(0, 1, 100)})
    config = DistributionChartConfig(columns=["A"])
    fig = plot_distribution_chart(df, config)
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1


def test_distribution_chart_with_quantiles() -> None:
    """Test quantile lines for KDE plot."""
    df = pd.DataFrame({"A": np.random.normal(0, 1, 100)})
    config = DistributionChartConfig(columns=["A"], quantiles={"A": [0.25, 0.75]})
    fig = plot_distribution_chart(df, config)
    # 1 KDE + 2 quantile lines = 3 traces
    assert len(fig.data) == 3


def test_distribution_chart_multiple_columns() -> None:
    """Test up to 3 KDE plots."""
    df = pd.DataFrame({"A": np.random.normal(0, 1, 100), "B": np.random.normal(1, 1, 100), "C": np.random.normal(-1, 1, 100)})
    config = DistributionChartConfig(columns=["A", "B", "C"])
    fig = plot_distribution_chart(df, config)
    assert len(fig.data) == 3


def test_distribution_chart_empty_df() -> None:
    """Should handle empty DataFrame gracefully."""
    df = pd.DataFrame({"A": []})
    config = DistributionChartConfig(columns=["A"])
    fig = plot_distribution_chart(df, config)
    assert isinstance(fig, Figure)


def test_distribution_chart_with_enough_data() -> None:
    """Should plot KDE when there are at least 2 data points."""
    df = pd.DataFrame({"A": np.random.normal(0, 1, 10)})
    config = DistributionChartConfig(columns=["A"])
    fig = plot_distribution_chart(df, config)
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1
