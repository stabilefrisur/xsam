"""
Tests for dual-axis line chart plotting in xsam charts module.
"""

import pandas as pd
from plotly.graph_objs import Figure
from xsam.charts.plot_types.dual_axis_line import DualAxisLineChartConfig, plot_dual_axis_line_chart


def test_dual_axis_line_chart_smoke() -> None:
    """Smoke test: should return a Figure for minimal valid input."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
    config = DualAxisLineChartConfig(left_y_column="A", right_y_column="B")
    fig = plot_dual_axis_line_chart(df, config)
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


def test_dual_axis_line_chart_labels() -> None:
    """Test custom axis and legend labels."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
    config = DualAxisLineChartConfig(
        left_y_column="A", right_y_column="B", left_y_name="Foo", right_y_name="Bar", title="Test"
    )
    fig = plot_dual_axis_line_chart(df, config)
    assert fig.layout.title.text == "Test"


def test_dual_axis_line_chart_empty_df() -> None:
    """Should handle empty DataFrame gracefully."""
    df = pd.DataFrame({"A": [], "B": []})
    config = DualAxisLineChartConfig(left_y_column="A", right_y_column="B")
    fig = plot_dual_axis_line_chart(df, config)
    assert isinstance(fig, Figure)
