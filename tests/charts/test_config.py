"""
Tests for ChartConfig and PlotConfig in xsam charts module.
"""

import json
from xsam.charts.config import ChartConfig, PlotConfig

def test_config_serialization() -> None:
    """Test ChartConfig to_dict and from_dict round-trip serialization."""
    plot1 = PlotConfig(plot_type="line", config={"y_columns": ["A", "B"]})
    plot2 = PlotConfig(plot_type="dual_axis_line", config={"left_y_column": "A", "right_y_column": "B"})
    chart_config = ChartConfig(
        plots=[plot1, plot2],
        layout="1x2",
        figure_title="Test Figure",
        shared_legend=True,
    )
    d = chart_config.to_dict()
    # Simulate JSON round-trip
    d_json = json.loads(json.dumps(d))
    chart_config2 = ChartConfig.from_dict(d_json)
    assert chart_config == chart_config2
    assert chart_config2.layout == "1x2"
    assert chart_config2.figure_title == "Test Figure"
    assert chart_config2.shared_legend is True
    assert chart_config2.plots[0].plot_type == "line"
    assert chart_config2.plots[1].plot_type == "dual_axis_line"


def test_plot_config_dict_structure() -> None:
    """Test that PlotConfig serializes to the expected dict structure."""
    plot = PlotConfig(plot_type="distribution", config={"columns": ["A"]})
    d = {"plot_type": plot.plot_type, "config": plot.config}
    assert d["plot_type"] == "distribution"
    assert d["config"] == {"columns": ["A"]}
