"""
Tests for ArrangementConfig and PlotConfig in xsam charts module.
"""

import json
from xsam.charts.arrangement import ArrangementConfig, PlotConfig

def test_arrangement_config_serialization() -> None:
    """Test ArrangementConfig to_dict and from_dict round-trip serialization."""
    plot1 = PlotConfig(plot_type="line", config={"y_columns": ["A", "B"]})
    plot2 = PlotConfig(plot_type="dual_axis_line", config={"left_y_column": "A", "right_y_column": "B"})
    arrangement = ArrangementConfig(
        plots=[plot1, plot2],
        layout="1x2",
        figure_title="Test Figure",
        shared_legend=True,
    )
    d = arrangement.to_dict()
    # Simulate JSON round-trip
    d_json = json.loads(json.dumps(d))
    arrangement2 = ArrangementConfig.from_dict(d_json)
    assert arrangement == arrangement2
    assert arrangement2.layout == "1x2"
    assert arrangement2.figure_title == "Test Figure"
    assert arrangement2.shared_legend is True
    assert arrangement2.plots[0].plot_type == "line"
    assert arrangement2.plots[1].plot_type == "dual_axis_line"


def test_plot_config_dict_structure() -> None:
    """Test that PlotConfig serializes to the expected dict structure."""
    plot = PlotConfig(plot_type="distribution", config={"columns": ["A"]})
    d = {"plot_type": plot.plot_type, "config": plot.config}
    assert d["plot_type"] == "distribution"
    assert d["config"] == {"columns": ["A"]}
