"""
Chart arrangement config for xsam charts module.

Defines data models for arranging multiple plots in a figure, with serialization support.
"""

from attrs import define, field
from typing import Any

@define(slots=True, frozen=True)
class PlotConfig:
    plot_type: str
    config: dict[str, Any]

@define(slots=True, frozen=True)
class ArrangementConfig:
    plots: list[PlotConfig]
    layout: str | None = None  # e.g., '2x2', '1x2', etc.
    figure_title: str | None = None
    shared_legend: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize arrangement config to a dict."""
        return {
            "plots": [
                {"plot_type": p.plot_type, "config": p.config} for p in self.plots
            ],
            "layout": self.layout,
            "figure_title": self.figure_title,
            "shared_legend": self.shared_legend,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ArrangementConfig":
        """Deserialize arrangement config from a dict."""
        plots = [PlotConfig(plot_type=p["plot_type"], config=p["config"]) for p in data["plots"]]
        return ArrangementConfig(
            plots=plots,
            layout=data.get("layout"),
            figure_title=data.get("figure_title"),
            shared_legend=data.get("shared_legend", False),
        )
