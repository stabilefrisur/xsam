"""
Chart arrangement config for xsam charts module.

Defines data models for arranging multiple plots in a figure, with serialization support.
"""

from attrs import define
from typing import Any

@define(slots=True, frozen=True)
class PlotConfig:
    plot_type: str | None = None  # None if this is a group/container
    config: dict[str, Any] | None = None  # None if this is a group/container
    children: list["PlotConfig"] | None = None  # Nested plots for groups/containers

    def to_dict(self) -> dict[str, Any]:
        if self.children is not None:
            return {
                "children": [child.to_dict() for child in self.children]
            }
        return {
            "plot_type": self.plot_type,
            "config": self.config
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PlotConfig":
        if "children" in data:
            return PlotConfig(
                plot_type=None,
                config=None,
                children=[PlotConfig.from_dict(child) for child in data["children"]]
            )
        return PlotConfig(
            plot_type=data["plot_type"],
            config=data["config"],
            children=None
        )

@define(slots=True, frozen=True)
class ChartConfig:
    plots: list[PlotConfig]
    layout: str | None = None  # e.g., '2x2', '1x2', etc.
    figure_title: str | None = None
    shared_legend: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plots": [p.to_dict() for p in self.plots],
            "layout": self.layout,
            "figure_title": self.figure_title,
            "shared_legend": self.shared_legend,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ChartConfig":
        plots = [PlotConfig.from_dict(p) for p in data["plots"]]
        return ChartConfig(
            plots=plots,
            layout=data.get("layout"),
            figure_title=data.get("figure_title"),
            shared_legend=data.get("shared_legend", False),
        )
