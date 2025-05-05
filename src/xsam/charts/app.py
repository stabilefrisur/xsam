"""
Dash app UI for xsam charts module.

Implements a responsive, best-practice dashboard for arranging and visualizing charts.

To use a programmatically provided DataFrame, call run_dash_app(df: pd.DataFrame).
"""

# === Imports ===
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
from dash.dependencies import ALL
from io import StringIO
import plotly.graph_objects as go

# === Constants ===
TEMPLATE = "simple_white"
external_stylesheets = [dbc.themes.SKETCHY]

# === Helper Functions ===
def safe_get(lst, idx, default):
    """Safely get an item from a list, or return default if out of range or None."""
    return lst[idx] if idx < len(lst) and lst[idx] is not None else default

def pad_or_truncate(lst, target_len, default):
            # Always return a value of the correct type, never None or undefined
            if len(lst) < target_len:
                return lst + [default] * (target_len - len(lst))
            elif len(lst) > target_len:
                return lst[:target_len]
            return lst

# === Main App Entrypoint ===
def run_dash_app(df: pd.DataFrame | None = None) -> None:
    """
    Run the Dash app. Must provide a DataFrame.

    Args:
        df (pd.DataFrame | None): DataFrame to use for plotting.
    """
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.title = "XSAM Charts"

    # === Layout ===
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("XSAM Charts", className="mb-2 mt-2"),
                dcc.Store(id="stored-data", data=df.to_json(date_format="iso")),
                dcc.Store(id="loaded-trigger"),
                html.Hr(),
                # dbc.Row([
                #     dbc.Col([
                #         html.Button("Save", id="save", className="btn btn-primary w-100 mt-2 mb-2"),
                #         dcc.Download(id="download")
                #     ], width=6),
                #     dbc.Col([
                #         html.Button("Load", id="load", className="btn btn-primary w-100 mt-2 mb-2"),
                #         dcc.Upload(id="upload")
                #     ], width=6),
                # ], className="mt-2 mb-2"),
                html.Label("Layout", className="mt-2"),
                dcc.Dropdown(
                    id="layout",
                    options=[{"label": layout_label, "value": layout_label} for layout_label in ["1x1", "1x2", "2x1", "2x2", "3x1", "4x1", "5x1"]],
                    value="1x1",
                    clearable=False,
                    className="mb-2"
                ),
                html.Label("Figure Title", className="mt-2"),
                dcc.Input(id="figure-title", type="text", value="", className="mb-2", style={"width": "100%"}),
                html.Hr(),
                html.Div(id="plot-title-controls", className="mt-2 mb-2"),
                html.Hr(),
                html.Div(id="plot-type-controls", className="mt-2 mb-2"),
                html.Hr(),
                html.Div(id="plot-config-controls", className="mt-2 mb-2"),
                html.Div(id="feedback", className="mt-2 mb-2"),
            ], width=3, style={"backgroundColor": "#f8f9fa", "padding": "24px", "minHeight": "100vh"}),
            dbc.Col([
                dcc.Loading(
                    id="loading-charts",
                    type="circle",
                    children=html.Div(id="charts-area")
                )
            ], width=9),
        ], className="gx-4"),
    ], fluid=True)

    # === Callbacks ===
    @app.callback(
        Output("plot-title-controls", "children"),
        Input("layout", "value"),
        Input({'type': 'plot-title', 'index': ALL}, 'value'),
    )
    def update_plot_title_controls(layout, plot_titles):
        n_plots = {"1x1": 1, "1x2": 2, "2x1": 2, "2x2": 4, "3x1": 3, "4x1": 4, "5x1": 5}[layout]
        controls = []
        for i in range(n_plots):
            controls.append(
                html.Div([
                    html.Label(f"Plot {i+1} Title"),
                    dcc.Input(id={'type': 'plot-title', 'index': i}, type="text", value=safe_get(plot_titles, i, ""), style={"width": "100%"}, className="mb-2"),
                ], className="mb-2")
            )
        return controls

    @app.callback(
        Output("plot-type-controls", "children"),
        Output("plot-config-controls", "children"),
        Input("stored-data", "data"),
        Input("layout", "value"),
        Input({'type': 'plot-type', 'index': ALL}, 'value'),
        State({'type': 'plot-config', 'index': ALL}, 'value'),
        State({'type': 'show-latest', 'index': ALL}, 'value'),
        State({'type': 'show-median', 'index': ALL}, 'value'),
        State({'type': 'quantiles', 'index': ALL}, 'value'),
        State({'type': 'left-y', 'index': ALL}, 'value'),
        State({'type': 'right-y', 'index': ALL}, 'value'),
        State({'type': 'areas', 'index': ALL}, 'value'),
        State({'type': 'xcol_ef', 'index': ALL}, 'value'),
        State({'type': 'xcol', 'index': ALL}, 'value'),
        State({'type': 'ycol', 'index': ALL}, 'value'),
        State({'type': 'reg-line', 'index': ALL}, 'value'),
        State({'type': 'std-err', 'index': ALL}, 'value'),
        State({'type': 'highlight-latest', 'index': ALL}, 'value'),
    )
    def update_plot_controls(
        data_json: str | None, layout: str, plot_types: list[str | None],
        plot_configs=None, show_latest=None, show_median=None, quantiles=None,
        left_y=None, right_y=None, areas=None, xcol_ef=None, xcol=None, ycol=None, reg_line=None, std_err=None, highlight_latest=None
    ) -> tuple[list, list]:
        """
        Dynamically generate plot type and config controls based on the DataFrame, layout, and selected plot types.
        Persist state for each plot's controls.
        """
        if not data_json:
            return [], []
        df = pd.read_json(StringIO(data_json))
        n_plots = {"1x1": 1, "1x2": 2, "2x1": 2, "2x2": 4, "3x1": 3, "4x1": 4, "5x1": 5}[layout]
        plot_type_controls = []
        plot_config_controls = []
        for i in range(n_plots):
            plot_type_id = {'type': 'plot-type', 'index': i}
            current_type = plot_types[i] if i < len(plot_types) and plot_types[i] else "line"
            plot_type_controls.append(
                html.Div([
                    html.Label(f"Plot {i+1} Type", title="Select the type of plot for this subplot."),
                    dcc.Dropdown(
                        id=plot_type_id,
                        options=[
                            {"label": "Line Chart", "value": "line"},
                            {"label": "Dual Axis Line", "value": "dual_axis_line"},
                            {"label": "Efficient Frontier Time", "value": "efficient_frontier_time"},
                            {"label": "Regression Scatter", "value": "regression_scatter"},
                            {"label": "Distribution (KDE)", "value": "distribution"},
                        ],
                        value=current_type,
                        clearable=False,
                        className="mb-2"
                    )
                ], className="mb-4")
            )
            # Always include all controls, hide those not relevant for the current type
            plot_config_controls.append(
                html.Div([
                    # Line/Distribution controls
                    html.Div([
                        html.Label(f"Plot {i+1} Columns", title="Select columns."),
                        dcc.Dropdown(
                            id={'type': 'plot-config', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            multi=True,
                            maxHeight=200,
                            className="mb-2",
                            value=safe_get(plot_configs or [], i, []),
                        ),
                        dbc.Checkbox(id={'type': 'show-latest', 'index': i}, label="Latest Marker", className="mb-1", value=safe_get(show_latest or [], i, False)),
                        dbc.Checkbox(id={'type': 'show-median', 'index': i}, label="Median Line", className="mb-1", value=safe_get(show_median or [], i, False)),
                        html.Label("Quantiles (e.g. 0.25,0.75)", className="mb-1"),
                        dcc.Input(id={'type': 'quantiles', 'index': i}, type="text", placeholder="", className="mb-1", debounce=True, style={"width": "100%"}, value=safe_get(quantiles or [], i, "")),
                    ], style={"display": "block" if current_type in ["line", "distribution"] else "none"}),
                    # Dual axis controls
                    html.Div([
                        html.Label(f"Plot {i+1} Left Y Column"),
                        dcc.Dropdown(
                            id={'type': 'left-y', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2",
                            value=safe_get(left_y or [], i, None),
                        ),
                        html.Label(f"Plot {i+1} Right Y Column"),
                        dcc.Dropdown(
                            id={'type': 'right-y', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2",
                            value=safe_get(right_y or [], i, None),
                        ),
                    ], style={"display": "block" if current_type == "dual_axis_line" else "none"}),
                    # Efficient frontier controls
                    html.Div([
                        html.Label(f"Plot {i+1} Area Columns (allocations)", title="Select columns for stacked areas."),
                        dcc.Dropdown(
                            id={'type': 'areas', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            multi=True,
                            className="mb-2",
                            value=safe_get(areas or [], i, []),
                        ),
                        html.Label(f"Plot {i+1} X Column (time)"),
                        dcc.Dropdown(
                            id={'type': 'xcol_ef', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2",
                            value=safe_get(xcol_ef or [], i, None),
                        ),
                    ], style={"display": "block" if current_type == "efficient_frontier_time" else "none"}),
                    # Regression scatter controls
                    html.Div([
                        html.Label(f"Plot {i+1} X Column"),
                        dcc.Dropdown(
                            id={'type': 'xcol', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2",
                            value=safe_get(xcol or [], i, None),
                        ),
                        html.Label(f"Plot {i+1} Y Column"),
                        dcc.Dropdown(
                            id={'type': 'ycol', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2",
                            value=safe_get(ycol or [], i, None),
                        ),
                        dbc.Checkbox(id={'type': 'reg-line', 'index': i}, label="Regression Line", className="mb-1", value=safe_get(reg_line or [], i, False)),
                        dbc.Checkbox(id={'type': 'std-err', 'index': i}, label="Std Error Band", className="mb-1", value=safe_get(std_err or [], i, False)),
                        dbc.Checkbox(id={'type': 'highlight-latest', 'index': i}, label="Latest Marker", className="mb-1", value=safe_get(highlight_latest or [], i, False)),
                    ], style={"display": "block" if current_type == "regression_scatter" else "none"}),
                ], className="mb-4")
            )
        return plot_type_controls, plot_config_controls

    @app.callback(
        Output("charts-area", "children"),
        Input("stored-data", "data"),
        Input("layout", "value"),
        Input("figure-title", "value"),
        Input({'type': 'plot-title', 'index': ALL}, 'value'),
        Input({'type': 'plot-type', 'index': ALL}, 'value'),
        Input({'type': 'plot-config', 'index': ALL}, 'value'),
        Input({'type': 'show-latest', 'index': ALL}, 'value'),
        Input({'type': 'show-median', 'index': ALL}, 'value'),
        Input({'type': 'quantiles', 'index': ALL}, 'value'),
        Input({'type': 'left-y', 'index': ALL}, 'value'),
        Input({'type': 'right-y', 'index': ALL}, 'value'),
        Input({'type': 'areas', 'index': ALL}, 'value'),
        Input({'type': 'xcol_ef', 'index': ALL}, 'value'),
        Input({'type': 'xcol', 'index': ALL}, 'value'),
        Input({'type': 'ycol', 'index': ALL}, 'value'),
        Input({'type': 'reg-line', 'index': ALL}, 'value'),
        Input({'type': 'std-err', 'index': ALL}, 'value'),
        Input({'type': 'highlight-latest', 'index': ALL}, 'value'),
        Input("loaded-trigger", "data"),
        prevent_initial_call=True,
    )
    def render_charts(
        data_json: str | None,
        layout: str,
        figure_title: str | None,
        plot_titles: list[str | None],
        plot_types: list[str | None],
        plot_configs: list[list[str] | None],
        show_latest: list[bool | None],
        show_median: list[bool | None],
        quantiles: list[str | None],
        left_y: list[str | None],
        right_y: list[str | None],
        areas: list[list[str] | None],
        xcol_ef: list[str | None],
        xcol: list[str | None],
        ycol: list[str | None],
        reg_line: list[bool | None],
        std_err: list[bool | None],
        highlight_latest: list[bool | None],
        _trigger: str | None,
    ) -> list:
        from plotly.subplots import make_subplots
        if not data_json:
            return [html.Div("No data uploaded.")]
        df = pd.read_json(StringIO(data_json))
        layout_map = {
            "1x1": (1, 1),
            "1x2": (1, 2),
            "2x1": (2, 1),
            "2x2": (2, 2),
            "3x1": (3, 1),
            "4x1": (4, 1),
            "5x1": (5, 1),
        }
        rows, cols = layout_map.get(layout, (1, 1))
        n_subplots = rows * cols
        subplot_titles = [safe_get(plot_titles, i, "") for i in range(n_subplots)]
        vertical_spacing = 0.2 / rows
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)],
            horizontal_spacing=0.1,
            vertical_spacing=vertical_spacing,
        )
        for i in range(n_subplots):
            plot_type = safe_get(plot_types, i, "line")
            plot_title = safe_get(plot_titles, i, f"Plot {i+1}")
            row = (i // cols) + 1
            col = (i % cols) + 1
            legend_group = f"group{i+1}"
            if plot_type == "line":
                y_columns = safe_get(plot_configs, i, [])
                show_latest_i = safe_get(show_latest, i, False)
                show_median_i = safe_get(show_median, i, False)
                quantiles_i = safe_get(quantiles, i, "")
                from xsam.charts.plot_types.line import LineChartConfig, plot_line_chart
                quantiles_list = None
                if quantiles_i:
                    try:
                        quantiles_list = [float(q.strip()) for q in quantiles_i.split(",") if q.strip()]
                    except Exception:
                        quantiles_list = None
                config = LineChartConfig(
                    columns=y_columns,
                    title=plot_title,
                    show_latest=bool(show_latest_i),
                    show_median=bool(show_median_i),
                    quantiles=quantiles_list,
                    legendgroup=legend_group,
                )
                subfig = plot_line_chart(df, config)
                for trace in subfig.data:
                    fig.add_trace(trace, row=row, col=col)
            elif plot_type == "dual_axis_line":
                left = safe_get(left_y, i, None)
                right = safe_get(right_y, i, None)
                from xsam.charts.plot_types.dual_axis_line import DualAxisLineChartConfig, plot_dual_axis_line_chart
                config = DualAxisLineChartConfig(
                    left_y_column=left,
                    right_y_column=right,
                    title=plot_title,
                    legendgroup=legend_group,
                )
                if left or right:
                    subfig = plot_dual_axis_line_chart(df, config)
                    for trace in subfig.data:
                        is_right = hasattr(trace, 'yaxis') and getattr(trace, 'yaxis', None) == 'y2'
                        fig.add_trace(trace, row=row, col=col, secondary_y=is_right)
            elif plot_type == "efficient_frontier_time":
                area_cols = safe_get(areas, i, [])
                x_col = safe_get(xcol_ef, i, None)
                if area_cols:
                    from xsam.charts.plot_types.efficient_frontier_time import EfficientFrontierTimeChartConfig, plot_efficient_frontier_time_chart
                    config = EfficientFrontierTimeChartConfig(
                        y_columns=area_cols,
                        x_column=x_col or (df.columns[0] if len(df.columns) > 0 else None),
                        title=plot_title,
                        legendgroup=legend_group,
                    )
                    subfig = plot_efficient_frontier_time_chart(df, config)
                    for trace in subfig.data:
                        fig.add_trace(trace, row=row, col=col)
            elif plot_type == "regression_scatter":
                x = safe_get(xcol, i, None)
                y = safe_get(ycol, i, None)
                reg_line_i = bool(safe_get(reg_line, i, False))
                std_err_i = bool(safe_get(std_err, i, False))
                highlight_latest_i = bool(safe_get(highlight_latest, i, False))
                if not x or not y:
                    fig.add_annotation(
                        text=f"Select both X and Y columns for Regression Scatter (Plot {i+1})<br>xcol: {x!r}, ycol: {y!r}",
                        xref="paper", yref="paper",
                        x=(i % cols + 0.5) / cols, y=1 - (i // cols + 0.5) / rows,
                        showarrow=False, font=dict(color="darkgray", size=14),
                    )
                    continue
                from xsam.charts.plot_types.regression_scatter import RegressionScatterChartConfig, plot_regression_scatter_chart
                config = RegressionScatterChartConfig(
                    x_column=x,
                    y_column=y,
                    title=plot_title,
                    regression_line=reg_line_i,
                    show_std_err=std_err_i,
                    highlight_latest=highlight_latest_i,
                    legendgroup=legend_group,
                )
                subfig = plot_regression_scatter_chart(df, config)
                for trace in subfig.data:
                    fig.add_trace(trace, row=row, col=col)
            elif plot_type == "distribution":
                y_columns = safe_get(plot_configs, i, [])
                show_latest_i = safe_get(show_latest, i, False)
                show_median_i = safe_get(show_median, i, False)
                quantiles_i = safe_get(quantiles, i, "")
                from xsam.charts.plot_types.distribution import DistributionChartConfig, plot_distribution_chart
                quantiles_list = None
                if quantiles_i:
                    try:
                        quantiles_list = [float(q.strip()) for q in quantiles_i.split(",") if q.strip()]
                    except Exception:
                        quantiles_list = None
                config = DistributionChartConfig(
                    columns=y_columns,
                    title=plot_title,
                    show_latest=bool(show_latest_i),
                    show_median=bool(show_median_i),
                    quantiles=quantiles_list,
                    legendgroup=legend_group,
                )
                subfig = plot_distribution_chart(df, config)
                for trace in subfig.data:
                    fig.add_trace(trace, row=row, col=col)
        fig.update_layout(
            height=400 * rows / (1 - vertical_spacing * (rows - 1)),
            legend=dict(groupclick="toggleitem"),
            showlegend=True,
            margin=dict(t=80, l=5, r=5, b=5),
            title={
                'text': figure_title or "",
                'font': {'size': 28}
            },
            template=TEMPLATE
        )
        return [dcc.Graph(figure=fig, config={"responsive": True}, style={"height": f"{400*rows}px"})]

    app.run(debug=True)

if __name__ == "__main__":
    # Generate a sample DataFrame. Use random numbers that are somewhat similar to price data.
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Date": dates,
        "Price": np.random.normal(loc=100, scale=10, size=len(dates)),
        "Volume": np.random.randint(1000, 5000, size=len(dates)),
        "Open": np.random.normal(loc=100, scale=5, size=len(dates)),
        "Close": np.random.normal(loc=100, scale=5, size=len(dates)),
    })
    data.set_index("Date", inplace=True)
    # Run the Dash app with the sample DataFrame  
    run_dash_app(data)
