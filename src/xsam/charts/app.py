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
import time

# === Constants ===
external_stylesheets = [dbc.themes.BOOTSTRAP]

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
                dbc.Row([
                    dbc.Col([
                        html.Button("Save", id="save", className="btn btn-primary w-100 mt-2 mb-2"),
                        dcc.Download(id="download")
                    ], width=6),
                    dbc.Col([
                        html.Button("Load", id="load", className="btn btn-primary w-100 mt-2 mb-2"),
                        dcc.Upload(id="upload")
                    ], width=6),
                ], className="mt-2 mb-2"),
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
                html.Div(id="plot-title-controls", className="mt-2 mb-2"),
                html.Div(id="plot-type-controls", className="mt-2 mb-4"),
                html.Hr(),
                html.Div(id="plot-config-controls", className="mt-4 mb-2"),
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
                    dcc.Input(id={'type': 'plot-title', 'index': i}, type="text", value=safe_get(plot_titles, i, f"Plot {i+1}"), style={"width": "100%"}, className="mb-2"),
                ], className="mb-2")
            )
        return controls

    @app.callback(
        Output("plot-type-controls", "children"),
        Output("plot-config-controls", "children"),
        Input("stored-data", "data"),
        Input("layout", "value"),
        Input({'type': 'plot-type', 'index': ALL}, 'value'),
    )
    def update_plot_controls(data_json: str | None, layout: str, plot_types: list[str | None]) -> tuple[list, list]:
        """
        Dynamically generate plot type and config controls based on the DataFrame, layout, and selected plot types.
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
                ], className="mb-3")
            )
            if current_type == "line":
                plot_config_controls.append(
                    html.Div([
                        html.Label(f"Plot {i+1} Columns", title="Select columns."),
                        dcc.Dropdown(
                            id={'type': 'plot-config', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            multi=True,
                            maxHeight=200,
                            className="mb-2"
                        ),
                        dbc.Checkbox(id={'type': 'show-latest', 'index': i}, label="Show Latest Marker", className="mb-1"),
                        dbc.Checkbox(id={'type': 'show-median', 'index': i}, label="Show Median Line", className="mb-1"),
                        html.Label("Quantiles (comma-separated, e.g. 0.25,0.75)", className="mt-1"),
                        dcc.Input(id={'type': 'quantiles', 'index': i}, type="text", placeholder="", className="mb-2", debounce=True, style={"width": "100%"}),
                    ], className="mb-3")
                )
            elif current_type == "dual_axis_line":
                plot_config_controls.append(
                    html.Div([
                        html.Label(f"Plot {i+1} Left Y Column"),
                        dcc.Dropdown(
                            id={'type': 'left-y', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2"
                        ),
                        html.Label(f"Plot {i+1} Right Y Column"),
                        dcc.Dropdown(
                            id={'type': 'right-y', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2"
                        ),
                    ], className="mb-3")
                )
            elif current_type == "efficient_frontier_time":
                plot_config_controls.append(
                    html.Div([
                        html.Label(f"Plot {i+1} Area Columns (allocations)", title="Select columns for stacked areas."),
                        dcc.Dropdown(
                            id={'type': 'areas', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            multi=True,
                            className="mb-2"
                        ),
                        html.Label(f"Plot {i+1} X Column (time)"),
                        dcc.Dropdown(
                            id={'type': 'xcol', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2"
                        ),
                    ], className="mb-3")
                )
            elif current_type == "regression_scatter":
                plot_config_controls.append(
                    html.Div([
                        html.Label(f"Plot {i+1} X Column"),
                        dcc.Dropdown(
                            id={'type': 'xcol', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2"
                        ),
                        html.Label(f"Plot {i+1} Y Column"),
                        dcc.Dropdown(
                            id={'type': 'ycol', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2"
                        ),
                        dbc.Checkbox(id={'type': 'reg-line', 'index': i}, label="Show Regression Line", className="mb-1"),
                        dbc.Checkbox(id={'type': 'std-err', 'index': i}, label="Show Std Error Band", className="mb-1"),
                        dbc.Checkbox(id={'type': 'highlight-latest', 'index': i}, label="Highlight Latest Observation", className="mb-1"),
                    ], className="mb-3")
                )
            elif current_type == "distribution":
                plot_config_controls.append(
                    html.Div([
                        html.Label(f"Plot {i+1} Columns (KDE)", title="Select columns for KDE plot."),
                        dcc.Dropdown(
                            id={'type': 'plot-config', 'index': i},
                            options=[{"label": c, "value": c} for c in df.columns],
                            multi=True,
                            maxHeight=200,
                            className="mb-2"
                        ),
                        dbc.Checkbox(id={'type': 'show-latest', 'index': i}, label="Show Latest Marker", className="mb-1"),
                        dbc.Checkbox(id={'type': 'show-median', 'index': i}, label="Show Median Line", className="mb-1"),
                        html.Label("Quantiles (comma-separated, e.g. 0.25,0.75)", className="mt-1"),
                        dcc.Input(id={'type': 'quantiles', 'index': i}, type="text", placeholder="", className="mb-2", debounce=True, style={"width": "100%"}),
                    ], className="mb-3")
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
        subplot_titles = [safe_get(plot_titles, i, f"Plot {i+1}") for i in range(n_subplots)]
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)],
            horizontal_spacing=0.1,  # tighter horizontal spacing
            vertical_spacing=0.1     # tighter vertical spacing
        )
        for i in range(n_subplots):
            plot_type = safe_get(plot_types, i, "line")
            plot_title = safe_get(plot_titles, i, f"Plot {i+1}")
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
                )
                subfig = plot_line_chart(df, config)
                for trace in subfig.data:
                    fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)
            elif plot_type == "dual_axis_line":
                left = safe_get(left_y, i, None)
                right = safe_get(right_y, i, None)
                from xsam.charts.plot_types.dual_axis_line import DualAxisLineChartConfig, plot_dual_axis_line_chart
                config = DualAxisLineChartConfig(
                    left_y_column=left,
                    right_y_column=right,
                    title=plot_title,
                )
                if left or right:
                    subfig = plot_dual_axis_line_chart(df, config)
                    for trace in subfig.data:
                        is_right = hasattr(trace, 'yaxis') and getattr(trace, 'yaxis', None) == 'y2'
                        fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1, secondary_y=is_right)
            elif plot_type == "efficient_frontier_time":
                area_cols = safe_get(areas, i, [])
                x_col = safe_get(xcol, i, None)
                if area_cols:
                    from xsam.charts.plot_types.efficient_frontier_time import EfficientFrontierTimeChartConfig, plot_efficient_frontier_time_chart
                    config = EfficientFrontierTimeChartConfig(
                        area_columns=area_cols,
                        x_column=x_col or (df.columns[0] if len(df.columns) > 0 else None),
                        title=plot_title,
                    )
                    subfig = plot_efficient_frontier_time_chart(df, config)
                    for trace in subfig.data:
                        fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)
            elif plot_type == "regression_scatter":
                x = safe_get(xcol, i, None)
                y = safe_get(ycol, i, None)
                reg_line_i = bool(safe_get(reg_line, i, False))
                std_err_i = bool(safe_get(std_err, i, False))
                highlight_latest_i = bool(safe_get(highlight_latest, i, False))
                if x and y:
                    from xsam.charts.plot_types.regression_scatter import RegressionScatterConfig, plot_regression_scatter
                    config = RegressionScatterConfig(
                        x_column=x,
                        y_column=y,
                        title=plot_title,
                        regression_line=reg_line_i,
                        show_std_err=std_err_i,
                        highlight_latest=highlight_latest_i,
                    )
                    subfig = plot_regression_scatter(df, config)
                    for trace in subfig.data:
                        fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)
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
                )
                subfig = plot_distribution_chart(df, config)
                for trace in subfig.data:
                    fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)
            # Add a dummy invisible trace as a divider in the legend, except after the last plot
            if i < n_subplots - 1:
                fig.add_trace({
                    'type': 'scatter',
                    'x': [None],
                    'y': [None],
                    'mode': 'lines',
                    'name': ' ',
                    'showlegend': True,
                    'line': {'color': 'rgba(0,0,0,0)'},
                    'hoverinfo': 'skip',
                }, row=1, col=1)
        fig.update_layout(height=400 * rows, showlegend=True, margin=dict(t=80, l=5, r=5, b=5), title={
            'text': figure_title or "",
            'font': {'size': 28}  # Increased font size
        })
        return [dcc.Graph(figure=fig, config={"responsive": True}, style={"height": f"{400*rows}px"})]

    @app.callback(
        Output("layout", "value"),
        Output("figure-title", "value"),
        Output({'type': 'plot-type', 'index': ALL}, 'value'),
        Output({'type': 'plot-config', 'index': ALL}, 'value'),
        Output({'type': 'show-latest', 'index': ALL}, 'value'),
        Output({'type': 'show-median', 'index': ALL}, 'value'),
        Output({'type': 'quantiles', 'index': ALL}, 'value'),
        Output("feedback", "children"),
        Output("download", "data"),
        Output("loaded-trigger", "data"),
        Input("save", "n_clicks"),
        Input("upload", "contents"),
        State("layout", "value"),
        State("figure-title", "value"),
        State({'type': 'plot-type', 'index': ALL}, 'value'),
        State({'type': 'plot-config', 'index': ALL}, 'value'),
        State({'type': 'show-latest', 'index': ALL}, 'value'),
        State({'type': 'show-median', 'index': ALL}, 'value'),
        State({'type': 'quantiles', 'index': ALL}, 'value'),
        prevent_initial_call=True,
    )
    def handle_chart_config(
        save_clicks, load_contents,
        layout, figure_title, plot_types, plot_configs, show_latest, show_median, quantiles
    ):
        import dash
        import base64
        import json
        from xsam.charts.config import ChartConfig, PlotConfig

        ctx = dash.callback_context
        triggered = ctx.triggered_id

        outputs = [
            dash.no_update,  # layout.value
            dash.no_update,  # figure-title.value
            [dash.no_update] * len(plot_types),      # plot-type (ALL)
            [dash.no_update] * len(plot_configs),    # plot-config (ALL)
            [dash.no_update] * len(show_latest),     # show-latest (ALL)
            [dash.no_update] * len(show_median),     # show-median (ALL)
            [dash.no_update] * len(quantiles),       # quantiles (ALL)
            dash.no_update,  # feedback.children
            dash.no_update,  # download.data
            dash.no_update,  # loaded-trigger.data
        ]

        if triggered == "save" and save_clicks:
            plots = []
            for i, plot_type in enumerate(plot_types):
                config = plot_configs[i] if i < len(plot_configs) else None
                if plot_type == "line":
                    plots.append(PlotConfig(
                        plot_type="line",
                        config={
                            "y_columns": config or [],
                            "show_latest": bool(show_latest[i]) if i < len(show_latest) else False,
                            "show_median": bool(show_median[i]) if i < len(show_median) else False,
                            "quantiles": [float(q.strip()) for q in quantiles[i].split(",") if q.strip()] if quantiles and i < len(quantiles) and quantiles[i] else [],
                        }
                    ))
                elif plot_type == "dual_axis_line":
                    left = safe_get(locals().get('left_y', []), i, None)
                    right = safe_get(locals().get('right_y', []), i, None)
                    plots.append(PlotConfig(
                        plot_type="dual_axis_line",
                        config={
                            "left_y_column": left,
                            "right_y_column": right,
                        }
                    ))
                elif plot_type == "efficient_frontier_time":
                    area_cols = safe_get(locals().get('areas', []), i, [])
                    x_col = safe_get(locals().get('xcol', []), i, None)
                    plots.append(PlotConfig(
                        plot_type="efficient_frontier_time",
                        config={
                            "area_columns": area_cols or [],
                            "x_column": x_col,
                        }
                    ))
                elif plot_type == "regression_scatter":
                    x = safe_get(locals().get('xcol', []), i, None)
                    y = safe_get(locals().get('ycol', []), i, None)
                    reg_line_i = bool(safe_get(locals().get('reg_line', []), i, False))
                    std_err_i = bool(safe_get(locals().get('std_err', []), i, False))
                    highlight_latest_i = bool(safe_get(locals().get('highlight_latest', []), i, False))
                    plots.append(PlotConfig(
                        plot_type="regression_scatter",
                        config={
                            "x_column": x,
                            "y_column": y,
                            "regression_line": reg_line_i,
                            "show_std_err": std_err_i,
                            "highlight_latest": highlight_latest_i,
                        }
                    ))
                elif plot_type == "distribution":
                    plots.append(PlotConfig(
                        plot_type="distribution",
                        config={
                            "columns": config or [],
                            "show_latest": bool(show_latest[i]) if i < len(show_latest) else False,
                            "show_median": bool(show_median[i]) if i < len(show_median) else False,
                            "quantiles": [float(q.strip()) for q in quantiles[i].split(",") if q.strip()] if quantiles and i < len(quantiles) and quantiles[i] else [],
                        }
                    ))
            chart_config = ChartConfig(
                plots=plots,
                layout=layout,
                figure_title=figure_title,
                shared_legend=False,
            )
            chart_config_json = json.dumps(chart_config.to_dict(), indent=2)
            outputs[7] = dbc.Alert("Chart config saved!", color="success", dismissable=True, duration=3000)
            outputs[8] = dict(content=chart_config_json, filename="xsam_chart_config.json")
            outputs[9] = str(time.time())  # update trigger
            return outputs

        if triggered == "upload" and load_contents:
            content_type, content_string = load_contents.split(',')
            decoded = base64.b64decode(content_string)
            chart_config_dict = json.loads(decoded.decode("utf-8"))
            pattern_outputs = load_chart_config(chart_config_dict, plot_types, plot_configs, show_latest, show_median, quantiles)
            return (
                dash.no_update,  # layout.value
                dash.no_update,  # figure-title.value
                pattern_outputs[0],  # plot-types
                pattern_outputs[1],  # plot-configs
                pattern_outputs[2],  # show-latest
                pattern_outputs[3],  # show-median
                pattern_outputs[4],  # quantiles
                pattern_outputs[5],  # feedback
                dash.no_update,      # download.data
                dash.no_update,      # loaded-trigger.data
            )

        return outputs

    def load_chart_config(chart_config_dict, plot_types, plot_configs, show_latest, show_median, quantiles,
                         left_y=None, right_y=None, areas=None, xcol=None, ycol=None, reg_line=None, std_err=None, highlight_latest=None):
        import dash
        from xsam.charts.config import ChartConfig
        if not chart_config_dict:
            return [dash.no_update]*5 + [dash.no_update]
        chart_config = ChartConfig.from_dict(chart_config_dict)
        plot_types_arr = [p.plot_type or "line" for p in chart_config.plots]
        plot_configs_arr = []
        show_latest_arr = []
        show_median_arr = []
        quantiles_arr = []
        left_y_arr = []
        right_y_arr = []
        areas_arr = []
        xcol_arr = []
        ycol_arr = []
        reg_line_arr = []
        std_err_arr = []
        highlight_latest_arr = []
        for p in chart_config.plots:
            if p.plot_type == "line":
                plot_configs_arr.append(p.config.get("y_columns", []))
                show_latest_arr.append(bool(p.config.get("show_latest", False)))
                show_median_arr.append(bool(p.config.get("show_median", False)))
                quantiles_arr.append(','.join(str(q) for q in p.config.get("quantiles", [])))
                left_y_arr.append(None)
                right_y_arr.append(None)
                areas_arr.append(None)
                xcol_arr.append(None)
                ycol_arr.append(None)
                reg_line_arr.append(None)
                std_err_arr.append(None)
                highlight_latest_arr.append(None)
            elif p.plot_type == "dual_axis_line":
                plot_configs_arr.append(None)
                show_latest_arr.append(None)
                show_median_arr.append(None)
                quantiles_arr.append(None)
                left_y_arr.append(p.config.get("left_y_column"))
                right_y_arr.append(p.config.get("right_y_column"))
                areas_arr.append(None)
                xcol_arr.append(None)
                ycol_arr.append(None)
                reg_line_arr.append(None)
                std_err_arr.append(None)
                highlight_latest_arr.append(None)
            elif p.plot_type == "efficient_frontier_time":
                plot_configs_arr.append(None)
                show_latest_arr.append(None)
                show_median_arr.append(None)
                quantiles_arr.append(None)
                left_y_arr.append(None)
                right_y_arr.append(None)
                areas_arr.append(p.config.get("area_columns", []))
                xcol_arr.append(p.config.get("x_column"))
                ycol_arr.append(None)
                reg_line_arr.append(None)
                std_err_arr.append(None)
                highlight_latest_arr.append(None)
            elif p.plot_type == "regression_scatter":
                plot_configs_arr.append(None)
                show_latest_arr.append(None)
                show_median_arr.append(None)
                quantiles_arr.append(None)
                left_y_arr.append(None)
                right_y_arr.append(None)
                areas_arr.append(None)
                xcol_arr.append(p.config.get("x_column"))
                ycol_arr.append(p.config.get("y_column"))
                reg_line_arr.append(p.config.get("regression_line", False))
                std_err_arr.append(p.config.get("show_std_err", False))
                highlight_latest_arr.append(p.config.get("highlight_latest", False))
            elif p.plot_type == "distribution":
                plot_configs_arr.append(p.config.get("columns", []))
                show_latest_arr.append(bool(p.config.get("show_latest", False)))
                show_median_arr.append(bool(p.config.get("show_median", False)))
                quantiles_arr.append(','.join(str(q) for q in p.config.get("quantiles", [])))
                left_y_arr.append(None)
                right_y_arr.append(None)
                areas_arr.append(None)
                xcol_arr.append(None)
                ycol_arr.append(None)
                reg_line_arr.append(None)
                std_err_arr.append(None)
                highlight_latest_arr.append(None)
        n_current = len(plot_types)
        return (
            pad_or_truncate(plot_types_arr, n_current, "line"),
            pad_or_truncate(plot_configs_arr, n_current, []),
            pad_or_truncate(show_latest_arr, n_current, False),
            pad_or_truncate(show_median_arr, n_current, False),
            pad_or_truncate(quantiles_arr, n_current, ""),
            dbc.Alert("Chart config loaded!", color="info", dismissable=True, duration=3000),
        )

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
