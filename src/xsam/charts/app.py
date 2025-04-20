"""
Dash app UI for xsam charts module.

Implements a responsive, best-practice dashboard for arranging and visualizing charts.

To use a programmatically provided DataFrame, call run_dash_app(df: pd.DataFrame).
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
from dash.dependencies import ALL
from io import StringIO
import time

# Placeholder for plot_types and arrangement imports
# from .plot_types import ...
# from .arrangement import ArrangementConfig, PlotConfig

external_stylesheets = [dbc.themes.LUX]

def run_dash_app(df: pd.DataFrame | None = None) -> None:
    """
    Run the Dash app. If a DataFrame is provided, it will be used as the working data and the upload UI will be hidden.

    Args:
        df (pd.DataFrame | None): DataFrame to use for plotting. If None, user must upload data.
    """
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.title = "XSAM Charts"

    # Store the DataFrame in dcc.Store for callbacks
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("XSAM Charts", className="mb-4 mt-2"),
                dcc.Store(id="stored-data", data=df.to_json(date_format="iso") if df is not None else None),
                html.Div([
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div([
                            "Drag and Drop or ", html.A("Select a CSV File")
                        ]),
                        style={
                            "width": "100%", "height": "60px", "lineHeight": "60px",
                            "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                            "textAlign": "center", "marginBottom": "20px"
                        },
                        multiple=False
                    ),
                    html.Div(id="data-upload-feedback", className="mb-2"),
                ], style={"display": "none" if df is not None else "block"}),
                html.Hr(),
                html.Label("Arrangement Layout", className="mt-2"),
                dcc.Dropdown(
                    id="arrangement-layout",
                    options=[
                        {"label": l, "value": l} for l in ["1x1", "1x2", "2x1", "2x2"]
                    ],
                    value="1x1",
                    clearable=False,
                    className="mb-3"
                ),
                html.Label("Figure Title"),
                dcc.Input(id="figure-title", type="text", value="", className="mb-3", style={"width": "100%"}),
                html.Div(id="plot-type-controls"),
                html.Div(id="plot-config-controls"),
                html.Button("Save Arrangement", id="save-arrangement", className="mt-3 btn btn-primary"),
                dcc.Upload(
                    id="load-arrangement",
                    children=html.Button("Load Arrangement", className="btn btn-secondary mt-2"),
                    multiple=False
                ),
                html.Div(id="arrangement-feedback", className="mt-2"),
                dcc.Download(id="download-arrangement"),
                dcc.Store(id="arrangement-loaded-trigger"),
            ], width=3, style={"backgroundColor": "#f8f9fa", "padding": "24px", "minHeight": "100vh"}),
            dbc.Col([
                dcc.Loading(
                    id="loading-charts",
                    type="circle",
                    children=html.Div(id="charts-area")
                )
            ], width=9)
        ], className="gx-4"),
    ], fluid=True)

    @app.callback(
        Output("plot-type-controls", "children"),
        Output("plot-config-controls", "children"),
        Input("stored-data", "data"),
        Input("arrangement-layout", "value"),
        Input({'type': 'plot-type', 'index': ALL}, 'value'),
    )
    def update_plot_controls(data_json: str | None, layout: str, plot_types: list[str | None]) -> tuple[list, list]:
        """
        Dynamically generate plot type and config controls based on the DataFrame, layout, and selected plot types.
        """
        if not data_json:
            return [], []
        df = pd.read_json(StringIO(data_json))
        n_plots = {"1x1": 1, "1x2": 2, "2x1": 2, "2x2": 4}[layout]
        plot_type_controls = []
        plot_config_controls = []
        for i in range(n_plots):
            plot_type_id = {'type': 'plot-type', 'index': i}
            plot_config_id = {'type': 'plot-config', 'index': i}
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
                        html.Label(f"Plot {i+1} Y Columns", title="Select up to 5 columns for Y axis."),
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
                            id=f"left-y-{i}",
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2"
                        ),
                        html.Label(f"Plot {i+1} Right Y Column"),
                        dcc.Dropdown(
                            id=f"right-y-{i}",
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
                            id=f"areas-{i}",
                            options=[{"label": c, "value": c} for c in df.columns],
                            multi=True,
                            className="mb-2"
                        ),
                        html.Label(f"Plot {i+1} X Column (time)"),
                        dcc.Dropdown(
                            id=f"xcol-{i}",
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
                            id=f"xcol-{i}",
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2"
                        ),
                        html.Label(f"Plot {i+1} Y Column"),
                        dcc.Dropdown(
                            id=f"ycol-{i}",
                            options=[{"label": c, "value": c} for c in df.columns],
                            className="mb-2"
                        ),
                        dbc.Checkbox(id=f"reg-line-{i}", label="Show Regression Line", className="mb-1"),
                        dbc.Checkbox(id=f"std-err-{i}", label="Show Std Error Band", className="mb-1"),
                        dbc.Checkbox(id=f"highlight-latest-{i}", label="Highlight Latest Observation", className="mb-1"),
                    ], className="mb-3")
                )
            elif current_type == "distribution":
                plot_config_controls.append(
                    html.Div([
                        html.Label(f"Plot {i+1} Columns (up to 3)", title="Select up to 3 columns for KDE plots."),
                        dcc.Dropdown(
                            id=f"kde-cols-{i}",
                            options=[{"label": c, "value": c} for c in df.columns],
                            multi=True,
                            className="mb-2"
                        ),
                        html.Label("Quantiles (comma-separated, e.g. 0.25,0.75) for each column"),
                        dcc.Input(id=f"kde-quantiles-{i}", type="text", placeholder="", className="mb-2", debounce=True, style={"width": "100%"}),
                    ], className="mb-3")
                )
        return plot_type_controls, plot_config_controls

    @app.callback(
        Output("charts-area", "children"),
        Input("stored-data", "data"),
        Input("arrangement-layout", "value"),
        Input("figure-title", "value"),
        Input({'type': 'plot-type', 'index': ALL}, 'value'),
        Input({'type': 'plot-config', 'index': ALL}, 'value'),
        Input({'type': 'show-latest', 'index': ALL}, 'value'),
        Input({'type': 'show-median', 'index': ALL}, 'value'),
        Input({'type': 'quantiles', 'index': ALL}, 'value'),
        Input("arrangement-loaded-trigger", "data"),
        prevent_initial_call=True,
    )
    def render_charts(
        data_json: str | None,
        layout: str,
        figure_title: str | None,
        plot_types: list[str | None],
        plot_configs: list[list[str] | None],
        show_latest: list[bool | None],
        show_median: list[bool | None],
        quantiles: list[str | None],
        _trigger: str | None,
    ) -> list:
        from plotly.subplots import make_subplots
        import plotly.graph_objs as go
        if not data_json:
            return [html.Div("No data uploaded.")]
        df = pd.read_json(StringIO(data_json))
        layout_map = {
            "1x1": (1, 1),
            "1x2": (1, 2),
            "2x1": (2, 1),
            "2x2": (2, 2),
        }
        rows, cols = layout_map.get(layout, (1, 1))
        n_subplots = rows * cols
        def safe_get(lst, idx, default):
            return lst[idx] if idx < len(lst) and lst[idx] is not None else default
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Plot {i+1}" for i in range(n_subplots)])
        for i in range(n_subplots):
            plot_type = safe_get(plot_types, i, "line")
            y_columns = safe_get(plot_configs, i, [])
            show_latest_i = safe_get(show_latest, i, False)
            show_median_i = safe_get(show_median, i, False)
            quantiles_i = safe_get(quantiles, i, "")
            if plot_type == "line" and y_columns:
                from xsam.charts.plot_types.line import LineChartConfig, plot_line_chart
                quantiles_list = None
                if quantiles_i:
                    try:
                        quantiles_list = [float(q.strip()) for q in quantiles_i.split(",") if q.strip()]
                    except Exception:
                        quantiles_list = None
                config = LineChartConfig(
                    y_columns=y_columns,
                    show_latest=bool(show_latest_i),
                    show_median=bool(show_median_i),
                    quantiles=quantiles_list,
                )
                subfig = plot_line_chart(df, config)
                for trace in subfig.data:
                    fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)
        fig.update_layout(height=400 * rows, showlegend=True, margin=dict(t=40, l=10, r=10, b=10), title=figure_title or "")
        return [dcc.Graph(figure=fig, config={"responsive": True}, style={"height": f"{400*rows}px"})]

    @app.callback(
        Output("arrangement-layout", "value"),
        Output("figure-title", "value"),
        Output({'type': 'plot-type', 'index': ALL}, 'value'),
        Output({'type': 'plot-config', 'index': ALL}, 'value'),
        Output({'type': 'show-latest', 'index': ALL}, 'value'),
        Output({'type': 'show-median', 'index': ALL}, 'value'),
        Output({'type': 'quantiles', 'index': ALL}, 'value'),
        Output("arrangement-feedback", "children"),
        Output("download-arrangement", "data"),
        Output("arrangement-loaded-trigger", "data"),
        Input("save-arrangement", "n_clicks"),
        Input("load-arrangement", "contents"),
        State("arrangement-layout", "value"),
        State("figure-title", "value"),
        State({'type': 'plot-type', 'index': ALL}, 'value'),
        State({'type': 'plot-config', 'index': ALL}, 'value'),
        State({'type': 'show-latest', 'index': ALL}, 'value'),
        State({'type': 'show-median', 'index': ALL}, 'value'),
        State({'type': 'quantiles', 'index': ALL}, 'value'),
        prevent_initial_call=True,
    )
    def handle_arrangement(
        save_clicks, load_contents,
        layout, figure_title, plot_types, plot_configs, show_latest, show_median, quantiles
    ):
        import dash
        import base64
        import json
        from xsam.charts.arrangement import ArrangementConfig, PlotConfig

        ctx = dash.callback_context
        triggered = ctx.triggered_id

        outputs = [
            dash.no_update,  # arrangement-layout.value
            dash.no_update,  # figure-title.value
            [dash.no_update] * len(plot_types),      # plot-type (ALL)
            [dash.no_update] * len(plot_configs),    # plot-config (ALL)
            [dash.no_update] * len(show_latest),     # show-latest (ALL)
            [dash.no_update] * len(show_median),     # show-median (ALL)
            [dash.no_update] * len(quantiles),       # quantiles (ALL)
            dash.no_update,  # arrangement-feedback.children
            dash.no_update,  # download-arrangement.data
            dash.no_update,  # arrangement-loaded-trigger.data
        ]

        def pad_or_truncate(lst, target_len):
            if (len(lst) < target_len):
                return lst + [dash.no_update] * (target_len - len(lst))
            else:
                return lst[:target_len]

        if triggered == "save-arrangement" and save_clicks:
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
                # ...extend for other plot types as needed...
            arrangement = ArrangementConfig(
                plots=plots,
                layout=layout,
                figure_title=figure_title,
                shared_legend=False,
            )
            arrangement_json = json.dumps(arrangement.to_dict(), indent=2)
            outputs[7] = dbc.Alert("Arrangement saved!", color="success", dismissable=True, duration=3000)
            outputs[8] = dict(content=arrangement_json, filename="xsam_arrangement.json")
            outputs[9] = str(time.time())  # update trigger
            return outputs

        if triggered == "load-arrangement" and load_contents:
            content_type, content_string = load_contents.split(',')
            decoded = base64.b64decode(content_string)
            arrangement_dict = json.loads(decoded.decode("utf-8"))
            pattern_outputs = load_arrangement_step2(arrangement_dict, plot_types, plot_configs, show_latest, show_median, quantiles)
            return (
                dash.no_update,  # arrangement-layout.value
                dash.no_update,  # figure-title.value
                pattern_outputs[0],  # plot-types
                pattern_outputs[1],  # plot-configs
                pattern_outputs[2],  # show-latest
                pattern_outputs[3],  # show-median
                pattern_outputs[4],  # quantiles
                pattern_outputs[5],  # feedback
                dash.no_update,      # download-arrangement.data
                dash.no_update,      # arrangement-loaded-trigger.data
            )

        return outputs

    def load_arrangement_step2(arrangement_dict, plot_types, plot_configs, show_latest, show_median, quantiles):
        import dash
        from xsam.charts.arrangement import ArrangementConfig
        if not arrangement_dict:
            return [dash.no_update]*5 + [dash.no_update]
        arrangement = ArrangementConfig.from_dict(arrangement_dict)
        plot_types_arr = [p.plot_type or "line" for p in arrangement.plots]
        plot_configs_arr = [p.config.get("y_columns", []) if p.config.get("y_columns", []) is not None else [] for p in arrangement.plots]
        show_latest_arr = [bool(p.config.get("show_latest", False)) for p in arrangement.plots]
        show_median_arr = [bool(p.config.get("show_median", False)) for p in arrangement.plots]
        quantiles_arr = [','.join(str(q) for q in p.config.get("quantiles", [])) if p.config.get("quantiles", []) is not None else "" for p in arrangement.plots]
        n_current = len(plot_types)
        def pad_or_truncate(lst, target_len, default):
            # Always return a value of the correct type, never None or undefined
            if len(lst) < target_len:
                return lst + [default] * (target_len - len(lst))
            elif len(lst) > target_len:
                return lst[:target_len]
            return lst
        return (
            pad_or_truncate(plot_types_arr, n_current, "line"),
            pad_or_truncate(plot_configs_arr, n_current, []),
            pad_or_truncate(show_latest_arr, n_current, False),
            pad_or_truncate(show_median_arr, n_current, False),
            pad_or_truncate(quantiles_arr, n_current, ""),
            dbc.Alert("Arrangement loaded!", color="info", dismissable=True, duration=3000),
        )

    app.run(debug=True)

# For CLI usage: fallback to upload if run directly
if __name__ == "__main__":
    # Generate a sample DataFrame for testing
    data = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
        "C": [2, 3, 4, 5, 6],
        "D": [6, 5, 4, 3, 2]
    })
    # Run the Dash app with the sample DataFrame  
    run_dash_app(data)
