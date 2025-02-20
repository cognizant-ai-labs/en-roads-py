"""
Component in charge of filtering out prescriptors by metric.
"""
import json

from dash import html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd

from app.components.component import Component
from app.utils import filter_metrics_json


class FilterComponent(Component):
    """
    Component in charge of filtering out prescriptors by metric specific to each context.
    The component stores the metrics to filter with.
    It also keeps track of the parameters that need to be updated for each slider.
    """
    def __init__(self, metrics: list[str]):
        self.metrics = list(metrics)

        with open("app/units.json", "r", encoding="utf-8") as f:
            self.units = json.load(f)

        self.names_map = dict(zip(self.metrics, ["Temperature change from 1850",
                                                 "Highest cost of energy",
                                                 "Government spending",
                                                 "Reduction in energy demand"]))

    def create_metric_sliders(self) -> html.Div:
        """
        Creates initial metric sliders and lines them up with their labels.
        TODO: We need to stop hard-coding their names and adjustments.
        TODO: Add a tooltip to the sliders to show their units.
        """
        sliders = []
        for i in range(len(self.metrics)):
            slider = dcc.RangeSlider(
                id={"type": "metric-slider", "index": i},
                min=0,
                max=1,
                value=[0, 1],
                marks={0: f"{0:.2f}", 1: f"{1:.2f}"},
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False,
                disabled=True
            )
            sliders.append(slider)

        div = html.Div(
            children=[
                dbc.Row(
                    className="pb-2",
                    children=[
                        dbc.Col(
                            width=5,
                            children=html.Label(
                                f"{self.names_map[self.metrics[i]]} ({self.units[self.metrics[i]]})",
                            ),
                        ),
                        dbc.Col(
                            children=sliders[i]
                        )
                    ]
                )
                for i in range(len(self.metrics))
            ]
        )
        return div

    def create_div(self) -> html.Div:
        """
        Create div for big demo app. Dynamically generates sliders based on metrics.
        """
        div = dbc.Card(
            className="h-100",
            color="secondary",
            outline=True,
            children=[
                dbc.CardHeader(html.H3("2. Filter Policies"), className="text-center"),
                dbc.CardBody(
                    children=[
                        dbc.Row(
                            dcc.Loading(
                                custom_spinner=html.H2(dbc.Spinner(color="primary")),
                                target_components={"metrics-store": "*"},
                                children=[
                                    self.create_metric_sliders(),
                                    dcc.Store(id="metrics-store")
                                ],
                            )
                        ),
                        dbc.Row(
                            justify="center",
                            children=[
                                dbc.Col(
                                    width=4,
                                    children=[
                                        dbc.Button(
                                            "0 Policies Selected",
                                            className="w-100",
                                            id="cand-counter",
                                            disabled=True,
                                            outline=True
                                        )
                                    ]
                                ),
                                dbc.Col(
                                    width=1,
                                    children=[
                                        dbc.Button(
                                            className="bi bi-arrow-counterclockwise",
                                            id="reset-button",
                                            color="secondary",
                                            disabled=True
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
        return html.Div(div)

    def register_callbacks(self, app):
        """
        Registers callbacks related to the filter sliders.
        """
        @app.callback(
            Output({"type": "metric-slider", "index": ALL}, "min"),
            Output({"type": "metric-slider", "index": ALL}, "max"),
            Output({"type": "metric-slider", "index": ALL}, "value"),
            Output({"type": "metric-slider", "index": ALL}, "marks"),
            Output({"type": "metric-slider", "index": ALL}, "disabled"),
            Output("reset-button", "disabled"),
            Input("metrics-store", "data"),
            Input("reset-button", "n_clicks"),
            prevent_initial_call=True
        )
        def update_filter_sliders(metrics_jsonl: list[dict[str, list]], _) -> list:
            """
            Update the filter slider min/max/value/marks based on the incoming metrics data. The output of this function
            is a list for each parameter for each slider.
            This also happens whenever we click the reset button.
            The reset button starts disabled but once the sliders are updated for the first time it becomes enabled.
            """
            metrics_df = pd.DataFrame(metrics_jsonl)
            total_output = [[], [], [], [], [], False]
            for metric in self.metrics:
                min_val = metrics_df[metric].min()
                max_val = metrics_df[metric].max()
                # We need to round down for the min value and round up for the max value
                min_val_rounded = min_val // 0.01 / 100
                max_val_rounded = max_val + 0.01
                value = [min_val_rounded, max_val_rounded]
                marks = {min_val_rounded: f"{min_val_rounded:.2f}", max_val_rounded: f"{max_val_rounded:.2f}"}

                # Append each output to corresponding output
                total_output[0].append(min_val_rounded)
                total_output[1].append(max_val_rounded)
                total_output[2].append(value)
                total_output[3].append(marks)
                total_output[4].append(False)

            return total_output

        @app.callback(
            Output("cand-counter", "children"),
            State("metrics-store", "data"),
            Input({"type": "metric-slider", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def count_selected_cands(metrics_json: dict[str, list], metric_ranges: list[tuple[float, float]]) -> str:
            """
            Counts the number of selected candidates based on the metric ranges from the sliders.
            """
            metrics_df = filter_metrics_json(metrics_json, metric_ranges)
            return f"{len(metrics_df) - 1} Policies Selected"
