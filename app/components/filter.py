"""
Component in charge of filtering out prescriptors by metric.
"""
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd


class FilterComponent:
    """
    Component in charge of filtering out prescriptors by metric specific to each context.
    The component stores the metrics to filter with as long as their corresponding HTML ids.
    It also keeps track of the parameters that need to be updated for each slider.
    """
    def __init__(self, metrics: list[str]):
        self.metrics = list(metrics)
        self.metric_ids = [metric.replace(" ", "-").replace(".", "_") for metric in self.metrics]
        self.updated_params = ["min", "max", "value", "marks"]

    def create_metric_sliders(self):
        """
        Creates initial metric sliders and lines them up with their labels.
        """
        sliders = []
        for metric in self.metrics:
            col_id = metric.replace(" ", "-").replace(".", "_")
            slider = dcc.RangeSlider(
                id=f"{col_id}-slider",
                min=0,
                max=1,
                value=[0, 1],
                marks={0: f"{0:.2f}", 1: f"{1:.2f}"},
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False
            )
            sliders.append(slider)

        div = html.Div(
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(html.Label(self.metrics[i]), width=4),
                        dbc.Col(sliders[i], width=8)
                    ]
                )
                for i in range(len(self.metrics))
            ]
        )
        return div

    def create_filter_div(self):
        """
        Creates div showing sliders to choose the range of metric values we want the prescriptors to have.
        TODO: Currently the slider tooltips show even while loading which is a bit of an eyesore.
        """
        div = html.Div(
            className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
            children=[
                dbc.Container(
                    fluid=True,
                    className="py-3",
                    children=[
                        html.H2("Filter AI Models by Desired Metric", className="text-center mb-5"),
                        html.P("One hundred AI models are trained to create different energy policies that have a \
                               diverse range of outcomes. Use the sliders below to filter the models that align with a \
                               desired behavior resulting from their automatically generated energy policy. See how \
                               this filtering affects the behavior of the policies in the below sections.",
                               className="text-center"),
                        dcc.Loading(
                            type="circle",
                            target_components={"metrics-store": "*"},
                            children=[
                                self.create_metric_sliders(),
                                dcc.Store(id="metrics-store")
                            ],
                        )
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        """
        Registers callbacks related to the filter sliders.
        """
        @app.callback(
            [Output(f"{metric_id}-slider", param) for metric_id in self.metric_ids for param in self.updated_params],
            Input("metrics-store", "data")
        )
        def update_filter_sliders(metrics_jsonl: list[dict[str, list]]) -> list:
            """
            Update the filter slider min/max/value/marks based on the incoming metrics data. The output of this function
            is a list of the updated parameters for each slider concatenated.
            """
            metrics_df = pd.DataFrame(metrics_jsonl)
            total_output = []
            for metric in self.metrics:
                metric_output = []
                min_val = metrics_df[metric].min()
                max_val = metrics_df[metric].max()
                # We need to round down for the min value and round up for the max value
                min_val_rounded = min_val // 0.01 / 100
                max_val_rounded = max_val + 0.01
                metric_output = [
                    min_val_rounded,
                    max_val_rounded,
                    [min_val_rounded, max_val_rounded],
                    {min_val_rounded: f"{min_val_rounded:.2f}", max_val_rounded: f"{max_val_rounded:.2f}"}
                ]
                total_output.extend(metric_output)

            return total_output
