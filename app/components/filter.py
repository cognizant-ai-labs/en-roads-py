"""
Component in charge of filtering out prescriptors by metric.
"""
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd


class FilterComponent:
    def create_metric_sliders(self, metrics_df: pd.DataFrame):
        sliders = []
        for col in metrics_df:
            col_id = col.replace(" ", "-").replace(".", "_")
            min_val = metrics_df[col].min()
            max_val = metrics_df[col].max()
            marks = {min_val: f"{min_val:.2f}", max_val: f"{max_val:.2f}"}
            slider = dcc.RangeSlider(
                id=f"{col_id}-slider",
                min=min_val,
                max=max_val,
                value=[min_val, max_val],
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False
            )
            sliders.append(slider)
        return sliders

    def create_filter_div(self):
        div = html.Div(
            className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
            children=[
                dbc.Container(
                    fluid=True,
                    className="py-3",
                    children=[
                        html.H2("Filter AI Models by Desired Metric", className="text-center"),
                        dcc.Loading(html.Div(id="filter-sliders"), type="circle", target_components="metrics-store")
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        @app.callback(
            Output("filter-sliders", "children"),
            Input("metrics-store", "data")
        )
        def update_filter_sliders(metrics_jsonl):
            metrics_df = pd.DataFrame(metrics_jsonl)
            return self.create_metric_sliders(metrics_df)
