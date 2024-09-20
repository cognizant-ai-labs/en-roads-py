"""
Component in charge of filtering out prescriptors by metric.
"""
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.classes import JUMBOTRON, CONTAINER, DESC_TEXT, HEADER
from app.utils import filter_metrics_json


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
        TODO: We need to stop hard-coding their names and adjustments.
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
                allowCross=False,
                disabled=True
            )
            sliders.append(slider)

        names_map = dict(zip(self.metrics, ["Temperature in 2100",
                                            "Highest cost of energy",
                                            "Government spending",
                                            "Reduction in energy demand"]))

        div = html.Div(
            children=[
                html.Div(
                    className="d-flex flex-row mb-2",
                    children=[
                        html.Label(names_map[self.metrics[i]], className="w-25"),  # w-25 and flex-grow-1 ensures they line up
                        html.Div(sliders[i], className="flex-grow-1")
                    ]
                )
                for i in range(len(self.metrics))
            ]
        )
        return div

    def plot_parallel_coordinates_line(self,
                                       metrics_json: dict[str, list],
                                       metric_ranges: list[tuple[float, float]]) -> go.Figure:
        """
        Plots a parallel coordinates plot of the prescriptor metrics.
        Starts by plotting "other" so that it's the bottom of the z axis.
        Then plots selected candidates in color.
        Finally plots the baseline on top.
        """
        fig = go.Figure()

        normalized_df = filter_metrics_json(metrics_json, metric_ranges, normalize=True)

        cand_idxs = list(normalized_df.index)[:-1]  # Leave out the baseline
        n_special_cands = min(10, len(cand_idxs))

        showlegend = True
        # If "other" is in the cand_idxs, plot all other candidates in lightgray
        for cand_idx in cand_idxs[n_special_cands:]:
            cand_metrics = normalized_df.loc[cand_idx].values
            fig.add_trace(go.Scatter(
                x=normalized_df.columns,
                y=cand_metrics,
                mode='lines',
                legendgroup="other",
                name="other",
                line=dict(color="lightgray"),
                showlegend=showlegend
            ))
            showlegend = False

        # Plot selected candidates besides baseline so it can be on top
        for color_idx, cand_idx in enumerate(cand_idxs[:n_special_cands]):
            cand_metrics = normalized_df.loc[cand_idx].values
            fig.add_trace(go.Scatter(
                x=normalized_df.columns,
                y=cand_metrics,
                mode='lines',
                name=str(cand_idx),
                line=dict(color=px.colors.qualitative.Plotly[color_idx])
            ))

        baseline_metrics = normalized_df.iloc[-1]
        fig.add_trace(go.Scatter(
            x=normalized_df.columns,
            y=baseline_metrics.values,
            mode='lines',
            name="baseline",
            line=dict(color="black")
        ))

        for i in range(len(normalized_df.columns)):
            fig.add_vline(x=i, line_color="black")

        full_metrics_df = pd.DataFrame(metrics_json)
        normalized_full = (full_metrics_df - full_metrics_df.mean()) / (full_metrics_df.std() + 1e-10)
        fig.update_layout(
            yaxis_range=[normalized_full.min().min(), normalized_full.max().max()],
            title={
                'text': "Normalized Policy Metrics",
                'x': 0.5,  # Center the title
                'xanchor': 'center',  # Anchor it at the center
                'yanchor': 'top'  # Optionally keep it anchored to the top
            }
        )

        return fig

    def create_filter_div(self):
        """
        Creates div showing sliders to choose the range of metric values we want the prescriptors to have.
        TODO: Currently the slider tooltips show even while loading which is a bit of an eyesore.
        """
        div = html.Div(
            className=JUMBOTRON,
            children=[
                dbc.Container(
                    fluid=True,
                    className=CONTAINER,
                    children=[
                        html.H2("Filter Policies by Desired Behavior", className=HEADER),
                        html.P("One hundred AI models are trained to create energy policies that make different trade \
                               offs in metrics. Use the sliders below to filter the AI generated policies \
                               that produce desired behavior. See the results of the filtering in the below sections.",
                               className=DESC_TEXT),
                        html.Div(
                            dcc.Loading(
                                type="circle",
                                target_components={"metrics-store": "*"},
                                children=[
                                    self.create_metric_sliders(),
                                    dcc.Store(id="metrics-store")
                                ],
                            ),
                            className="w-100"
                        ),
                        html.Div(
                            className="d-flex flex-row mb-2",
                            children=[
                                dbc.Button(
                                    "0 Policies Selected",
                                    id="cand-counter",
                                    disabled=True,
                                    outline=True,
                                    className="me-1",
                                    style={"width": "200px"}  # TODO: We hard-code the width here because of text size
                                ),
                                dbc.Button("Reset Filters", id="reset-button", disabled=True)
                            ]
                        ),
                        html.Div(
                            dbc.Accordion(
                                dbc.AccordionItem(dcc.Graph(id="parcoords-figure"), title="View Parallel Coordinates"),
                                start_collapsed=True,
                            ),
                            className="w-100"
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
            [Output(f"{metric_id}-slider", "disabled") for metric_id in self.metric_ids],
            Output("reset-button", "disabled"),
            Input("metrics-store", "data"),
            Input("reset-button", "n_clicks"),
            prevent_initial_call=True
        )
        def update_filter_sliders(metrics_jsonl: list[dict[str, list]], _) -> list:
            """
            Update the filter slider min/max/value/marks based on the incoming metrics data. The output of this function
            is a list of the updated parameters for each slider concatenated.
            This also happens whenever we click the reset button.
            The reset button starts disabled but once the sliders are updated for the first time it becomes enabled.
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
            total_output.extend([False] * len(self.metric_ids))  # Enable all sliders
            total_output.append(False)  # Enable reset button
            return total_output

        @app.callback(
            Output("parcoords-figure", "figure"),
            State("metrics-store", "data"),
            [Input(f"{metric_id}-slider", "value") for metric_id in self.metric_ids],
            prevent_initial_call=True
        )
        def filter_parcoords_figure(metrics_json: dict[str, list], *metric_ranges) -> go.Figure:
            """
            Filters parallel coordinates figure based on the metric ranges from the sliders.
            """
            return self.plot_parallel_coordinates_line(metrics_json, metric_ranges)

        @app.callback(
            Output("cand-counter", "children"),
            State("metrics-store", "data"),
            [Input(f"{metric_id}-slider", "value") for metric_id in self.metric_ids],
            prevent_initial_call=True
        )
        def count_selected_cands(metrics_json: dict[str, list], *metric_ranges) -> str:
            """
            Counts the number of selected candidates based on the metric ranges from the sliders.
            """
            metrics_df = filter_metrics_json(metrics_json, metric_ranges)
            return f"{len(metrics_df) - 1} Policies Selected"
