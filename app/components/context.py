"""
Component displaying the context and handling its related functions.
"""
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ContextComponent():
    """
    Component in charge of displaying the preset contexts in a scatter and the corresponding context sliders.
    Adjusts context sliders to match selected preset and displays a blurb about the selected SSP.
    """
    def __init__(self):

        self.context_cols = ["_long_term_gdp_per_capita_rate",
                             "_near_term_gdp_per_capita_rate",
                             "_transition_time_to_reach_long_term_gdp_per_capita_rate",
                             "_global_population_in_2100"]
        self.varnames = ["Long-Term Economic Growth (GDPPP)",
                         "Near-Term Economic Growth (GDPPP)",
                         "Transition Time (years)",
                         "Population (B)"]

        # Round context df here instead of automatically by Dash so that we know for sure how it's rounding.
        self.context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        for col in self.context_cols:
            row = input_specs[input_specs["varId"] == col].iloc[0]
            step = row["step"]
            decimals = -1 * int(np.log10(step))
            self.context_df[col] = self.context_df[col].round(decimals)

        self.ssp_df = pd.read_csv("app/ssps.csv")

    def create_context_scatter(self):
        """
        Creates scatter plot of context scenarios.
        """
        ar6_df = pd.read_csv("experiments/scenarios/ar6_snapshot_1723566520.csv/ar6_snapshot_1723566520.csv")
        ar6_df = ar6_df.dropna(subset=["Scenario"])
        ar6_df = ar6_df.dropna(axis=1)
        ar6_df = ar6_df[ar6_df["Scenario"].str.contains("Baseline")]
        pop_df = ar6_df[ar6_df["Variable"] == "Population"]
        gdp_df = ar6_df[ar6_df["Variable"] == "GDP|PPP"]

        context_chart_df = pd.DataFrame()
        context_chart_df["scenario"] = ar6_df["Scenario"].unique()
        context_chart_df["scenario"] = context_chart_df["scenario"].str.split("-", expand=True)[0]
        context_chart_df["population"] = pop_df["2100"].values / 1000
        context_chart_df["gdp"] = gdp_df["2100"].values / 1000
        context_chart_df = context_chart_df.sort_values(by="scenario")
        # pylint: disable=unsupported-assignment-operation
        context_chart_df["description"] = ["SSP1: Sustainable Development",
                                           "SSP2: Middle of the Road",
                                           "SSP3: Regional Rivalry",
                                           "SSP4: Inequality",
                                           "SSP5: Fossil-Fueled"]

        fig = go.Figure()

        # pylint: disable=unsubscriptable-object
        fig.add_trace(go.Scatter(
            x=context_chart_df["population"],
            y=context_chart_df["gdp"],
            mode="markers+text",
            text=context_chart_df["scenario"],
            textposition="top center",
            marker=dict(color=px.colors.qualitative.Safe)
        ))

        fig.update_layout(
            title={
                "text": "Select a Pre-Set Context Scenario",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Population in 2100 (B)",
            yaxis_title="GDP in 2100 (T$)",
            showlegend=False,
        )
        return fig

    def create_context_div(self):
        """
        Creates div showing context scatter plot next to context sliders.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        sliders = []
        for i, (context_col, varname) in enumerate(zip(self.context_cols, self.varnames)):
            row = input_specs[input_specs["varId"] == context_col].iloc[0]
            label_slider = dbc.Row([
                # TODO: It's annoying that bootstrap only does 12 columns and 5/7 split doesn't look nice.
                dbc.Col(
                    width=6,
                    children=[html.Label(varname)]
                ),
                dbc.Col(
                    width=6,
                    children=[
                        dcc.Slider(
                            id=f"context-slider-{i}",
                            min=row["minValue"],
                            max=row["maxValue"],
                            step=row["step"],
                            value=row["defaultValue"],
                            tooltip={"placement": "bottom"},
                            marks=None,
                        )
                    ]
                )
            ])
            sliders.append(label_slider)

        # TODO: Fix the widths to be responsive
        sliders_div = html.Div(
            children=sliders
        )

        div = html.Div(
            className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
            children=[
                dbc.Container(
                    fluid=True,
                    className="py-3 d-flex flex-column h-100",
                    children=[
                        dbc.Row(html.H2("Select a Context Scenario to Optimize For", className="text-center mb-5")),
                        dbc.Row(
                            className="mb-2 w-70 text-center mx-auto",
                            children=[html.P("According to the AR6 climate report: 'The five Shared Socioeconomic \
                                             Pathways were designed to span a range of challenges to climate change \
                                             mitigation and adaptation.' Select one of these scenarios by clicking it \
                                             in the scatter plot below. If desired, manually modify the scenario \
                                             with the sliders.")]
                        ),
                        dbc.Row(
                            className="flex-grow-1",
                            children=[
                                dbc.Col(
                                    className="h-100",
                                    children=[dcc.Graph(
                                        id="context-scatter",
                                        figure=self.create_context_scatter()
                                    )],
                                ),
                                dbc.Col(
                                    className="d-flex flex-column h-100",
                                    children=[
                                        sliders_div,
                                        # TODO: Make the box big enough to fit the text
                                        html.Div(
                                            id="ssp-desc",
                                            children=[self.construct_ssp_desc(0)],
                                            className="flex-grow-1 overflow-auto border rounded-3 p-2",
                                            style={"height": "275px"}
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        return div

    def construct_ssp_desc(self, ssp_idx):
        """
        Constructs the description text for the SSP.
        """
        ssp_row = self.ssp_df.iloc[ssp_idx]
        div = html.Div([
            html.H4([
                ssp_row["ssp"] + ": " + ssp_row["description"] + " ",
                html.I(className=f"bi bi-{ssp_row['icon']}")
            ]),
            html.P(ssp_row["text"])
        ])
        return div

    def register_callbacks(self, app):
        """
        Registers app's callbacks.
        """
        @app.callback(
            [Output(f"context-slider-{i}", "value") for i in range(4)],
            Input("context-scatter", "clickData")
        )
        def click_context(click_data):
            """
            Updates context sliders when a context point is clicked.
            TODO: Sometimes this function lags, not sure why.
            """
            if click_data:
                # TODO: This assumes the SSPs in the ssps.csv file are in order which they are
                scenario = int(click_data["points"][0]["pointNumber"])
            else:
                scenario = 0

            scenario = f"SSP{scenario+1}-Baseline"
            row = self.context_df[self.context_df["scenario"] == scenario].iloc[0]
            return [row[self.context_cols[i]] for i in range(4)]

        @app.callback(
            Output("ssp-desc", "children"),
            [Input(f"context-slider-{i}", "value") for i in range(4)],
            prevent_initial_call=True
        )
        def update_ssp_desc(*context_values):
            """
            If we click on a point, show the description of the SSP.
            If there is no click data, that means we updated the sliders, so remove the description.
            """
            match = self.context_df[self.context_cols].eq(context_values).all(axis=1)

            # If a match is found, retrieve the row, otherwise handle the case where it isn't found
            if match.any():
                ssp_idx = match.idxmax()
                return self.construct_ssp_desc(ssp_idx)
            else:
                return html.Div(
                    children=[
                        html.H4("Custom Scenario Selected")
                    ]
                )
