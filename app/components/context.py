"""
Component displaying the context and handling its related functions.
"""
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.classes import JUMBOTRON, CONTAINER, DESC_TEXT, HEADER
from enroadspy import load_input_specs


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
        input_specs = load_input_specs()
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
        context_chart_df["description"] = ["SSP1: Sustainable Development",
                                           "SSP2: Middle of the Road",
                                           "SSP3: Regional Rivalry",
                                           "SSP4: Inequality",
                                           "SSP5: Fossil-Fueled"]

        fig = go.Figure()

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

    def create_context_sliders(self):
        """
        Creates labels aligned with sliders for the context variables.
        """
        input_specs = load_input_specs()
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
                            id={"type": "context-slider", "index": i},
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
        return sliders_div

    def create_ssp_dropdown(self):
        """
        Creates dropdown to select an SSP
        """
        values = [f"SSP{i+1}" for i in range(5)]
        labels = [
            "SSP1: Sustainable Development",
            "SSP2: Middle of the Road",
            "SSP3: Regional Rivalry",
            "SSP4: Inequality",
            "SSP5: Fossil-Fueled"
        ]
        colors = [
            "#1a8955",
            "#688114",
            "#dc3848",
            "#aa6900",
            "#aa6900"
        ]
        options = []
        for value, label, color in zip(values, labels, colors):
            options.append({"label": html.Span([label], style={"color": color}), "value": value})

        dropdown = dcc.Dropdown(
            id="ssp-dropdown",
            options=options,
            placeholder="Select a Scenario"
        )
        return dropdown

    def create_context_div(self):
        """
        Creates div showing context scatter plot next to context sliders.
        """
        div = html.Div(
            className=JUMBOTRON,
            children=[
                dbc.Container(
                    fluid=True,
                    className=CONTAINER,
                    children=[
                        html.H2("Select a Context Scenario to Optimize For", className=HEADER),
                        html.P("According to the AR6 climate report: 'The five Shared Socioeconomic \
                                            Pathways were designed to span a range of challenges to climate change \
                                            mitigation and adaptation'. Select one of these scenarios by clicking it \
                                            in the scatter plot below. If desired, manually modify the scenario \
                                            with the sliders.", className=DESC_TEXT),
                        dbc.Row(
                            className="w-100",
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
                                        self.create_context_sliders(),
                                        # TODO: Make the box big enough to fit the text
                                        html.Div(
                                            id="ssp-desc",
                                            children=[html.H4("Select a Scenario")],
                                            className="flex-grow-1 overflow-auto border rounded-3 p-2",
                                            style={"height": "275px"}
                                        )
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            children=[
                                dbc.Button(
                                    "AI Generate Policies for Scenario",
                                    id="presc-button",
                                    className="me-1 mb-2 w-25",
                                    n_clicks=0
                                ),

                            ]
                        )
                    ]
                )
            ]
        )

        return div

    def create_context_div_big(self):
        """
        Creates big context div for larger demos.
        """
        sliders_div = self.create_context_sliders()
        div = html.Div(
            children=[
                html.H3("1. Select Context Scenario", className="text-center"),
                dbc.Row(
                    className="g-0",
                    justify="center",
                    children=self.create_ssp_dropdown(),
                ),
                dbc.Row(
                    children=[
                        dbc.Col(sliders_div, width=9),
                        dbc.Col(
                            width=3,
                            children=html.Div(
                                id="ssp-desc",
                                children=[html.H4("Select a Scenario")],
                                className="flex-grow-1 overflow-auto border rounded-3 p-2",
                                style={"height": "175px"}
                            )
                        )
                    ]
                ),
                dbc.Row(
                    justify="center",
                    children=dbc.Button(
                        "2. Generate AI Policies for Scenario",
                        id="presc-button",
                        className="me-1 mb-2 w-50",
                        n_clicks=0
                    )
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
        # @app.callback(
        #     Output({"type": "context-slider", "index": ALL}, "value"),
        #     Input("context-scatter", "clickData"),
        #     prevent_initial_call=True
        # )
        # def click_context(click_data):
        #     """
        #     Updates context sliders when a context point is clicked.
        #     """
        #     # TODO: This assumes the SSPs in the ssps.csv file are in order which they are
        #     scenario = int(click_data["points"][0]["pointNumber"])
        #     scenario = f"SSP{scenario+1}-Baseline"
        #     row = self.context_df[self.context_df["scenario"] == scenario].iloc[0]
        #     return [row[self.context_cols[i]] for i in range(4)]

        @app.callback(
            Output("ssp-desc", "children"),
            Input({"type": "context-slider", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def update_ssp_desc(context_values):
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

        @app.callback(
            Output("presc-button", "disabled", allow_duplicate=True),
            Output("presc-button", "children", allow_duplicate=True),
            Output("presc-button", "color", allow_duplicate=True),
            Input("presc-button", "n_clicks"),
            prevent_initial_call=True
        )
        def disable_button(n_clicks):
            """
            Disables the button after it is clicked and displays a loading message.
            """
            return n_clicks > 0, "Please wait...", "warning"

        @app.callback(
            Output("presc-button", "disabled", allow_duplicate=True),
            Output("presc-button", "children", allow_duplicate=True),
            Output("presc-button", "color", allow_duplicate=True),
            Input("reset-button", "disabled"),
            prevent_initial_call=True
        )
        def enable_button(reset_disabled):
            """
            Enables the button when the filtering is done and resets it.
            """
            return False, "2. Generate AI Policies for Scenario", "primary"

    def register_callbacks_big(self, app):
        """
        Registers callbacks for the big demo app.
        """
        @app.callback(
            Output({"type": "context-slider", "index": ALL}, "value"),
            Input("ssp-dropdown", "value"),
            prevent_initial_call=True,
            allow_duplicates=True
        )
        def select_ssp_dropdown(value):
            """
            When we select a dropdown value, update the sliders.
            TODO: How can we make it go the other way, where moving a slider updates the dropdown?
            Currently it may create an infinite loop...
            """
            scenario = f"{value}-Baseline"
            row = self.context_df[self.context_df["scenario"] == scenario].iloc[0]
            return [row[self.context_cols[i]] for i in range(4)]
        
        # @app.callback(
        #     Output("ssp-dropdown", "value"),
        #     Input({"type": "context-slider", "index": ALL}, "value"),
        #     State("ssp-dropdown", "value"),
        #     prevent_initial_call=True,
        #     allow_duplicates=True
        # )
        # def adjust_slider_reset_dropdown(context_values, ssp_value):
        #     """
        #     If we move a slider and we have a selected SSP, reset the SSP to None.
        #     """
        #     match = self.context_df[self.context_cols].eq(context_values).all(axis=1)

        #     # If a match is found, retrieve the row, otherwise handle the case where it isn't found
        #     if match.any():
        #         ssp_idx = match.idxmax()
        #         return f"SSP{ssp_idx+1}"
        #     else:
        #         return ssp_value
