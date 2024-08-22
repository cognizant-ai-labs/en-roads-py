"""
Component displaying the context and handling its related functions.
"""
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.express as px

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
                         "Transition Time",
                         "Population"]
        
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

        fig = px.scatter(context_chart_df,
                        x="population",
                        y="gdp",
                        color="scenario",
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        labels={"population": "Population 2100 (B)", "gdp": "GDPPP 2100 (T)"},
                        hover_data={"description": True, "population": False, "scenario": False, "gdp": False})

        fig.update_layout(
            title = {
                "text": "Select a Pre-Set Context Scenario",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Population 2100 (B)",
            yaxis_title="GDP 2100 (T$)",
            showlegend=True
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
            label_slider = [
                html.Div(
                    style={"grid-column": "1", "grid-row": f"{i+1}"},
                    children=[html.Label(varname)]
                ),
                html.Div(
                    style={"grid-column": "2", "grid-row": f"{i+1}"},
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
            ]
            sliders.extend(label_slider)

        # TODO: Fix the widths to be responsive
        sliders_div = html.Div(
            style={"display": "grid", "grid-template-columns": "1fr 55%"},
            children=sliders
        )

        div = html.Div(
            className="contentBox",
            children=[
                html.H2("Select a Context Scenario to Optimize For", style={"textAlign": "center"}),
                html.Div(
                    style={"display": "grid", "grid-template-columns": "50% 50%", "margin-bottom": "20px"},
                    children=[
                        html.Div(
                            style={"grid-column": "1", "height": "100%", "width": "100%", "padding-bottom": "40px"},
                            children=[
                                dcc.Graph(id="context-scatter", figure=self.create_context_scatter())
                            ]
                        ),
                        html.Div(
                            style={"grid-column": "2", "width": "100%", "margin-top": "3%"},
                            children=[
                                sliders_div,
                                html.Div(
                                    id="ssp-desc",
                                    children=[self.construct_ssp_desc(0)]
                                )
                            ]
                        ),
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
            html.H4(ssp_row["ssp"] + ": " + ssp_row["description"]),
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
                scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1
            else:
                scenario = 0
            
            scenario = f"SSP{scenario+1}-Baseline"
            row = self.context_df[self.context_df["scenario"] == scenario].iloc[0]
            return row[self.context_cols[0]], row[self.context_cols[1]], row[self.context_cols[2]], row[self.context_cols[3]]
        
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

