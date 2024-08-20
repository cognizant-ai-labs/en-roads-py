from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class ContextComponent():

    def __init__(self):
        self.context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
        self.context_cols = ["_long_term_gdp_per_capita_rate",
                             "_near_term_gdp_per_capita_rate",
                             "_transition_time_to_reach_long_term_gdp_per_capita_rate",
                             "_global_population_in_2100"]

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
                        labels={"population": "Population 2100 (B)", "gdp": "GDPPP 2100 (T)"},
                        hover_data={"description": True, "population": False, "scenario": False, "gdp": False})

        fig.update_layout(
            title = {
                "text": "Select a Context Scenario",
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
        for i, context_col in enumerate(self.context_cols):
            row = input_specs[input_specs["varId"] == context_col].iloc[0]
            slider = dcc.Slider(
                id=f"context-slider-{i}",
                min=row["minValue"],
                max=row["maxValue"],
                step=row["step"],
                value=row["defaultValue"],
                tooltip={"placement": "bottom"},
                marks=None
            )
            sliders.append(slider)

        div = html.Div([
            html.Div([
                dcc.Graph(id="context-scatter", figure=self.create_context_scatter())
            ], style={"width": "50%", "display": "inline-block"}),
            html.Div(sliders, style={"width": "50%", "display": "inline-block"}),
        ])

        return div

    def register_callbacks(self, app):
        """
        Registers app's callbacks.
        """
        @app.callback(
            Output("context-slider-0", "value"),
            Output("context-slider-1", "value"),
            Output("context-slider-2", "value"),
            Output("context-slider-3", "value"),
            Input("context-scatter", "clickData")
        )
        def click_context(click_data):
            """
            Updates context sliders when a context point is clicked.
            """
            if click_data:
                scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1
            else:
                scenario = 0
            
            scenario = f"SSP{scenario+1}-Baseline"
            row = self.context_df[self.context_df["scenario"] == scenario].iloc[0]
            return row[self.context_cols[0]], row[self.context_cols[1]], row[self.context_cols[2]], row[self.context_cols[3]]
