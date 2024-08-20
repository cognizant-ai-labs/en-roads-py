from dash import Input, Output, html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.utils import EvolutionHandler

class OutcomeComponent():
    """
    TODO: Move cand_idxs that are displayed into a separate component selector.
    TODO: Cache results so changing the graph type doesn't reload the data.
    """
    def __init__(self, evolution_handler: EvolutionHandler, cand_idxs: list[str]):
        self.evolution_handler = evolution_handler
        self.cand_idxs = cand_idxs
        self.context_cols = ["_long_term_gdp_per_capita_rate",
                             "_near_term_gdp_per_capita_rate",
                             "_transition_time_to_reach_long_term_gdp_per_capita_rate",
                             "_global_population_in_2100"]
        self.plot_outcomes = ["Temperature change from 1850",
                              "Adjusted cost of energy per GJ",
                              "Government net revenue from adjustments", 
                              "Total Primary Energy Demand"]

    def plot_outcome_over_time(self, outcome: str, outcomes_jsonl: list[list[dict[str, float]]], cand_idxs: list[int]):
        """
        Plots all the candidates' prescribed actions' outcomes for a given context.
        Also plots the baseline given the context.
        TODO: Fix colors to match parcoords
        """
        outcomes_dfs = [pd.DataFrame(outcomes_json) for outcomes_json in outcomes_jsonl]
        color_map = [c for c in px.colors.qualitative.Plotly]

        fig = go.Figure()
        showlegend = True
        for cand_idx, outcomes_df in enumerate(outcomes_dfs[:-1]):
            outcomes_df["year"] = list(range(1990, 2101))
            if cand_idx not in cand_idxs:
                legend = showlegend
                name = "other"
                line = dict(color="lightgray")
                showlegend = False

                fig.add_trace(go.Scatter(
                    x=outcomes_df["year"],
                    y=outcomes_df[outcome],
                    mode='lines',
                    name=name,
                    line=line,
                    showlegend=legend
                ))

        for cand_idx in cand_idxs:
            outcomes_df = outcomes_dfs[cand_idx]
            legend = True
            name = str(cand_idx)
            line = dict(color=color_map[cand_idxs.index(cand_idx)])
            fig.add_trace(go.Scatter(
                x=outcomes_df["year"],
                y=outcomes_df[outcome],
                mode='lines',
                name=name,
                line=line,
                showlegend=legend
            ))

        baseline_outcomes_df = outcomes_dfs[-1]
        baseline_outcomes_df["year"] = list(range(1990, 2101))
        fig.add_trace(go.Scatter(
            x=baseline_outcomes_df["year"],
            y=baseline_outcomes_df[outcome],
            mode='lines',
            name="baseline",
            line=dict(color="black"),
            showlegend=True
        ))

        fig.update_layout(
            title = {
                "text": f"{outcome} Over Time",
                "x": 0.5,
                "xanchor": "center",},
        )
        return fig
    
    def create_outcomes_div(self):
        """
        Note: We have nested loads here. The outer load is for both graphs and triggers when the outcomes store
        is updated. Otherwise, we have individual loads for each graph.
        """
        div = html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(self.plot_outcomes, self.plot_outcomes[0], id="outcome-dropdown-1")
                ], style={"width": "50%", "display": "inline-block"}),
                html.Div([
                    dcc.Dropdown(self.plot_outcomes, self.plot_outcomes[1], id="outcome-dropdown-2")
                ], style={"width": "50%", "display": "inline-block"}),
            ]),
            html.Div([
                dcc.Loading([
                    dcc.Store(id="context-actions-store"),
                    dcc.Store(id="outcomes-store"),
                    html.Div([
                        dcc.Graph(id="outcome-graph-1")
                    ], style={"width": "50%", "display": "inline-block"}),
                    html.Div([
                        dcc.Graph(id="outcome-graph-2")
                    ], style={"width": "50%", "display": "inline-block"}),
                ], type="circle", target_components={"context-actions-store": "*", "outcomes-store": "*"})
            ])
        ])

        return div

    def register_callbacks(self, app):
        @app.callback(
            Output("context-actions-store", "data"),
            Output("outcomes-store", "data"),
            [Input(f"context-slider-{i}", "value") for i in range(4)]
        )
        def update_outcomes_store(*context_values):
            context_dict = dict(zip(self.context_cols, context_values))
            context_actions_dicts = self.evolution_handler.prescribe_all(context_dict)
            outcomes_dfs = self.evolution_handler.context_actions_to_outcomes(context_actions_dicts)
            baseline_outcomes_df = self.evolution_handler.context_baseline_outcomes(context_dict)
            outcomes_dfs.append(baseline_outcomes_df)

            outcomes_jsonl = [outcomes_df[self.plot_outcomes].to_dict("records") for outcomes_df in outcomes_dfs]
            return context_actions_dicts, outcomes_jsonl

        @app.callback(
            Output("outcome-graph-1", "figure"),
            Input("outcome-dropdown-1", "value"),
            Input("outcomes-store", "data"),
            Input("cand-select-dropdown", "value")
        )
        def update_outcomes_plot_1(outcome, outcomes_jsonl, cand_idxs):
            """
            Updates outcome plot when specific outcome is selected or context scatter point is clicked.
            """
            if not cand_idxs:
                cand_idxs = []
            fig = self.plot_outcome_over_time(outcome, outcomes_jsonl, cand_idxs)
            return fig

        @app.callback(
            Output("outcome-graph-2", "figure"),
            Input("outcome-dropdown-2", "value"),
            Input("outcomes-store", "data"),
            Input("cand-select-dropdown", "value")
        )
        def update_outcomes_plot_2(outcome, outcomes_jsonl, cand_idxs):
            """
            Updates outcome plot when specific outcome is selected or context scatter point is clicked.
            """
            if not cand_idxs:
                cand_idxs = []
            fig = self.plot_outcome_over_time(outcome, outcomes_jsonl, cand_idxs)
            return fig
