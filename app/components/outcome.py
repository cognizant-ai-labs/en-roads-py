"""
OutcomeComponent class for the outcome section of the app.
"""
from dash import Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.utils import EvolutionHandler, filter_metrics_json


class OutcomeComponent():
    """
    Component in charge of showing the outcomes of the prescribed actions for the selected prescriptors.
    Has drop downs to allow the user to select which outcomes they want to see.
    TODO: Make it so we only load the selected prescriptors.
    """
    def __init__(self, evolution_handler: EvolutionHandler):
        self.evolution_handler = evolution_handler
        self.context_cols = ["_long_term_gdp_per_capita_rate",
                             "_near_term_gdp_per_capita_rate",
                             "_transition_time_to_reach_long_term_gdp_per_capita_rate",
                             "_global_population_in_2100"]
        self.plot_outcomes = ["Temperature change from 1850",
                              "Adjusted cost of energy per GJ",
                              "Government net revenue from adjustments",
                              "Total Primary Energy Demand"]

        self.metric_ids = [metric.replace(" ", "-").replace(".", "_") for metric in self.evolution_handler.outcomes]

    def plot_outcome_over_time(self, outcome: str, outcomes_jsonl: list[list[dict[str, float]]], cand_idxs: list[int]):
        """
        Plots all the candidates' prescribed actions' outcomes for a given context.
        Also plots the baseline given the context.
        TODO: Fix colors to match parcoords
        """
        best_cand_idxs = cand_idxs[:10]
        outcomes_dfs = [pd.DataFrame(outcomes_json) for outcomes_json in outcomes_jsonl]
        color_map = px.colors.qualitative.Plotly

        fig = go.Figure()
        showlegend = True
        for cand_idx in cand_idxs[10:]:
            outcomes_df = outcomes_dfs[cand_idx]
            outcomes_df["year"] = list(range(1990, 2101))
            # Legend group other so all the other candidates get removed when we click on it.
            # Name other because the first other candidate represents them all in the legend
            fig.add_trace(go.Scatter(
                x=outcomes_df["year"],
                y=outcomes_df[outcome],
                mode='lines',
                legendgroup="other",
                name="other",
                showlegend=showlegend,
                line=dict(color="lightgray")
            ))
            showlegend = False

        for cand_idx in best_cand_idxs:
            outcomes_df = outcomes_dfs[cand_idx]
            outcomes_df["year"] = list(range(1990, 2101))
            fig.add_trace(go.Scatter(
                x=outcomes_df["year"],
                y=outcomes_df[outcome],
                mode='lines',
                name=str(cand_idx),
                line=dict(color=color_map[cand_idxs.index(cand_idx)]),
                showlegend=True
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

        # Standardize the max and min of the y-axis so that the graphs are comparable when we start filtering
        # models.
        y_min = min([outcomes_df[outcome].min() for outcomes_df in outcomes_dfs])
        y_max = max([outcomes_df[outcome].max() for outcomes_df in outcomes_dfs])

        fig.update_layout(
            title={
                "text": f"{outcome} Over Time",
                "x": 0.5,
                "xanchor": "center"},
            yaxis_range=[y_min, y_max]
        )
        return fig

    def create_outcomes_div(self):
        """
        Note: We have nested loads here. The outer load is for both graphs and triggers when the outcomes store
        is updated. Otherwise, we have individual loads for each graph.
        """
        div = html.Div(
            className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
            children=[
                dbc.Container(
                    fluid=True,
                    className="py-3",
                    children=[
                        dbc.Row(html.H2("Outcomes for Selected Policies", className="text-center mb-5")),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        dcc.Dropdown(
                                            id="outcome-dropdown-1",
                                            options=self.plot_outcomes,
                                            value=self.plot_outcomes[0]
                                        )
                                    ]
                                ),
                                dbc.Col(
                                    children=[
                                        dcc.Dropdown(
                                            id="outcome-dropdown-2",
                                            options=self.plot_outcomes,
                                            value=self.plot_outcomes[1]
                                        )
                                    ]
                                )
                            ]
                        ),
                        dcc.Loading(
                            target_components={"context-actions-store": "*", "outcomes-store": "*"},
                            type="circle",
                            children=[
                                dcc.Store(id="context-actions-store"),
                                dcc.Store(id="outcomes-store"),
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            dcc.Graph(id="outcome-graph-1"),
                                            width=6
                                        ),
                                        dbc.Col(
                                            dcc.Graph(id="outcome-graph-2"),
                                            width=6
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

    def register_callbacks(self, app):
        """
        Registers callbacks relating to the outcomes section of the app.
        """
        @app.callback(
            Output("context-actions-store", "data"),
            Output("outcomes-store", "data"),
            Output("metrics-store", "data"),
            Output("energy-policy-store", "data"),
            [Input(f"context-slider-{i}", "value") for i in range(4)]
        )
        def update_results_stores(*context_values):
            """
            When the context sliders are changed, prescribe actions for the context for all candidates. Then run them
            through En-ROADS to get the outcomes. Finally process the outcomes into metrics. Store the context-actions
            dicts, outcomes dfs, and metrics df in stores.
            Also stores the energy policies in the energy-policy-store in link.py.
            TODO: Make this only load selected candidates.
            """
            # Prescribe actions for all candidates via. torch
            context_dict = dict(zip(self.context_cols, context_values))
            context_actions_dicts = self.evolution_handler.prescribe_all(context_dict)

            # Attach baseline (no actions)
            context_actions_dicts.append(dict(**context_dict))

            # Run En-ROADS on all candidates and save as jsonl
            outcomes_dfs = self.evolution_handler.context_actions_to_outcomes(context_actions_dicts)
            outcomes_jsonl = [outcomes_df[self.plot_outcomes].to_dict("records") for outcomes_df in outcomes_dfs]

            # Process outcomes into metrics and save
            metrics_df = self.evolution_handler.outcomes_to_metrics(context_actions_dicts, outcomes_dfs)
            metrics_json = metrics_df.to_dict("records")

            # Parse energy demand policy from outcomes for use in link.py
            energies = ["coal", "oil", "gas", "renew and hydro", "bio", "nuclear", "new tech"]
            demands = [f"Primary energy demand of {energy}" for energy in energies]
            energy_policy_jsonl = [outcomes_df[demands].to_dict("records") for outcomes_df in outcomes_dfs]

            return context_actions_dicts, outcomes_jsonl, metrics_json, energy_policy_jsonl

        @app.callback(
            Output("outcome-graph-1", "figure"),
            Output("outcome-graph-2", "figure"),
            State("metrics-store", "data"),
            Input("outcome-dropdown-1", "value"),
            Input("outcome-dropdown-2", "value"),
            Input("outcomes-store", "data"),
            [Input(f"{metric_id}-slider", "value") for metric_id in self.metric_ids],
        )
        def update_outcomes_plots(metrics_json, outcome1, outcome2, outcomes_jsonl, *metric_ranges):
            """
            Updates outcome plot when specific outcome is selected or context scatter point is clicked.
            """
            metrics_df = filter_metrics_json(metrics_json, metric_ranges)
            cand_idxs = list(metrics_df.index)

            fig1 = self.plot_outcome_over_time(outcome1, outcomes_jsonl, cand_idxs)
            fig2 = self.plot_outcome_over_time(outcome2, outcomes_jsonl, cand_idxs)
            return fig1, fig2
        
        @app.callback(
            Output("cand-link-select", "options"),
            State("metrics-store", "data"),
            [Input(f"{metric_id}-slider", "value") for metric_id in self.metric_ids]
        )
        def update_cand_link_select(metrics_json: dict[str, list],
                                    *metric_ranges: list[tuple[float, float]]) -> list[int]:
            """
            Updates the available candidates in the link dropdown based on metric ranges.
            """
            metrics_df = filter_metrics_json(metrics_json, metric_ranges)
            cand_idxs = list(metrics_df.index)
            return cand_idxs
