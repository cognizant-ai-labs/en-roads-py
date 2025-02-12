"""
OutcomeComponent class for the outcome section of the app.
"""
import json
import random

from dash import Input, Output, State, html, dcc, MATCH, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.component import Component
from app.utils import EvolutionHandler, filter_metrics_json


class OutcomeComponent(Component):
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

        with open("app/units.json", "r", encoding="utf-8") as f:
            self.units = json.load(f)

    def plot_outcome_over_time(self, outcome: str, outcomes_jsonl: list[list[dict[str, float]]], cand_idxs: list[int]):
        """
        Plots all the candidates' prescribed actions' outcomes for a given context.
        Also plots the baseline given the context.
        TODO: Fix colors to match parcoords
        """
        best_cand_idxs = cand_idxs[:10]
        outcomes_dfs = [pd.DataFrame(outcomes_json) for outcomes_json in outcomes_jsonl]
        # Used later to standardize the max and min of the y-axis so that the graphs are comparable when we start
        # filtering the models.
        y_min = min([outcomes_df[outcome].min() for outcomes_df in outcomes_dfs])
        y_max = max([outcomes_df[outcome].max() for outcomes_df in outcomes_dfs])

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

        # Plot the year we start changing (2024)
        fig.add_shape(
            type="line",
            x0=2024,
            y0=y_min,
            x1=2024,
            y1=y_max,
            line=dict(color="black", width=1, dash="dash")
        )

        # If we're looking at temperature, put in some dashed lines for the 1.5 and 2.0 degree targets
        if outcome == "Temperature change from 1850":
            fig.add_shape(
                type="line",
                x0=1990,
                y0=1.5,
                x1=2100,
                y1=1.5,
                line=dict(color="black", width=1, dash="dash"),
            )
            fig.add_shape(
                type="line",
                x0=1990,
                y0=2,
                x1=2100,
                y1=2,
                line=dict(color="gray", width=1, dash="dash"),
            )

        fig.update_layout(
            title={
                "text": f"{outcome}",
                "x": 0.5,
                "xanchor": "center"},
            yaxis_range=[y_min, y_max],
            yaxis_title=outcome + f" ({self.units[outcome]})"
        )
        return fig

    def create_outcome_graph_label(self, idx: int) -> html.Div:
        """
        Creates pairs of outcome dropdown and outcome graph so they're lined up.
        """
        pair = html.Div(
            children=[
                html.Div(
                    dcc.Dropdown(
                        id={"type": "outcome-dropdown", "index": idx},
                        options=self.plot_outcomes,
                        value=self.plot_outcomes[idx],
                        disabled=True
                    ),
                ),
                dcc.Graph(id={"type": "outcome-graph", "index": idx})
            ]
        )
        return pair

    def create_custom_spinner(self):
        """
        Creates a custom spinner with some fun pre-set text.
        """
        phrases = [
            "Saving the planet...",
            "Planting some trees...",
            "Protecting polar bears' habitats...",
            "Performing nuclear fission...",
            "Taxing carbon...",
            "Building wind turbines...",
            "De-acidifying the ocean...",
            "Preventing famine...",
            "Stopping hurricanes...",
            "Fostering international cooperation...",
            "Funding climate research...",
            "Impeaching climate deniers...",
            "Asking Taylor to stop flying so much...",
            "Inspiring the youth...",
            "Cleaning up rivers...",
            "Preserving biodiversity..."
        ]
        phrase = random.choice(phrases)
        return html.Div(
            className="d-flex flex-column align-items-center",
            children=[html.H2(phrase), html.H2(dbc.Spinner(color="primary"))]
        )

    def create_div(self):
        """
        Creates the outcomes div for the big demo. We want all the graphs to be lined up in a row.
        """
        outcome_labels = [self.create_outcome_graph_label(i) for i in range(4)]

        div = html.Div(
            className="w-100",
            children=[
                dcc.Loading(
                    id="outcomes-loading",
                    target_components={"context-actions-store": "*", "outcomes-store": "*"},
                    overlay_style={"visibility": "visible", "opacity": 0.2},
                    custom_spinner=html.Div(id="outcomes-spinner", children=self.create_custom_spinner()),
                    children=[
                        dcc.Store(id="context-actions-store"),
                        dcc.Store(id="outcomes-store"),
                        dbc.Row(
                            className="g-0",
                            children=[
                                dbc.Col(outcome_label, width=3) for outcome_label in outcome_labels
                            ]
                        )
                    ]
                )
            ]
        )
        return div

    def create_filtered_outcome_plot(self,
                                     metrics_json: dict,
                                     outcome: str,
                                     outcomes_jsonl: list[dict],
                                     metric_ranges: list[tuple[float, float]]):
        """
        Creates a filtered outcome plot based on the the filtered metrics and the given outcome and outcomes data.
        """
        metrics_df = filter_metrics_json(metrics_json, metric_ranges)
        cand_idxs = list(metrics_df.index)[:-1]  # So we don't include the baseline
        fig = self.plot_outcome_over_time(outcome, outcomes_jsonl, cand_idxs)
        return fig

    def register_callbacks(self, app):
        """
        Registers callbacks relating to the outcomes section of the app.
        """
        @app.callback(
            Output("context-actions-store", "data"),
            Output("outcomes-store", "data"),
            Output("metrics-store", "data"),
            Input("presc-button", "n_clicks"),
            State({"type": "context-slider", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def update_results_stores(_, context_values: list[float]):
            """
            When the presc button is pressed, prescribe actions for the context for all candidates. Then run them
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
            # energies = ["coal", "oil", "gas", "renew and hydro", "bio", "nuclear", "new tech"]
            # demands = [f"Primary energy demand of {energy}" for energy in energies]
            # energy_policy_jsonl = [outcomes_df[demands].to_dict("records") for outcomes_df in outcomes_dfs]

            # return context_actions_dicts, outcomes_jsonl, metrics_json, energy_policy_jsonl
            return context_actions_dicts, outcomes_jsonl, metrics_json

        @app.callback(
            Output({"type": "outcome-graph", "index": ALL}, "figure", allow_duplicate=True),
            Output({"type": "outcome-dropdown", "index": ALL}, "disabled", allow_duplicate=True),
            State("metrics-store", "data"),
            State({"type": "outcome-dropdown", "index": ALL}, "value"),
            State("outcomes-store", "data"),
            Input({"type": "metric-slider", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def filter_outcomes_plots(metrics_json: dict,
                                  outcomes: list[str],
                                  outcomes_jsonl: list[dict],
                                  metric_ranges: list[tuple[float, float]]):
            """
            Filters outcome when sliders are changed. Also un-disables them after loading.
            """
            figs = [self.create_filtered_outcome_plot(metrics_json, o, outcomes_jsonl, metric_ranges) for o in outcomes]
            return figs, [False] * len(outcomes)

        @app.callback(
            Output({"type": "outcome-graph", "index": MATCH}, "figure", allow_duplicate=True),
            State("metrics-store", "data"),
            Input({"type": "outcome-dropdown", "index": MATCH}, "value"),
            State("outcomes-store", "data"),
            State({"type": "metric-slider", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def change_outcome_type(metrics_json: dict,
                                outcome: str,
                                outcomes_jsonl: list[dict],
                                metric_ranges: list[tuple[float, float]]):
            """
            Changes the type of outcome being displayed when the dropdown is selected.
            """
            return self.create_filtered_outcome_plot(metrics_json, outcome, outcomes_jsonl, metric_ranges)

        @app.callback(
            Output("cand-link-select", "options"),
            State("metrics-store", "data"),
            Input({"type": "metric-slider", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def update_cand_link_select(metrics_json: dict[str, list],
                                    metric_ranges: list[tuple[float, float]]) -> list[int]:
            """
            Updates the available candidates in the link dropdown based on metric ranges.
            """
            metrics_df = filter_metrics_json(metrics_json, metric_ranges)
            cand_idxs = list(metrics_df.index)[:-1]  # So we don't include the baseline
            return cand_idxs
        
        @app.callback(
            Output("outcomes-spinner", "children", allow_duplicate=True),
            Input("presc-button", "n_clicks"),
            prevent_initial_call=True
        )
        def update_outcomes_loading_spinner(_):
            """
            Updates the spinner with a fun new phrase every time the presc button is clicked.
            """
            return self.create_custom_spinner()
