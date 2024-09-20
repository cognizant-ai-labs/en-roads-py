"""
Link Component.
"""
import json

from dash import Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from app.classes import JUMBOTRON, CONTAINER, DESC_TEXT, HEADER
from enroadspy import load_input_specs
from enroadspy.generate_url import actions_to_url


class LinkComponent():
    """
    Component in charge of displaying the links to En-ROADS.
    """

    def __init__(self, cand_idxs: list[int]):
        self.cand_idxs = cand_idxs

        self.colors = ["brown", "red", "blue", "green", "pink", "lightblue", "orange"]
        self.energies = ["coal", "oil", "gas", "renew and hydro", "bio", "nuclear", "new tech"]
        self.demands = [f"Primary energy demand of {energy}" for energy in self.energies]

        with open("app/categories.json", "r", encoding="utf-8") as f:
            self.categories = json.load(f)
        self.input_specs = load_input_specs()

    def plot_energy_policy(self, energy_policy_jsonl: list[dict[str, list]], cand_idx: int) -> go.Figure:
        """
        Plots density chart from energy policy.
        Removes demands that are all 0.
        """
        policy_dfs = [pd.DataFrame(policy) for policy in energy_policy_jsonl]
        max_val = max(policy_df[self.demands].sum(axis=1).max(axis=0) for policy_df in policy_dfs)
        policy_df = policy_dfs[cand_idx]

        # Preprocess our policy df making it cumulative
        demands = [demand for demand in self.demands if policy_df[demand].max() > 0]
        for i in range(len(demands)-1, -1, -1):
            for j in range(i-1, -1, -1):
                policy_df[demands[i]] += policy_df[demands[j]]
        policy_df["year"] = list(range(1990, 2101))

        fig = go.Figure()

        for demand, color, energy in zip(demands, self.colors, self.energies):
            # Skip if demand is all 0
            if policy_df[demand].max() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=policy_df["year"],
                y=policy_df[demand],
                mode="lines",
                name=energy,
                line=dict(color=color),
                fill="tonexty"
            ))

        fig.update_layout(
            yaxis_range=[0, max_val],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=100, b=0),
            title=dict(
                text=f"Policy {cand_idx} Energy Source Distribution Over Time",
                x=0.5,
                y=0.9,
                xanchor="center"
            )
        )
        return fig

    def translate_context_actions_dict(self, context_actions_dict: dict[str, float]) -> html.Div:
        """
        Translates a context actions dict into a nice div to display
        """
        children = []
        for category in self.categories:
            children.append(html.H4(category))
            remove = True  # If we don't have any actions in this category, remove it
            for action in self.categories[category]:
                if action in context_actions_dict:
                    remove = False
                    input_spec = self.input_specs[self.input_specs["varId"] == action].iloc[0]
                    val = context_actions_dict[action]
                    if input_spec["kind"] == "slider":
                        formatting = input_spec["format"]
                        val_formatted = f"{val:{formatting}}"
                    else:
                        val_formatted = "on" if val else "off"
                    children.append(html.P(f"{input_spec['varName']}: {val_formatted}"))
            if remove:
                children.pop()

        return html.Div(children)

    def create_link_div(self):
        """
        Creates content box containing links to the candidates to view.
        TODO: Make link unclickable while outcomes are loading.
        """
        div = html.Div(
            className=JUMBOTRON,
            children=[
                dbc.Container(
                    fluid=True,
                    className=CONTAINER,
                    children=[
                        html.H2("View Policy Energy Sources and Explore Policy in En-ROADS",
                                className=HEADER),
                        html.P("Select a policy to preview its resulting distribution of energy sources over time. \
                               Then click on the link to explore and fine-tune the policy in En-ROADS.",
                               className=DESC_TEXT),
                        html.Div(
                            className="d-flex flex-row w-25 justify-content-center",
                            children=[
                                html.Label("Policy: ", className="pt-1 me-1"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="cand-link-select",
                                        options=[],
                                        placeholder="Select a policy",
                                    ),
                                    className="flex-grow-1"
                                )
                            ]
                        ),
                        dcc.Loading(
                            type="circle",
                            target_components={"energy-policy-store": "*"},
                            children=[
                                dcc.Store(id="energy-policy-store"),
                                dcc.Graph(id="energy-policy-graph", className="mb-2")
                            ]
                        ),
                        html.Div(
                            className="d-flex flex-row justify-content-center",
                            children=[
                                dbc.Button("Show Actions", id="show-actions-button", className="me-1", n_clicks=0),
                                dbc.Button(
                                    "Explore Policy in En-ROADS",
                                    id="cand-link",
                                    target="_blank",
                                    rel="noopener noreferrer",
                                    disabled=True
                                )
                            ]
                        ),
                        dbc.Modal(
                            id="actions-modal",
                            scrollable=True,
                            is_open=False,
                            children=[
                                dbc.ModalHeader(dbc.ModalTitle("Actions")),
                                dbc.ModalBody("These are the actions taken", id="actions-body")
                            ]
                        )
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        """
        Registers callbacks for the links component.
        """
        @app.callback(
            Output("energy-policy-graph", "figure"),
            Input("energy-policy-store", "data"),
            Input("cand-link-select", "value")
        )
        def update_energy_policy_graph(energy_policy_jsonl: list[dict[str, list]], cand_idx) -> go.Figure:
            if cand_idx is not None:
                return self.plot_energy_policy(energy_policy_jsonl, cand_idx)

            # If we have no cand id just return a blank figure asking the user to select a candidate.
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text="Select a policy to view its energy source distribution",
                    x=0.5,
                    xanchor="center"
                )
            )
            return fig

        @app.callback(
            Output("cand-link", "href"),
            Output("cand-link", "disabled"),
            Input("context-actions-store", "data"),
            Input("cand-link-select", "value")
        )
        def update_cand_link(context_actions_dicts: list[dict[str, float]], cand_idx) -> tuple[str, bool]:
            """
            Updates the candidate link when a specific candidate is selected.
            Additionally un-disables the button if this is the first time we're selecting a candidate.
            """
            if cand_idx is not None:
                cand_dict = context_actions_dicts[cand_idx]
                link = actions_to_url(cand_dict)
                return link, False
            return "", True

        @app.callback(
            Output("actions-modal", "is_open"),
            Input("show-actions-button", "n_clicks"),
            State("actions-modal", "is_open")
        )
        def toggle_actions_modal(n_clicks, is_open):
            """
            Toggles the actions modal on and off.
            """
            if n_clicks:
                return True
            return is_open

        @app.callback(
            Output("actions-body", "children"),
            Output("show-actions-button", "disabled"),
            State("context-actions-store", "data"),
            Input("cand-link-select", "value")
        )
        def update_actions_body(context_actions_dicts: list[dict[str, float]], cand_idx) -> tuple[str, bool]:
            """
            Updates the body of the modal when a candidate is selected.
            """
            if cand_idx is not None:
                context_actions_dict = context_actions_dicts[cand_idx]
                return self.translate_context_actions_dict(context_actions_dict), False
            return "", True
