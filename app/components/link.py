"""
Link Component.
"""
from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from app.classes import JUMBOTRON, CONTAINER, DESC_TEXT, HEADER
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

    def plot_energy_policy(self, energy_policy_jsonl: list[dict[str, list]], cand_idx: int) -> go.Figure:
        """
        Plots density chart from energy policy.
        Removes demands that are all 0.
        """
        policy_df = pd.DataFrame(energy_policy_jsonl[cand_idx])

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

    def create_button_group(self) -> html.Div:
        """
        Creates button group to select candidate to link to.
        """
        button_options = [{"label": str(cand_idx), "value": cand_idx} for cand_idx in self.cand_idxs]
        button_group = html.Div(
            [
                dbc.RadioItems(
                    id="cand-link-select",
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    options=button_options,
                    value=0,
                ),
            ],
            className="radio-group d-flex justify-content-center",
        )
        return button_group

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
                        dcc.Loading(
                            type="circle",
                            children=[
                                dcc.Store(id="energy-policy-store"),
                                dcc.Graph(id="energy-policy-graph", className="mb-2")
                            ]
                        ),
                        dbc.Row(
                            className="w-75 mx-auto",
                            align="center",
                            children=[
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="cand-link-select",
                                        options=[],
                                        placeholder="Select a policy"
                                    ),
                                    width={"size": 3, "offset": 1}
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Explore & Fine-Tune Policy in En-ROADS",
                                        id="cand-link",
                                        target="_blank",
                                        rel="noopener noreferrer",
                                        size="lg",
                                        disabled=True
                                    ),
                                    width={"size": 4}
                                )
                            ]
                        ),
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
