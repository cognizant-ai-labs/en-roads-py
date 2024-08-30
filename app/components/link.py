"""
Link Component.
"""
from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from generate_url import actions_to_url


class LinkComponent():
    """
    Component in charge of displaying the links to En-ROADS.
    """

    def __init__(self, cand_idxs: list[int]):
        self.cand_idxs = cand_idxs

        self.colors = ["brown", "red", "blue", "green", "pink", "lightblue", "orange"]
        self.energies = ["coal", "oil", "gas", "renew and hydro", "bio", "nuclear", "new tech"]
        self.demands = [f"Primary energy demand of {energy}" for energy in self.energies]

    def plot_energy_policy(self, energy_policy_jsonl, cand_idx):
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
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    def create_button_group(self):
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
            className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
            children=[
                dbc.Container(
                    fluid=True,
                    className="py-3 d-flex flex-column justify-content-center",
                    children=[
                        html.H2("View Energy Policy and Visualize/Modify Actions in En-ROADS",
                                className="text-center mb-2"),
                        html.P("Click on a candidate to preview the distribution of energy sources over time due to \
                               its prescribed energy policy. Then click on the link to view the full policy in \
                               En-ROADS.",
                               className="text-center w-70 mb-2 mx-auto"),
                        html.Div(
                            className="w-50 mx-auto",
                            children=[self.create_button_group()]
                        ),
                        dcc.Loading(
                            type="circle",
                            children=[
                                dcc.Store(id="energy-policy-store"),
                                dcc.Graph(id="energy-policy-graph", className="mb-2")
                            ]
                        ),
                        dbc.Button("View in En-ROADS",
                                   id="cand-link",
                                   target="_blank",
                                   rel="noopener noreferrer",
                                   className="w-25 mx-auto")
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        @app.callback(
            Output("energy-policy-graph", "figure"),
            Input("energy-policy-store", "data"),
            Input("cand-link-select", "value")
        )
        def update_energy_policy_graph(energy_policy_jsonl, cand_idx):
            return self.plot_energy_policy(energy_policy_jsonl, cand_idx)

        @app.callback(
            Output("cand-link", "href"),
            Input("context-actions-store", "data"),
            Input("cand-link-select", "value")
        )
        def update_cand_links(context_actions_dicts: list[dict[str, float]], cand_idx):
            """
            Updates the candidate link when a specific candidate is selected.
            """
            cand_dict = context_actions_dicts[cand_idx]
            link = actions_to_url(cand_dict)
            return link
