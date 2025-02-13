"""
Link Component.
"""
from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc

from app.components.component import Component
from app.components.timeline import TimelineComponent
from enroadspy import load_input_specs
from enroadspy.generate_url import actions_to_url


class LinkComponent(Component):
    """
    Component in charge of displaying the links to En-ROADS.
    """

    def __init__(self, cand_idxs: list[int], actions: list[str]):
        self.cand_idxs = [i for i in cand_idxs]

        self.colors = ["brown", "red", "blue", "green", "pink", "lightblue", "orange"]
        self.energies = ["coal", "oil", "gas", "renew and hydro", "bio", "nuclear", "new tech"]
        self.demands = [f"Primary energy demand of {energy}" for energy in self.energies]

        self.input_specs = load_input_specs()

        self.timeline_component = TimelineComponent(actions)

    def create_div(self):
        """
        Creates div showing an iframe displaying enroads, a dropdown to select the candidate, and a button to show the
        actions via. the timeline component.
        """
        timeline_div = self.timeline_component.create_div()
        div = html.Div(
            children=[
                dbc.Row(
                    style={"height": "80vh"},
                    children=[
                        html.Iframe(src="google.com", id="enroads-iframe")
                    ]
                ),
                dbc.Row(
                    justify="center",
                    align="center",
                    className="mt-2",
                    children=[
                        dbc.Col(
                            html.A(
                                dbc.Button(
                                    className="bi bi-arrow-up",
                                    color="secondary"
                                ),
                                href="#main-page",
                                id="back-button"
                            ),
                            width=1
                        ),
                        dbc.Col(html.Label("Policy:"), width="auto"),
                        dbc.Col(
                            html.Div(
                                dcc.Dropdown(
                                    id="cand-link-select",
                                    options=[],
                                    placeholder="Select a policy",
                                    maxHeight=100
                                )
                            ),
                            width=2
                        ),
                        dbc.Col(timeline_div, width=2)
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        """
        Registers callbacks for the links component.
        Also registers the timeline component's callbacks.
        """
        @app.callback(
            Output("enroads-iframe", "src"),
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
                return link
            return ""

        @app.callback(
            Output("cand-link-select", "value"),
            Input("back-button", "n_clicks"),
            prevent_initial_call=True
        )
        def clear_iframe(_):
            """
            Clears the iframe by deselecting the cand link selector.
            """
            return None

        # Register the timeline component's callbacks.
        self.timeline_component.register_callbacks(app)
