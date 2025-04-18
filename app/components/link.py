"""
En-ROADS linking component.
"""
from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc

from app.components.component import Component
from app.components.timeline import TimelineComponent
from enroadspy.generate_url import actions_to_url


class LinkComponent(Component):
    """
    Component in charge of the iFrame that displays En-ROADS.
    The iFrame is cleared when the user clicks away from the screen so that we don't get weird scrolling issues.
    """
    def __init__(self, actions: list[str]):
        self.timeline_component = TimelineComponent(actions)

    def create_div(self) -> html.Div:
        """
        Creates div showing an iframe displaying enroads, a dropdown to select the candidate, and a button to show the
        actions via. the timeline component.
        """
        timeline_div = self.timeline_component.create_div()
        div = html.Div(
            children=[
                dbc.Row(
                    className="w-100, mb-3",
                    justify="center",
                    align="center",
                    children=dbc.Col(
                        width="auto",
                        children=dbc.Button("3. Open Policy Explorer", id="enroads-button")
                    )
                ),
                dbc.Modal(
                    id="enroads-modal",
                    is_open=False,
                    fullscreen=True,
                    children=[
                        dbc.ModalHeader(html.H2("Policy Explorer")),
                        dbc.ModalBody(
                            children=[
                                dbc.Row(
                                    className="w-100 mb-3",
                                    justify="center",
                                    align="center",
                                    children=[
                                        dbc.Col(
                                            width=3,
                                            children=dcc.Dropdown(
                                                id="cand-link-select",
                                                options=[],
                                                placeholder="Explore single policy",
                                                maxHeight=100
                                            ),
                                        ),
                                        dbc.Col(
                                            width="auto",
                                            children=timeline_div
                                        )
                                    ]
                                ),
                                dbc.Row(
                                    html.Iframe(
                                        style={"height": "75vh", "width": "100%"},
                                        src="google.com",
                                        id="enroads-iframe"
                                    )
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
        Registers callbacks for the links component.
        Also registers the timeline component's callbacks.
        """
        @app.callback(
            Output("enroads-iframe", "src"),
            Input("context-actions-store", "data"),
            Input("cand-link-select", "value")
        )
        def update_cand_link(context_actions_dicts: list[dict[str, float]], cand_idx: int) -> tuple[str, bool]:
            """
            Updates the candidate link when a specific candidate is selected.
            """
            if cand_idx is not None:
                cand_dict = context_actions_dicts[cand_idx]
                link = actions_to_url(cand_dict)
                return link
            return ""

        @app.callback(
            Output("enroads-modal", "is_open"),
            Input("enroads-button", "n_clicks"),
            prevent_initial_call=True
        )
        def toggle_enroads_modal(_):
            """
            Toggles the enroads modal.
            """
            return True

        # Register the timeline component's callbacks.
        self.timeline_component.register_callbacks(app)
