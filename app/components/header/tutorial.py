"""
Tutorial component housing all the tooltips that pop up when you press the tutorial button.
"""
import dash_bootstrap_components as dbc
from dash import Input, Output, State, ALL, html

from app.components.component import Component


class TutorialComponent(Component):
    """
    Component storing all the tooltips that target various parts of the app.
    """
    def create_div(self) -> html.Div:
        """
        Creates the button to show the tooltips as well as the tooltips themselves.
        """
        tutorial_params = {"is_open": False, "trigger": None}
        div = html.Div(
            children=[
                dbc.Button(id="tutorial-button", className="bi bi-question-circle", color="secondary"),
                dbc.Tooltip(target="tutorial-button", placement="bottom", children="Show/Hide Tutorial"),
                dbc.Tooltip(id={"type": "tutorial", "index": 0},
                            target="ssp-dropdown",
                            children="1a. Select scenario",
                            **tutorial_params),
                dbc.Tooltip(id={"type": "tutorial", "index": 1},
                            target={"type": "context-slider", "index": 0},
                            children="(1b. Optional: adjust scenario)",
                            **tutorial_params),
                dbc.Tooltip(id={"type": "tutorial", "index": 2},
                            target="presc-button",
                            children="1c. Run AI models",
                            **tutorial_params),
                dbc.Tooltip(id={"type": "tutorial", "index": 3},
                            target={"type": "metric-slider", "index": 0},
                            children="2. Filter metrics as desired",
                            **tutorial_params),
                dbc.Tooltip(id={"type": "tutorial", "index": 4},
                            target="scroll-button",
                            children="3. Examine individual policies",
                            **tutorial_params)
            ]
        )
        return div

    def register_callbacks(self, app):
        """
        Registers tutorial callbacks.
        """
        @app.callback(
            Output({"type": "tutorial", "index": ALL}, "is_open"),
            Input("tutorial-button", "n_clicks"),
            State({"type": "tutorial", "index": ALL}, "is_open"),
            prevent_initial_call=True
        )
        def toggle_tutorial(_, is_opens):
            """
            Toggles the tutorial tooltips
            """
            if is_opens[0]:
                return [False] * len(is_opens)
            return [True] * len(is_opens)
