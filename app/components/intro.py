"""
Component showing the little intro blurb and the arrow to get started.
"""
from dash import html, Input, Output
import dash_bootstrap_components as dbc


class IntroComponent():
    """
    Title card component
    """
    def create_intro_div(self):
        """
        Creates the intro title card describing the project.
        """
        div = html.Div(
            children=[
                dbc.Row(
                    html.H2("Decision Making for Climate Change", className="display-4 w-50 mx-auto text-center mb-3")
                ),
                dbc.Row(
                    html.P("Immediate action is required to combat climate change. \
                           The technology behind Cognizant NeuroAI brings automatic decision-making to the En-ROADS \
                           platform, a powerful climate change simulator. A decision-maker can be ready for any \
                           scenario: choosing an automatically generated policy that suits their needs best, with the \
                           ability to manually modify the policy and see its results. This tool is brought together \
                           under Project Resilience, a United Nations initiative to use AI for good.",
                           className="lead w-50 mx-auto text-center")
                ),
                dbc.Row(
                    style={"height": "60vh"}
                ),
                dbc.Row(
                    children=[
                        dbc.Col(
                            dbc.Button(
                                "Explore the IPCC SSPs",
                                id="ssp-button",
                                color="light"
                            ),
                            width={"size": 2, "offset": 4}
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Reach the Paris Agreement",
                                id="paris-button",
                                color="light"
                            ),
                            width={"size": 2}
                        )
                    ]
                ),
                dbc.Row(
                    style={"height": "10vh"}
                )
            ]
        )

        return div

    def register_callbacks(self, app):
        """
        Registers intro components callbacks
        """
        app.clientside_callback(
            """
            function(n_clicks) {
                if (n_clicks) {
                    document.getElementById('context').scrollIntoView({behavior: 'smooth'});
                }
                return "";
            }
            """,
            Output("ssp-button", "href"),
            [Input("ssp-button", "n_clicks")]
        )

        app.clientside_callback(
            """
            function(n_clicks) {
                if (n_clicks) {
                    document.getElementById('paris').scrollIntoView({behavior: 'smooth'});
                }
                return "";
            }
            """,
            Output("paris-button", "href"),
            [Input("paris-button", "n_clicks")]
        )
