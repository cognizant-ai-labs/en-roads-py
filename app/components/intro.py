"""
Component showing the little intro blurb and the arrow to get started.
"""
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc


class IntroComponent():
    """
    Title card component
    """

    def create_intro_paragraph(self):
        div = html.Div(
            children=[
                dbc.Row(
                    html.H2("Decision Making for Climate Change", className="display-4 w-50 mx-auto text-center mb-3")
                ),
                dbc.Row(
                    html.P(
                        [
                            "Immediate action is required to combat climate change. The technology behind ",
                            html.A(
                                "Cognizant NeuroAI",
                                href="https://www.cognizant.com/us/en/services/ai/ai-lab",
                                style={"color": "black"}
                            ),
                            " brings automatic decision-making to the ",
                            html.A(
                                "En-ROADS platform",
                                href="https://www.climateinteractive.org/en-roads/",
                                style={"color": "black"}
                            ),
                            ", a powerful climate change simulator. A decision-maker can be ready for any \
                            scenario: choosing an automatically generated policy that suits their needs best, with the \
                            ability to manually modify the policy and see its results. This tool is brought together \
                            under ",
                            html.A(
                                "Project Resilience",
                                href="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/\
                                    Pages/project-resilience.aspx",
                                style={"color": "black"}
                            ),
                            ", a United Nations initiative to use AI for good."
                        ],
                        className="lead w-50 mx-auto text-center"
                    )
                )
            ]
        )
        return div

    def create_intro_div(self):
        """
        Creates the intro title card describing the project.
        """
        intro = self.create_intro_paragraph()
        div = html.Div(
            children=[
                intro,
                dbc.Row(
                    style={"height": "70vh"}
                ),
                dbc.Row(
                    html.P("Get Started:", className="w-50 text-center mx-auto text-white h4")
                ),
                dbc.Row(
                    html.I(className="bi bi-arrow-up w-50 text-center mx-auto text-white h1")
                ),
                dbc.Row(
                    style={"height": "5vh"}
                )
            ]
        )

        return div
    
    def create_intro_div_big(self):
        intro = self.create_intro_paragraph()
        div = html.Div(
            children=[
                dbc.Button(id="intro-button", className="bi bi-wind", color="secondary"),
                dbc.Modal(
                    id="intro-modal",
                    is_open=True,
                    fullscreen=True,
                    children=[
                        dbc.ModalBody(
                            dbc.Card(
                                children=[
                                    dbc.CardImg(
                                        src="https://upload.wikimedia.org/wikipedia/commons/5/54/Power_County_Wind_Farm_002.jpg",
                                    ),
                                    dbc.CardImgOverlay(
                                        dbc.CardBody(
                                            children=[
                                                intro,
                                                dbc.Row(
                                                    style={"height": "5vh"}
                                                ),
                                                dbc.Row(
                                                    justify="center",
                                                    children=[
                                                        dbc.Col(
                                                            dbc.Button(
                                                                "Get Started",
                                                                id="close-intro-button",
                                                                color="secondary"
                                                            ),
                                                            width="auto"
                                                        )
                                                    ]
                                                )
                                            ]
                                        )
                                    )
                                ]
                            )
                        )
                    ]
                )
            ]
        )

        return div
    
    def register_callbacks_big(self, app):
        @app.callback(
            Output("intro-modal", "is_open"),
            [Input("intro-button", "n_clicks"), Input("close-intro-button", "n_clicks")],
            State("intro-modal", "is_open"),
            prevent_initial_call=True,
        )
        def toggle_intro_modal(n_open, n_close, is_open):
            return not is_open

