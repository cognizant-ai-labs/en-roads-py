"""
Component showing the little intro blurb and the button to get started.
"""
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc

from app.components.component import Component


class IntroComponent(Component):
    """
    Title card component
    """
    def create_intro_paragraph(self) -> html.Div:
        """
        Creates the intro paragraph describing the project.
        """
        div = html.Div(
            children=[
                dbc.Row(
                    html.H2("Decision Making for Climate Change", className="display-2 w-50 mx-auto text-center mb-3")
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
                                "En-ROADS Climate Solutions Simulator",
                                href="https://www.climateinteractive.org/en-roads/",
                                style={"color": "black"}
                            ),
                            ". A decision-maker can be ready for any \
                            scenario: choosing an automatically generated policy that suits their needs best, with the \
                            ability to manually modify the policy and see its results. This tool is brought together \
                            under ",
                            html.A(
                                "Project Resilience",
                                href="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/\
                                    Pages/project-resilience.aspx",
                                style={"color": "black"}
                            ),
                            ", a collaboration between ",
                            html.A(
                                "Cognizant AI Labs",
                                href="https://www.cognizant.com/us/en/services/ai/ai-lab",
                                style={"color": "black"}
                            ),
                            " and the United Nations to use AI for good."
                        ],
                        className="lead w-50 mx-auto text-center"
                    )
                )
            ]
        )
        return div

    def create_div(self):
        intro = self.create_intro_paragraph()
        div = html.Div(
            children=[
                dbc.Button(id="intro-button", className="bi bi-info-circle", color="secondary"),
                dbc.Tooltip(target="intro-button", placement="bottom", children="Intro Slide"),
                dbc.Modal(
                    id="intro-modal",
                    is_open=True,
                    fullscreen=True,
                    children=[
                        dbc.ModalBody(
                            dbc.Card(
                                style={"height": "100%", "overflow": "hidden"},
                                children=[
                                    dbc.CardImg(
                                        src="https://upload.wikimedia.org/wikipedia/commons/5/54/Power_County_Wind_Farm_002.jpg", # noqa
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

    def register_callbacks(self, app):
        """
        Registers callback responsible for toggling the intro modal.
        """
        @app.callback(
            Output("intro-modal", "is_open"),
            [Input("intro-button", "n_clicks"), Input("close-intro-button", "n_clicks")],
            State("intro-modal", "is_open"),
            prevent_initial_call=True,
        )
        def toggle_intro_modal(_, __, is_open):
            """
            When a button is clicked, toggle the modal. We don't need to worry about tracking if it's open or not
            because if you can see the open button it's closed and vice versa.
            """
            return not is_open
