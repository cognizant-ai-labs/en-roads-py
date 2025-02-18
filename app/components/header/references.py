"""
References Component file
"""
from dash import html, Input, Output
import dash_bootstrap_components as dbc

from app.components.component import Component


class ReferencesComponent(Component):
    """
    Component to handle references
    """
    def create_div(self):
        """
        Creates div displaying references
        """
        div = html.Div([
            dbc.Button(id="references-button", className="bi bi-book", color="secondary"),
            dbc.Modal(
                id="references-modal",
                is_open=False,
                fullscreen=True,
                children=[
                    dbc.ModalHeader(html.H2("References")),
                    dbc.ModalBody(
                        children=[
                            html.P([
                                "For more info about Project Resilience, visit the ",
                                html.A("Project Resilience Platform",
                                       href="project-resilience.github.io"),
                                " or the ",
                                html.A("United Nations ITU Page",
                                    href="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/Pages/project-resilience.aspx") # noqa
                            ]),
                            html.P([
                                "Project Resilience is a collaboration between the United Nations and Cognizant \
                                    AI Labs. More info about the lab can be found here: ",
                                html.A("https://www.cognizant.com/us/en/services/ai/ai-lab",
                                       href="https://www.cognizant.com/us/en/services/ai/ai-lab")
                            ]),
                            html.P([
                                "The code for Project Resilience can be found on ",
                                html.A("Github", href="https://github.com/project-resilience")
                            ]),
                            html.P([
                                "The code for this project can be found ",
                                html.A("here", href="https://github.com/cognizant-ai-labs/en-roads-py")
                            ]),
                            html.P(["Then En-ROADS model is developed by Climate Interactive. More information can be\
                                    found here: ",
                                    html.A("https://www.climateinteractive.org/en-roads/",
                                           href="https://www.climateinteractive.org/en-roads/")]),
                            html.P("Keywan Riahi et al. \"The Shared Socioeconomic Pathways and their energy, \
                                   land use, and greenhouse gas emissions implications: An overview,\" in Global \
                                   Environmental Change, vol. 42, pp. 153-168, 2017."),
                            html.P("IPCC, 2023: Climate Change 2023: Synthesis Report. Contribution of Working Groups \
                                   I, II and III to the Sixth Assessment Report of the Intergovernmental Panel on \
                                   Climate Change [Core Writing Team, H. Lee and J. Romero (eds.)]. IPCC, Geneva, \
                                   Switzerland, pp. 35-115, doi: 10.59327/IPCC/AR6-9789291691647."),
                            html.P(["Wind turbine image source: ",
                                    html.A("U.S. Department of Energy",
                                           href="https://www.flickr.com/people/37916456@N02")])
                        ]
                    )
                ]
            )
        ])
        return div

    def register_callbacks(self, app):
        """
        Registers the opening/closing of the modal.
        """
        @app.callback(
            Output("references-modal", "is_open"),
            Input("references-button", "n_clicks"),
            prevent_initial_call=True
        )
        def toggle_video_modal(_):
            return True
