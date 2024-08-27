"""
References Component file
"""
from dash import html
import dash_bootstrap_components as dbc

class ReferencesComponent():
    """
    Component to handle references
    """
    def create_references_div(self):
        """
        Creates div displaying references
        """
        div = html.Div(
            className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
            children=[
                dbc.Container(
                    fluid=True,
                    className="py-3",
                    children=[
                        html.H2("References", className="text-center mb-2"),
                        html.P([
                            "For more info about Project Resilience, visit the ",
                            html.A("United Nations ITU Page", href="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/Pages/project-resilience.aspx")
                        ]),
                        html.P([
                            "The code for Project Resilience can be found on ",
                            html.A("Github", href="https://github.com/project-resilience")
                        ]),
                        html.P([
                            "The code for this project can be found ",
                            html.A("here", href="https://github.com/danyoungday/en-roads-py")
                        ]),
                        html.P("Keywan Riahi et al. \"The Shared Socioeconomic Pathways and their energy, land use, and \
                                greenhouse gas emissions implications: An overview,\" in Global Environmental Change, \
                                vol. 42, pp. 153-168, 2017."),
                        html.P("[En-ROADS Citation Here]"),
                        html.P(["Wind turbine image source: ", html.A("U.S. Department of Energy", href="https://www.flickr.com/people/37916456@N02")])
                    ]
                )
            ]
        )
        return div
