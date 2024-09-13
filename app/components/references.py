"""
References Component file
"""
from dash import html
import dash_bootstrap_components as dbc

from app.classes import JUMBOTRON, CONTAINER, HEADER


class ReferencesComponent():
    """
    Component to handle references
    """
    def create_references_div(self):
        """
        Creates div displaying references
        """
        div = html.Div(
            id="references",
            className=JUMBOTRON,
            children=[
                dbc.Container(
                    fluid=True,
                    className=CONTAINER[:-18],  # Left-align our references
                    children=[
                        html.H2("References", className=HEADER),
                        html.P([
                            "For more info about Project Resilience, visit the ",
                            html.A("United Nations ITU Page",
                                   href="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/Pages/project-resilience.\
                                    aspx")
                        ]),
                        html.P([
                            "The code for Project Resilience can be found on ",
                            html.A("Github", href="https://github.com/project-resilience")
                        ]),
                        html.P([
                            "The code for this project can be found ",
                            html.A("here", href="https://github.com/danyoungday/en-roads-py")
                        ]),
                        html.P(["Then En-ROADS model is developed by Climate Interactive. More information can be\
                               found here: ",
                               html.A("https://www.climateinteractive.org/en-roads/",
                                      href="https://www.climateinteractive.org/en-roads/")]),
                        html.P("Keywan Riahi et al. \"The Shared Socioeconomic Pathways and their energy, land use, \
                               and greenhouse gas emissions implications: An overview,\" in Global Environmental \
                               Change, vol. 42, pp. 153-168, 2017."),
                        html.P("IPCC, 2023: Climate Change 2023: Synthesis Report. Contribution of Working Groups I, \
                               II and III to the Sixth Assessment Report of the Intergovernmental Panel on Climate \
                               Change [Core Writing Team, H. Lee and J. Romero (eds.)]. IPCC, Geneva, Switzerland, \
                               pp. 35-115, doi: 10.59327/IPCC/AR6-9789291691647."),
                        html.P(["Wind turbine image source: ",
                                html.A("U.S. Department of Energy", href="https://www.flickr.com/people/37916456@N02")])
                    ]
                )
            ]
        )
        return div
