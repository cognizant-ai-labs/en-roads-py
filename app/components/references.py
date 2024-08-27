"""
References Component file
"""
from dash import html

class ReferencesComponent():
    """
    Component to handle references
    """
    def create_references_div(self):
        """
        Creates div displaying references
        """
        div = html.Div(
            className="contentBox",
            children=[
                html.H2("References", style={"textAlign": "center"}),
                html.P([
                    "Code for this project can be found ",
                    html.A("here", href="https://github.com/danyoungday/en-roads-py")
                ]),
                    
                html.P("Keywan Riahi et al. \"The Shared Socioeconomic Pathways and their energy, land use, and \
                        greenhouse gas emissions implications: An overview,\" in Global Environmental Change, \
                        vol. 42, pp. 153-168, 2017."),
                html.P("[En-ROADS Citation Here]")
            ]
        )
        return div
