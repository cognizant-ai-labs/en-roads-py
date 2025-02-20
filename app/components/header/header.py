"""
HeaderComponent class for the header of the main page of the app.
"""
from dash import html
import dash_bootstrap_components as dbc

from app.components.component import Component
from app.components.header.intro import IntroComponent
from app.components.header.references import ReferencesComponent
from app.components.header.tutorial import TutorialComponent


class HeaderComponent(Component):
    """
    Component housing the intro, references, and video components as well as displaying logos and the title.
    """
    def __init__(self):
        self.intro_component = IntroComponent()
        self.references_component = ReferencesComponent()
        self.tutorial_component = TutorialComponent()

    def create_thumbnail(self, src):
        """
        Creates a uniform format image thumbnail for the header.
        """
        return html.Img(
            src=src,
            className="img-thumbnail",
            style={"height": "5rem", "width": "5rem", "border": "none", "outline": "none"}
        )

    def create_div(self):
        div = html.Div(
            className="mb-5",
            children=dbc.Row(
                className="width-100",
                justify="between",
                align="center",
                children=[
                    dbc.Col(
                        width="auto",
                        style={"display": "flex"},
                        children=[
                            self.create_thumbnail("https://companieslogo.com/img/orig/CTSH-82a8444b.png?t=1720244491"),
                            self.create_thumbnail("https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/PublishingImages/Pages/default/Project%20Resilience%20Pink-Mauve.png") # noqa
                        ]
                    ),
                    dbc.Col(
                        width="auto",
                        children=html.H1("Decision Making for Climate Change", className="text-center")
                    ),
                    dbc.Col(
                        width="auto",
                        children=dbc.Row([
                            dbc.Col(self.intro_component.create_div()),
                            dbc.Col(self.tutorial_component.create_div()),
                            dbc.Col(self.references_component.create_div())
                        ])
                    )
                ]
            )
        )
        return div

    def register_callbacks(self, app):
        """
        Registers the individual callbacks of the header components.
        """
        self.intro_component.register_callbacks(app)
        self.references_component.register_callbacks(app)
        self.tutorial_component.register_callbacks(app)
