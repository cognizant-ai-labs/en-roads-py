"""
HeaderComponent class for the header of the main page of the app.
"""
from dash import html
import dash_bootstrap_components as dbc

from app.components.component import Component
from app.components.header.intro import IntroComponent
from app.components.header.references import ReferencesComponent
from app.components.header.video import VideoComponent


class HeaderComponent(Component):
    """
    Component housing the intro, references, and video components as well as displaying logos and the title.
    """
    def __init__(self):
        self.intro_component = IntroComponent()
        self.references_component = ReferencesComponent()
        self.video_component = VideoComponent()

    def create_div(self):
        div = html.Div(
            className="mb-5",
            children=dbc.Row(
                justify="between",
                align="center",
                children=[
                    dbc.Col(
                        width=1,
                        style={"display": "flex"},
                        children=[
                            html.Img(
                                src="https://companieslogo.com/img/orig/CTSH-82a8444b.png?t=1720244491",
                                className="img-thumbnail",
                                style={"height": "5rem", "width": "5rem", "border": "none", "outline": "none"}
                            ),
                            html.Img(
                                src="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/PublishingImages/Pages/default/Project%20Resilience%20Pink-Mauve.png", # noqa
                                className="img-thumbnail",
                                style={"height": "5rem", "width": "5rem", "border": "none", "outline": "none"}
                            )
                        ]
                    ),
                    dbc.Col(
                        width=6,
                        children=html.H1("Decision Making for Climate Change", className="text-center")
                    ),
                    dbc.Col(
                        width="auto",
                        children=dbc.Row([
                            dbc.Col(self.intro_component.create_div()),
                            dbc.Col(self.video_component.create_div()),
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
        self.video_component.register_callbacks(app)
