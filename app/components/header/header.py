"""
HeaderComponent class for the header of the main page of the app.
"""
from dash import html
import dash_bootstrap_components as dbc

from app.components.component import Component
from app.components.header.intro import IntroComponent
from app.components.header.references import ReferencesComponent
from app.components.header.tutorial import TutorialComponent
from app.components.header.video import VideoComponent


class HeaderComponent(Component):
    """
    Component housing the intro, references, and video components as well as displaying logos and the title.
    """
    def __init__(self):
        self.intro_component = IntroComponent()
        self.references_component = ReferencesComponent()
        self.tutorial_component = TutorialComponent()
        self.video_component = VideoComponent()

    def create_thumbnail(self, src: str, href: str) -> html.A:
        """
        Creates a uniform format image thumbnail for the header.
        """
        return html.A(
            href=href,
            children=html.Img(
                src=src,
                className="img-thumbnail",
                style={"height": "5rem", "width": "5rem", "border": "none", "outline": "none"}
            )
        )

    def create_div(self) -> html.Div:
        """
        Creates a row of logos, the title, and the icons for the various header components.
        We use bootstrap classes to center the title with the rest of the content.
        """
        thumbnails = html.Div(
            style={"display": "flex"},
            children=[
                self.create_thumbnail("https://companieslogo.com/img/orig/CTSH-82a8444b.png?t=1720244491",
                                      "https://www.cognizant.com/us/en/services/ai/ai-lab"),
                self.create_thumbnail("https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/PublishingImages/Pages/default/Project%20Resilience%20Pink-Mauve.png", # noqa
                                      "https://project-resilience.github.io/platform/")
            ]
        )

        title = html.Div(
            className="position-absolute start-50 translate-middle-x",
            children=html.H1("Decision Making for Climate Change", className="text-center")
        )

        icons = dbc.Row([
            dbc.Col(self.intro_component.create_div()),
            dbc.Col(self.tutorial_component.create_div()),
            dbc.Col(self.video_component.create_div()),
            dbc.Col(self.references_component.create_div())
        ])

        # We do bootstrap convention instead of dbc here because we want to use the start-50 translate-middle-x classes
        div = html.Div(
            className="mb-3",
            style={"border-bottom": "1px solid #ccc"},
            children=html.Div(
                className="d-flex justify-content-between align-items-center position-relative",
                children=[
                    thumbnails,
                    title,
                    icons
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
        self.video_component.register_callbacks(app)
