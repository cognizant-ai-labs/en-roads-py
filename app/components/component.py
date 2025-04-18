"""
Interface defining what a component should look like. This lets us easily and dynamically call all their callbacks
and then add them to the main app layout. Single components should register all their callbacks at once and only
have a single div that they render.
"""
from abc import ABC, abstractmethod

from dash import Dash, html


class Component(ABC):
    """
    Interface for components. They must register their callbacks and create their div.
    """
    @abstractmethod
    def create_div(self) -> html.Div:
        """
        Creates the div housing the component. This gets called by the main layout.
        """

    @abstractmethod
    def register_callbacks(self, app: Dash):
        """
        Registers the callbacks for the component. This gets called by the main app.
        """
