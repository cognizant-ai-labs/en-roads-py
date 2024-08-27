"""
Link Component.
"""
from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc

from generate_url import actions_to_url

class LinkComponent():
    """
    Component in charge of displaying the links to En-ROADS.
    """
    # html.A("View Candidate", id="cand-link", href="#", target="_blank", rel="noopener noreferrer")

    def __init__(self, cand_idxs: list[int]):
        self.cand_idxs = cand_idxs

    def create_button_group(self):
        """
        Creates button group to select candidate to link to.
        """
        button_group = dbc.ButtonGroup(
            className="justify-content-center w-50 mx-auto",
            children=[
                dbc.Button(str(cand_idx),
                           id=f"cand-link-{cand_idx}",
                           color="primary",
                           target="_blank",
                           rel="noopener noreferrer")
                for cand_idx in self.cand_idxs
            ]
        )
        return button_group

    def create_link_div(self):
        """
        Creates content box containing links to the candidates to view.
        TODO: Make link unclickable while outcomes are loading.
        """
        div = html.Div(
            className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
            children=[
                dbc.Container(
                    fluid=True,
                    className="py-3 d-flex flex-column",
                    children=[
                        html.H2("View/Modify Actions in En-ROADS", className="text-center mb-2"),
                        html.P("Choose a Prescriptor", className="text-center"),
                        self.create_button_group()
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        @app.callback(
            [Output(f"cand-link-{cand_idx}", "href") for cand_idx in self.cand_idxs],
            Input("context-actions-store", "data"),
        )
        def update_cand_links(context_actions_dicts: list[dict[str, float]]):
            """
            Updates the candidate link when a specific candidate is selected.
            """
            links = []
            for cand_idx in self.cand_idxs:
                cand_dict = context_actions_dicts[cand_idx]
                link = actions_to_url(cand_dict)
                links.append(link)
            return links

