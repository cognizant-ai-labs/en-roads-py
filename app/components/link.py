"""
Link Component.
"""
from dash import Input, Output, html, dcc

from generate_url import actions_to_url

class LinkComponent():
    """
    Component in charge of displaying the links to En-ROADS.
    """
    # html.A("View Candidate", id="cand-link", href="#", target="_blank", rel="noopener noreferrer")

    def __init__(self, cand_idxs: list[int]):
        self.cand_idxs = cand_idxs

    def create_link_div(self):
        """
        Creates content box containing links to the candidates to view.
        TODO: Make link unclickable while outcomes are loading.
        """
        div = html.Div(
            className="contentBox",
            children=[
                html.H2("View Prescriptor Actions", style={"textAlign": "center"}),
                html.Div(
                    style={"display": "grid", "grid-template-columns": "50% 50%"},
                    children=[
                        html.Div(
                            style={"grid-column": "1", "display": "flex", "justify-content": "flex-end", "width": "100%"},
                            children=[
                                dcc.Dropdown(self.cand_idxs, self.cand_idxs[0], id="link-select-dropdown", clearable=False)
                            ]
                        ),
                        html.Div(
                            style={"grid-column": "2"},
                            children= [
                                html.A("View Candidate", id="link-button", href="#", target="_blank", rel="noopener noreferrer")
                            ]
                        )
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        @app.callback(
            Output("link-button", "href"),
            Input("context-actions-store", "data"),
            Input("link-select-dropdown", "value")
        )
        def update_cand_links(context_actions_dicts: list[dict[str, float]], cand_idx: int):
            """
            Updates the candidate link when a specific candidate is selected.
            """
            cand_dict = context_actions_dicts[cand_idx]
            link = actions_to_url(cand_dict)
            return link

