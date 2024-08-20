from dash import Input, Output, html

from generate_url import actions_to_url

class LinkComponent():
    # html.A("View Candidate", id="cand-link", href="#", target="_blank", rel="noopener noreferrer")

    def __init__(self, cand_idxs: list[int]):
        self.cand_idxs = cand_idxs

    def create_link_div(self):
        div = html.Div(id="links")
        return div

    def register_callbacks(self, app):
        @app.callback(
            Output("links", "children"),
            Input("context-actions-store", "data")
        )
        def update_cand_links(context_actions_dicts: list[dict[str, float]]):
            """
            Updates the candidate link when a specific candidate is selected.
            """
            links = []
            for cand_idx in self.cand_idxs:
                ca_dict = context_actions_dicts[cand_idx]
                link = actions_to_url(ca_dict)
                links.append(html.A(f"View Candidate {cand_idx}",
                                    href=link,
                                    target="_blank",
                                    rel="noopener noreferrer"))
            
            return links
