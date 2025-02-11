import dash_bootstrap_components as dbc
from dash import Input, Output, html

class VideoComponent:
    """
    Component storing the initial video to be played demonstrating the demo
    """
    def create_video_div(self):
        """
        Creates the div housing the modal that shows the video.
        """
        div = html.Div(
            children=[
                dbc.Button(id="video-button", className="bi bi-film", color="secondary"),
                dbc.Modal(
                    id="video-modal",
                    is_open=False,
                    fullscreen=True,
                    children=[
                        dbc.ModalHeader(),
                        dbc.ModalBody(
                            html.Iframe(
                                style={"height": "90vh", "width": "100%"},
                                src="https://www.youtube.com/embed/h4emhArlQ0s?autoplay=1&mute=1&controls=0"
                            )
                        )
                    ]
                )
            ]
        )
        return div
    
    def register_callbacks(self, app):
        """
        Registers video callbacks.
        """
        @app.callback(
            Output("video-modal", "is_open"),
            Input("video-button", "n_clicks"),
            prevent_initial_call=True
        )
        def toggle_video_modal(n_clicks):
            return True
