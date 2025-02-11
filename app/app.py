"""
Demo app for En-ROADS optimization results.
"""
import dash
from dash import html
import dash_bootstrap_components as dbc

from app.components.intro import IntroComponent
from app.components.context import ContextComponent
from app.components.filter import FilterComponent
from app.components.outcome import OutcomeComponent
from app.components.link import LinkComponent
from app.components.references import ReferencesComponent
from app.components.video import VideoComponent
from app.utils import EvolutionHandler

evolution_handler = EvolutionHandler()
actions = evolution_handler.actions
metrics = evolution_handler.outcomes.keys()
# The candidates are sorted by rank then distance so the 'best' ones are the first 10
sample_idxs = list(range(10))

intro_component = IntroComponent()
context_component = ContextComponent()
filter_component = FilterComponent(metrics)
outcome_component = OutcomeComponent(evolution_handler)
link_component = LinkComponent(sample_idxs, actions)
references_component = ReferencesComponent()
video_component = VideoComponent()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, "assets/styles.css"])
server = app.server
app.title = "Climate Change Decision Making"

intro_component.register_callbacks(app)
context_component.register_callbacks(app)
filter_component.register_callbacks(app)
outcome_component.register_callbacks(app)
link_component.register_callbacks(app)
video_component.register_callbacks(app)

app.layout = html.Div(
    children=[
        dbc.Container(
            id="main-page",
            style={"height": "100vh"},
            fluid=True,
            children=[
                dbc.Row(
                    className="mb-5",
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
                                    src="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/PublishingImages/Pages/default/Project%20Resilience%20Pink-Mauve.png",
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
                            style={"display": "flex"},
                            children=[
                                intro_component.create_div(),
                                video_component.create_div()
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    children=[
                        outcome_component.create_div()
                    ]
                ),
                dbc.Row(
                    className="mb-5",
                    children=[
                        dbc.Col(
                            context_component.create_div()
                        ),
                        dbc.Col(
                            filter_component.create_div()
                        )
                    ]
                ),
                dbc.Row(
                    justify="center",
                    children=dbc.Col(
                        html.A(dbc.Button("3. Examine Individual Policy"), href="#link-page"),
                        width={"size": 3, "offset": 1}
                    )
                )
            ]
        ),
        dbc.Container(
            id="link-page",
            style={"height": "100vh"},
            fluid=True,
            children=[
                dbc.Row([
                    link_component.create_div()
                ])
            ]
        )
    ]
)

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=4057, use_reloader=True, threaded=True)
