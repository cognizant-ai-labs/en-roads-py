"""
Demo app for En-ROADS optimization results.
"""
import dash
from dash import html
import dash_bootstrap_components as dbc

from app.components.context import ContextComponent
from app.components.filter import FilterComponent
from app.components.header.header import HeaderComponent
from app.components.outcome import OutcomeComponent
from app.components.link import LinkComponent

from app.utils import EvolutionHandler

evolution_handler = EvolutionHandler()
context = evolution_handler.context
actions = evolution_handler.actions
metrics = evolution_handler.outcomes.keys()
# The candidates are sorted by rank then distance so the 'best' ones are the first 10
sample_idxs = list(range(10))

context_component = ContextComponent(context)
filter_component = FilterComponent(metrics)
header_component = HeaderComponent()
outcome_component = OutcomeComponent(evolution_handler)
link_component = LinkComponent(actions)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, "assets/styles.css"])
server = app.server
app.title = "Climate Change Decision Making"

context_component.register_callbacks(app)
filter_component.register_callbacks(app)
header_component.register_callbacks(app)
outcome_component.register_callbacks(app)
link_component.register_callbacks(app)

app.layout = html.Div(
    children=[
        dbc.Container(
            id="main-page",
            fluid=True,
            children=[
                header_component.create_div(),
                dbc.Row(outcome_component.create_div()),
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
                dbc.Row(link_component.create_div())
            ]
        )
    ]
)

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=4057, use_reloader=True, threaded=True)
