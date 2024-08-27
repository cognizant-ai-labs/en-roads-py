"""
Demo app for En-ROADS optimization results.
"""
import dash
from dash import html
import dash_bootstrap_components as dbc

from app.components.context import ContextComponent
from app.components.outcome import OutcomeComponent
from app.components.parallel import ParallelComponent
from app.components.link import LinkComponent
from app.components.references import ReferencesComponent
from app.utils import EvolutionHandler

evolution_handler = EvolutionHandler()
sample_idxs = list(range(10))

context_component = ContextComponent()
parallel_component = ParallelComponent(evolution_handler.load_initial_metrics_df(),
                                       sample_idxs,
                                       evolution_handler.outcomes)
outcome_component = OutcomeComponent(evolution_handler, sample_idxs)
link_component = LinkComponent(sample_idxs)
references_component = ReferencesComponent()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "assets/styles.css"])
app.title = "Climate Change Decision Making"

context_component.register_callbacks(app)
parallel_component.register_callbacks(app)
outcome_component.register_callbacks(app)
link_component.register_callbacks(app)

# Layout of the app
app.layout = html.Div(
    className="cognizant",
    style={"backgroundColor": "#f0f0f0", "margin": "0"},
    children=[
        html.H1("Climate Change Decision Making Page", style={"textAlign": "center"}),
        context_component.create_context_div(),
        parallel_component.create_parallel_div(),
        outcome_component.create_outcomes_div(),
        link_component.create_link_div(),
        references_component.create_references_div()
    ]
)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)