"""
Demo app for En-ROADS optimization results.
"""
import dash
from dash import html
import dash_bootstrap_components as dbc

from app.components.intro import IntroComponent
from app.components.context import ContextComponent
from app.components.filter import FilterComponent
from app.components.parallel import ParallelComponent
from app.components.outcome import OutcomeComponent
from app.components.link import LinkComponent
from app.components.references import ReferencesComponent
from app.utils import EvolutionHandler

evolution_handler = EvolutionHandler()
# The candidates are sorted by rank then distance so the 'best' ones are the first 10
sample_idxs = list(range(10))

intro_component = IntroComponent()
context_component = ContextComponent()
filter_component = FilterComponent()
parallel_component = ParallelComponent(evolution_handler.load_initial_metrics_df(),
                                       sample_idxs,
                                       evolution_handler.outcomes)
outcome_component = OutcomeComponent(evolution_handler, sample_idxs)
link_component = LinkComponent(sample_idxs)
references_component = ReferencesComponent()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, "assets/styles.css"])
server = app.server
app.title = "Climate Change Decision Making"

context_component.register_callbacks(app)
filter_component.register_callbacks(app)
parallel_component.register_callbacks(app)
outcome_component.register_callbacks(app)
link_component.register_callbacks(app)

# Layout of the app
app.layout = html.Div(
    children=[
        intro_component.create_intro_div(),
        context_component.create_context_div(),
        filter_component.create_filter_div(),
        parallel_component.create_parallel_div(),
        outcome_component.create_outcomes_div(),
        link_component.create_link_div(),
        references_component.create_references_div()
    ]
)

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=4057, use_reloader=True, threaded=True)
