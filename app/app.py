"""
Demo app for En-ROADS optimization results.
"""
import json

import dill
import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd

from enroads_runner import EnroadsRunner
from evolution.outcomes.outcome_manager import OutcomeManager
from generate_url import actions_to_url
from moo.problems.nn_problem import NNProblem

from app.components.context import ContextComponent
from app.components.outcome import OutcomeComponent
from app.components.parallel import ParallelComponent
from app.components.link import LinkComponent
from app.utils import EvolutionHandler

evolution_handler = EvolutionHandler()
sample_idxs = list(range(10))

context_component = ContextComponent()
parallel_component = ParallelComponent(evolution_handler.load_initial_metrics_df(),
                                       sample_idxs,
                                       evolution_handler.outcomes)
outcome_component = OutcomeComponent(evolution_handler, sample_idxs)
link_component = LinkComponent(sample_idxs)

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Climate Change Decision Making"

context_component.register_callbacks(app)
outcome_component.register_callbacks(app)
link_component.register_callbacks(app)

# Layout of the app
app.layout = html.Div([
    html.H1("Climate Change Decision Making Page", style={"textAlign": "center"}),
    parallel_component.create_parallel_div(),
    context_component.create_context_div(),
    outcome_component.create_outcomes_div(),
    link_component.create_link_div()
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)