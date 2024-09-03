"""
Utility functions to be used throughout the evolution module.
"""
from enroadspy import load_input_specs


def modify_config(config: dict):
    """
    Sets up our config because we're too lazy to write everything down multiple times.
    If we don't specify a context we use all the variables besides the actions.
    We construct the model input and output sizes from the context and actions.
    We set up the eval params with the context, actions, and outcomes.
    """
    # Set up context if not provided
    input_specs = load_input_specs()
    actions = config["actions"]
    if len(config["context"]) == 0:
        adj_context = input_specs[~input_specs["varId"].isin(actions)]
        assert len(adj_context) == len(input_specs) - len(actions), \
            f"Context is not the correct length. Expected {len(input_specs) - len(actions)}, got {len(adj_context)}."
        config["context"] = adj_context["varId"].tolist()

    # Set up model params
    config["model_params"]["in_size"] = len(config["context"])
    config["model_params"]["out_size"] = len(config["actions"])

    # Eval params
    config["eval_params"]["context"] = config["context"]
    config["eval_params"]["actions"] = config["actions"]
    config["eval_params"]["outcomes"] = config["outcomes"]

    return config
