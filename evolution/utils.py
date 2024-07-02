import pandas as pd

def modify_config(config: dict):
    # Set up context if not provided
    input_specs = pd.read_json("inputSpecs.jsonl", lines=True)
    actions = config["actions"]
    if len(config["context"]) == 0:
        adj_context = input_specs[~input_specs["varId"].isin(actions)]
        assert len(adj_context) == len(input_specs) - len(actions), f"Context is not the correct length. Expected {len(input_specs) - len(actions)}, got {len(adj_context)}."
        config["context"] = adj_context["varId"].tolist()

    # Set up model params
    config["model_params"]["in_size"] = len(config["context"])
    config["model_params"]["out_size"] = len(config["actions"])

    # Eval params
    config["eval_params"]["context"] = config["context"]
    config["eval_params"]["actions"] = config["actions"]
    config["eval_params"]["outcomes"] = config["outcomes"]

    return config