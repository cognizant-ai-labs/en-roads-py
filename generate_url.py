"""
Script that allows us to visualize the results of our model on the En-ROADS website.
"""
import argparse
import json
from pathlib import Path
import shutil
import webbrowser

import pandas as pd

from evolution.candidate import Candidate
from evolution.evaluation.evaluator import Evaluator


def main():
    """
    Takes in a candidate id and opens the browser to the enroads model with the candidate's actions.
    URL takes args as &p{id}={value}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str)
    parser.add_argument("--cand_id", required=True, type=str)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cand_id = args.cand_id
    open_browser(results_dir, cand_id, 0)


def open_browser(results_dir, cand_id, input_idx):
    """
    Loads seed from results_dir, loads context based on results_dir's config, runs context through model,
    then opens browser to en-roads with the prescribed actions and proper context.
    """
    config = json.load(open(results_dir / "config.json", encoding="utf-8"))

    # Get prescribed actions from model
    evaluator = Evaluator(config["context"], config["actions"], config["outcomes"])
    candidate = Candidate.from_seed(results_dir / cand_id.split("_")[0] / f"{cand_id}.pt",
                                    config["model_params"],
                                    config["actions"],
                                    config["outcomes"])
    context_tensor, context_vals = evaluator.context_dataset[input_idx]
    actions_dicts = candidate.prescribe(context_tensor.to("mps").unsqueeze(0))
    actions_dict = actions_dicts[0]
    context_dict = evaluator.reconstruct_context_dicts([context_vals])[0]
    actions_dict.update(context_dict)

    url = actions_to_url(actions_dict)

    webbrowser.open(url)


def actions_to_url(actions_dict: dict[str, float]) -> str:
    """
    Converts an actions dict to a URL.
    """
    # Parse actions into format for URL
    input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
    id_vals = {}
    for action, val in actions_dict.items():
        row = input_specs[input_specs["varId"] == action].iloc[0]
        id_vals[row["id"]] = val

    template = "https://en-roads.climateinteractive.org/scenario.html?v=24.6.0"
    for key, val in id_vals.items():
        template += f"&p{key}={val}"

    return template


def generate_actions_dict(url: str):
    """
    Reverse-engineers an actions dict based on a given URL.
    """
    input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
    actions_dict = {}
    for param_val in url.split("&")[1:]:
        param, val = param_val.split("=")
        param = param[1:]
        row = input_specs[input_specs["id"] == int(param)].iloc[0]
        actions_dict[row["varId"]] = float(val)

    return actions_dict


if __name__ == "__main__":
    main()