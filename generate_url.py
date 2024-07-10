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
    open_browser(results_dir, cand_id)

def open_browser(results_dir, cand_id):
    config = json.load(open(results_dir / "config.json", encoding="utf-8"))

    # Get prescribed actions from model
    temp_dir = Path("temp_dir")
    evaluator = Evaluator(temp_dir, config["context"], config["actions"], config["outcomes"])
    candidate = Candidate.from_seed(results_dir / cand_id.split("_")[0] / f"{cand_id}.pt",
                                    config["model_params"],
                                    config["actions"],
                                    config["outcomes"])
    [torch_context] = next(iter(evaluator.torch_context))
    actions_dicts = candidate.prescribe(torch_context)
    actions_dict = actions_dicts[0]

    # Parse actions into format for URL
    input_specs = pd.read_json("inputSpecs.jsonl", lines=True)
    id_vals = {}
    for action, val in actions_dict.items():
        row = input_specs[input_specs["varId"] == action].iloc[0]
        id_vals[row["id"]] = val

    template = "https://en-roads.climateinteractive.org/scenario.html?v=24.6.0"
    for key, val in id_vals.items():
        template += f"&p{key}={val}"

    webbrowser.open(template)

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()