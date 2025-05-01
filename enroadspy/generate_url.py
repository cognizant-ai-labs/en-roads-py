"""
Script that allows us to visualize the results of our model on the En-ROADS website.
"""
import argparse
from pathlib import Path
import webbrowser

from presp.prescriptor import NNPrescriptorFactory
import torch
import yaml

from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.evaluation.evaluator import EnROADSEvaluator
from evolution.utils import process_config


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


def open_browser(results_dir: Path, cand_id: str, context_idx: int):
    """
    Loads seed from results_dir, loads context based on results_dir's config, runs context through model,
    then opens browser to en-roads with the prescribed actions and proper context.
    """
    with open(results_dir / "config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = process_config(config)

    # Get prescribed actions from model
    evaluator = EnROADSEvaluator(config["context"], config["actions"], config["outcomes"])

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    factory = NNPrescriptorFactory(EnROADSPrescriptor, config["model_params"], device, actions=config["actions"])
    candidate = factory.load(results_dir / cand_id.split("_")[0] / f"{cand_id}")

    context_tensor, context_vals = evaluator.context_dataset[context_idx]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    [actions_dict] = candidate.forward(context_tensor.to(device).unsqueeze(0))
    context_dict = evaluator.reconstruct_context_dicts([context_vals])[0]
    actions_dict.update(context_dict)

    url = actions_to_url(actions_dict)

    webbrowser.open(url)


def actions_to_url(actions_dict: dict[int, float]) -> str:
    """
    Converts an actions dict to a URL.
    """
    template = "https://en-roads.climateinteractive.org/scenario.html?v=24.6.0"
    for key, val in actions_dict.items():
        template += f"&p{key}={val}"

    return template


def generate_actions_dict(url: str):
    """
    Reverse-engineers an actions dict based on a given URL.
    """
    actions_dict = {}
    for param_val in url.split("&")[1:]:
        param, val = param_val.split("=")
        param = param[1:]
        actions_dict[int(param)] = float(val)

    return actions_dict


if __name__ == "__main__":
    main()
