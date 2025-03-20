"""
Script used to run the evolution process.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from presp.prescriptor import NNPrescriptorFactory
from presp.evolution import Evolution
import yaml

from evolution.evaluation.evaluator import EnROADSEvaluator
from evolution.candidate import EnROADSPrescriptor
from evolution.utils import modify_config


def main():
    """
    Parses arguments, modifies config to reduce the amount of manual text added to it, then runs the evolution process.
    Prompts the user to overwrite the save path if it already exists.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    args = parser.parse_args()

    config_path = "evolution/configs/config.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # config = modify_config(config)
    print(json.dumps(config, indent=4))

    if Path(config["evolution_params"]["save_path"]).exists():
        inp = input("Save path already exists, do you want to overwrite? (y/n):")
        if inp.lower() == "y":
            shutil.rmtree(config["evolution_params"]["save_path"])
        else:
            print("Exiting")
            sys.exit()

    prescriptor_factory = NNPrescriptorFactory(EnROADSPrescriptor,
                                               model_params=config["model_params"],
                                               device=config["device"],
                                               actions=config["actions"])

    evaluator = EnROADSEvaluator(config["context"],
                                 config["actions"],
                                 config["outcomes"],
                                 batch_size=config["batch_size"],
                                 device=config["device"])

    evolution = Evolution(**config["evolution_params"], prescriptor_factory=prescriptor_factory, evaluator=evaluator)
    evolution.run_evolution()


if __name__ == "__main__":
    main()
