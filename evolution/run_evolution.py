"""
Script used to run the evolution process.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

# from presp.prescriptor import NNPrescriptorFactory
from presp.evolution import Evolution
import yaml

from evolution.evaluation.evaluator import EnROADSEvaluator
# from evolution.novelty import NoveltyEvaluator
# from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.candidates.direct import DirectFactory


def main():
    """
    Parses arguments, modifies config to reduce the amount of manual text added to it, then runs the evolution process.
    Prompts the user to overwrite the save path if it already exists.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(json.dumps(config, indent=4))

    save_path = Path(config["evolution_params"]["save_path"])
    if save_path.exists():
        inp = input("Save path already exists, do you want to overwrite? (y/n):")
        if inp.lower() == "y":
            shutil.rmtree(save_path)
        else:
            print("Exiting")
            sys.exit()

    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # prescriptor_factory = NNPrescriptorFactory(EnROADSPrescriptor,
    #                                            model_params=config["model_params"],
    #                                            device=config["device"],
    #                                            actions=config["actions"])
    prescriptor_factory = DirectFactory(config["actions"])

    evaluator = EnROADSEvaluator(context=config["context"],
                                 actions=config["actions"],
                                 outcomes=config["outcomes"],
                                 n_jobs=config["n_jobs"],
                                 batch_size=config["batch_size"],
                                 device=config["device"],
                                 decomplexify=config.get("decomplexify", False))
    # evaluator = NoveltyEvaluator(
    #     context=config["context"],
    #     actions=config["actions"]
    # )
    evolution = Evolution(**config["evolution_params"], prescriptor_factory=prescriptor_factory, evaluator=evaluator)
    evolution.run_evolution()


if __name__ == "__main__":
    main()
