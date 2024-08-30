"""
Python script to run optimization according to a config json file.
"""
import argparse
import json
from pathlib import Path
import shutil

import dill
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from moo.problems.novelty_problem import NoveltyProblem


def optimize(config: dict):
    """
    Running pymoo optimization according to our config file.
    """
    problem = NoveltyProblem(config["actions"], config["outcomes"], 3)

    algorithm = DE(
        pop_size=config["pop_size"]
    )

    res = minimize(problem,
                   algorithm,
                   get_termination("n_iter", config["n_generations"]),
                   seed=42,
                   save_history=True,
                   verbose=True)

    with open(Path(config["save_path"]) / "results", "wb") as f:
        dill.dump(res, f)


def main():
    """
    Main logic loading our config and running optimization.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if Path(config["save_path"]).exists():
        inp = input("Save path already exists, do you want to overwrite? (y/n):")
        if inp.lower() == "y":
            shutil.rmtree(config["save_path"])
        else:
            print("Exiting")
            exit()

    Path(config["save_path"]).mkdir(parents=True)
    with open(Path(config["save_path"]) / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    optimize(config)


if __name__ == "__main__":
    main()
