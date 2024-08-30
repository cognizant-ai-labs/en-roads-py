"""
Python script to run optimization according to a config json file.
"""
import argparse
import json
from pathlib import Path
import shutil

import dill
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from moo.problems.enroads_problem import EnroadsProblem


def seed_default(actions: list[str], pop_size: int) -> np.ndarray:
    """
    Creates an initial population with one candidate with the default behavior. The rest are randomly initialized.
    """
    X = np.random.random((pop_size, len(actions)))

    input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
    for i, action in enumerate(actions):
        row = input_specs[input_specs["varId"] == action].iloc[0]
        X[0, i] = row["defaultValue"]

    return X


def optimize(config: dict):
    """
    Running pymoo optimization according to our config file.
    """
    problem = EnroadsProblem(config["actions"], config["outcomes"])

    X0 = seed_default(config["actions"], config["pop_size"])

    algorithm = NSGA2(
        pop_size=config["pop_size"],
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        survival=RankAndCrowding(crowding_func=config["crowding_func"]),
        eliminate_duplicates=True,
        sampling=X0
    )

    res = minimize(problem,
                   algorithm,
                   get_termination("n_gen", config["n_generations"]),
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
