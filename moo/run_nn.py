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

from moo.problems.nn_problem import NNProblem


def optimize(config: dict):
    """
    Running pymoo optimization according to our config file.
    """

    context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
    context_df = context_df.drop(columns=["F", "scenario"])
    model_params = {"in_size": len(context_df.columns), "hidden_size": 16, "out_size": len(config["actions"])}
    problem = NNProblem(context_df, model_params, config["actions"], config["outcomes"])

    algorithm = NSGA2(
        pop_size=config["pop_size"],
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        survival=RankAndCrowding(crowding_func=config["crowding_func"]),
        eliminate_duplicates=True
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
