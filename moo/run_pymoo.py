"""
Python script to run optimization according to a config json file.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import dill
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from moo.problems.enroads_problem import EnroadsProblem, seed_default
from moo.problems.nn_problem import NNProblem, seed_nn


def create_default_problem(actions: list[str], outcomes: dict[str, bool]) -> EnroadsProblem:
    """
    Create a default EnroadsProblem instance.
    """
    return EnroadsProblem(actions, outcomes)


def create_nn_problem(actions: list[str], outcomes: dict[str, bool]) -> NNProblem:
    """
    Creates problem that uses neural network with context.
    TODO: Make the context file selectable.
    """
    context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
    context_df = context_df.drop(columns=["F", "scenario"])
    model_params = [
        {"type": "linear", "in_features": len(context_df.columns), "out_features": 16},
        {"type": "tanh"},
        {"type": "linear", "in_features": 16, "out_features": len(actions)},
        {"type": "sigmoid"}
    ]    
    problem = NNProblem(context_df, model_params, actions, outcomes)
    return problem


def optimize(config: dict, nn: bool):
    """
    Running pymoo optimization according to our config file.
    """
    if not nn:
        problem = create_default_problem(config["actions"], config["outcomes"])
        X0 = seed_default(problem, config["actions"], config["pop_size"])
        alg_params = {"sampling": X0}
    else:
        problem = create_nn_problem(config["actions"], config["outcomes"])
        X0 = seed_nn(problem, config["pop_size"], config.get("seed_urls", None), epochs=config.get("seed_epochs", 1000))
        alg_params = {"sampling": X0}

    algorithm = NSGA2(
        pop_size=config["pop_size"],
        crossover=SBX(prob=0.9, eta=15),
        # crossover=UniformCrossover(),
        mutation=PM(eta=20),
        survival=RankAndCrowding(crowding_func=config["crowding_func"]),
        eliminate_duplicates=True,
        **alg_params
    )

    res = minimize(problem,
                   algorithm,
                   get_termination("n_gen", config["n_generations"]),
                   seed=42,
                   save_history=True,
                   verbose=True)

    with open(Path(config["save_path"]) / "results", "wb") as f:
        dill.dump(res, f)

    np.save(Path(config["save_path"]) / "X.npy", res.pop.get("X"))
    np.save(Path(config["save_path"]) / "F.npy", res.pop.get("F"))


def main():
    """
    Main logic loading our config and running optimization.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument("--nn", action="store_true", help="Use neural network with context")
    args = parser.parse_args()

    nn = args.nn
    if nn:
        print("Running Neural Network Context Problem")

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if Path(config["save_path"]).exists():
        inp = input("Save path already exists, do you want to overwrite? (y/n):")
        if inp.lower() == "y":
            shutil.rmtree(config["save_path"])
        else:
            print("Exiting")
            sys.exit()

    Path(config["save_path"]).mkdir(parents=True)
    with open(Path(config["save_path"]) / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    optimize(config, nn)


if __name__ == "__main__":
    main()
