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
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from enroadspy import load_input_specs
from moo.problems.enroads_problem import EnroadsProblem
from moo.problems.nn_problem import NNProblem


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
    model_params = {"in_size": len(context_df.columns), "hidden_size": 16, "out_size": len(actions)}
    problem = NNProblem(context_df, model_params, actions, outcomes)
    return problem


def seed_default(problem: EnroadsProblem, actions: list[str], pop_size: int) -> np.ndarray:
    """
    Creates an initial population with one candidate with the default behavior, one with the minimum value, and one
    with the maximum value.
    """
    sampling = FloatRandomSampling()
    X = sampling(problem, pop_size).get("X")

    input_specs = load_input_specs()
    for i, action in enumerate(actions):
        row = input_specs[input_specs["varId"] == action].iloc[0]
        X[0, i] = row["defaultValue"]
        if row["kind"] == "slider":
            X[1, i] = row["minValue"]
            X[2, i] = row["maxValue"]
        else:
            X[1, i] = row["offValue"]
            X[2, i] = row["onValue"]

    return X


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
        alg_params = {}

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
