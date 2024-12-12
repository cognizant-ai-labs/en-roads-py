"""
Utility functions for experimentation
"""
import json
from pathlib import Path

import dill
import numpy as np
import torch

from evolution.candidate import Candidate
from moo.problems.enroads_problem import EnroadsProblem


class Experimenter:
    """
    Helper functions to be used in experimentation.
    """
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        with open(results_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        self.context = config["context"]
        self.actions = config["actions"]

        self.model_params = config["model_params"]
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def get_candidate_actions(self,
                              candidate: Candidate,
                              torch_context: torch.Tensor,
                              context_vals: torch.Tensor) -> dict[str, float]:
        """
        Gets actions from a candidate given a context
        """
        [actions_dict] = candidate.prescribe(torch_context.to(self.device).unsqueeze(0))
        context_dict = dict(zip(self.context, context_vals.tolist()))
        actions_dict.update(context_dict)
        return actions_dict

    def get_candidate_from_id(self, cand_id: str) -> Candidate:
        """
        Loads a candidate from an id.
        """
        cand_path = self.results_dir / cand_id.split("_")[0] / f"{cand_id}.pt"
        return Candidate.from_seed(cand_path, self.model_params, self.actions)
    

class PymooExperimenter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        with open(results_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        self.actions = config["actions"]
        self.outcomes = config["outcomes"]

        with open(results_dir / "results", 'rb') as f:
            self.res = dill.load(f)
            print("Loaded Checkpoint:", self.res)

        self.X = np.load(results_dir / "X.npy")
        self.F = np.load(results_dir / "F.npy")

        self.problem = EnroadsProblem(self.actions, self.outcomes)

        _, self.hist_F, _, _ = self.extract_history()

        self.outcomes_dfs = self.generate_outcomes_dfs()

    def extract_history(self):
        hist = self.res.history
        n_evals = []             # corresponding number of function evaluations\
        hist_F = []              # the objective space values in each generation
        hist_cv = []             # constraint violation in each generation
        hist_cv_avg = []         # average constraint violation in the whole population
        for algo in hist:

            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = algo.opt

            # store the least contraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
        return n_evals, hist_F, hist_cv, hist_cv_avg

    def generate_outcomes_dfs(self):
        outcomes_dfs = []
        for cand_idx in range(self.X.shape[0]):
            actions_dict = self.problem.params_to_actions_dict(self.X[cand_idx])
            outcomes_df = self.problem.runner.evaluate_actions(actions_dict)
            outcomes_dfs.append(outcomes_df)
        
        return outcomes_dfs
