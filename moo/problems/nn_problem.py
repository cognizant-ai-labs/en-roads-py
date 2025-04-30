"""
Custom problem for PyMoo to optimize En-ROADS.
"""
from typing import Optional

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import FloatRandomSampling
import torch
from torch.utils.data import DataLoader

from enroadspy.enroads_runner import EnroadsRunner
from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.evaluation.data import ContextDataset
from evolution.outcomes.outcome_manager import OutcomeManager
from evolution.seeding.train_seeds import create_seeds


class NNProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for En-ROADS in which we optimize the parameters of a neural network.
    """
    def __init__(self,
                 context_df: pd.DataFrame,
                 model_params: list[dict],
                 actions: list[str],
                 outcomes: dict[str, bool],
                 batch_size: int = 128,
                 device: str = "cpu"):

        num_params = 0
        for layer in model_params:
            if layer["type"] == "linear":
                num_params += (layer["in_features"] + 1) * layer["out_features"]

        xl = np.array([-1 for _ in range(num_params)])
        xu = np.array([1 for _ in range(num_params)])
        super().__init__(n_var=num_params, n_obj=len(outcomes), n_ieq_constr=0, xl=xl, xu=xu)

        # To evaluate candidate solutions
        self.runner = EnroadsRunner()
        self.actions = list(actions)
        self.outcomes = dict(outcomes.items())
        self.model_params = model_params
        self.outcome_manager = OutcomeManager(list(self.outcomes.keys()))

        self.context_df = context_df
        self.context_ds = ContextDataset(context_df)
        self.batch_size = batch_size
        self.device = device

    def params_to_context_actions_dicts(self, x: np.ndarray) -> list[dict[str, float]]:
        """
        Takes a set of candidate parameters, loads it into a candidate, prescribes actions for each context, and
        returns the contexts and actions as a list of dicts.
        """
        # Create candidate from params and pass contexts through it to get actions dicts for each context
        candidate = EnROADSPrescriptor.from_pymoo_params(x, self.model_params, self.actions)
        context_actions_dicts = []
        context_dl = DataLoader(self.context_ds, batch_size=self.batch_size, shuffle=False)
        for batch, _ in context_dl:
            context_actions_dicts.extend(candidate.forward(batch.to(self.device)))

        # Attaches correct context to each action dict
        assert len(context_actions_dicts) == len(self.context_df)
        for actions_dict, (_, row) in zip(context_actions_dicts, self.context_df.iterrows()):
            context_dict = row.to_dict()
            actions_dict.update(context_dict)

        return context_actions_dicts

    def run_enroads(self, context_actions_dicts: list[dict[str, float]]) -> list[pd.DataFrame]:
        """
        Takes a list of context + actions dicts and runs enroads for each, returning a list of outcomes_dfs.
        """
        outcomes_dfs = []
        for context_actions_dict in context_actions_dicts:
            outcomes_df = self.runner.evaluate_actions(context_actions_dict)
            outcomes_dfs.append(outcomes_df)

        return outcomes_dfs

    def _evaluate(self, x, out, *args, **kwargs):
        context_actions_dicts = self.params_to_context_actions_dicts(x)
        outcomes_dfs = self.run_enroads(context_actions_dicts)

        # Process outcomes into metrics
        results = []
        for context_actions_dict, outcomes_df in zip(context_actions_dicts, outcomes_dfs):
            results_dict = self.outcome_manager.process_outcomes(context_actions_dict, outcomes_df)
            results.append(results_dict)

        # For each outcome, take the mean over all contexts, then negate if we are maximizing
        f = []
        for outcome, minimize in self.outcomes.items():
            outcome_val = np.mean([result[outcome] for result in results])
            if not minimize:
                outcome_val *= -1
            f.append(outcome_val)

        out["F"] = f
        out["G"] = []


def candidate_to_params(candidate: EnROADSPrescriptor) -> np.ndarray:
    """
    Takes a candidate and flattens its parameters into a 1d numpy array
    """
    state_dict = candidate.model.state_dict()

    linear_idxs = []
    for i, layer in enumerate(candidate.model_params):
        if layer["type"] == "linear":
            linear_idxs.append(i)

    params = []
    for i in linear_idxs:
        params.append(state_dict[f"{i}.weight"].flatten())
        params.append(state_dict[f"{i}.bias"].squeeze())
    params = torch.cat(params).cpu().numpy()
    return params


def seed_nn(problem: NNProblem, pop_size: int, seed_urls: Optional[list[str]] = None, epochs=1000) -> np.ndarray:
    """
    Seeds the neural network problem by creating candidates from seed URLs as well as default seeding behavior, then
    converts PyTorch model parameters into a flattened numpy array.
    """
    print(f"Seeding problem for {epochs} epochs using {len(seed_urls) if seed_urls else 0} custom seeds...")
    candidates = create_seeds(problem.model_params, problem.context_ds, problem.actions, seed_urls, epochs)
    seed_params = np.array([candidate_to_params(candidate) for candidate in candidates])

    sampling = FloatRandomSampling()
    X = sampling(problem, pop_size).get("X")
    X[:len(seed_params)] = seed_params
    print(f"Created {len(seed_params)} seed candidates.")
    return X
