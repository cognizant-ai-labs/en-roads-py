"""
Custom problem for PyMoo to optimize En-ROADS.
"""
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from sklearn.preprocessing import StandardScaler
import torch

from enroads_runner import EnroadsRunner
from evolution.candidate import Candidate
from evolution.outcomes.outcome_manager import OutcomeManager


class NNProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for En-ROADS in which we optimize the parameters of a neural network.
    """
    def __init__(self, context_df: pd.DataFrame, model_params: dict, actions: list[str], outcomes: dict[str, bool], batch_size=128):
        in_size = model_params["in_size"]
        hidden_size = model_params["hidden_size"]
        out_size = model_params["out_size"]
        num_params = (in_size + 1) * hidden_size + (hidden_size + 1) * out_size

        xl = np.array([-1 for _ in range(num_params)])
        xu = np.array([1 for _ in range(num_params)])
        super().__init__(n_var=num_params, n_obj=len(outcomes), n_ieq_constr=0, xl=xl, xu=xu)

        # To evaluate candidate solutions
        self.runner = EnroadsRunner()
        self.actions = [action for action in actions]
        self.outcomes = {k: v for k, v in outcomes.items()}
        self.model_params = model_params
        self.outcome_manager = OutcomeManager(list(self.outcomes.keys()))

        self.context_df = context_df
        context_ds = ContextDataset(context_df)
        self.context_dl = torch.utils.data.DataLoader(context_ds, batch_size=batch_size, shuffle=False)

    def params_to_context_actions_dicts(self, x: np.ndarray) -> list[dict[str, float]]:
        """
        Takes a set of candidate parameters, loads it into a candidate, prescribes actions for each context, and
        returns the contexts and actions as a list of dicts.
        """
        # Create candidate from params and pass contexts through it to get actions dicts for each context
        candidate = Candidate.from_pymoo_params(x, self.model_params, self.actions, self.outcomes)
        context_actions_dicts = []
        for batch in self.context_dl:
            context_tensor, _ = batch
            context_actions_dicts.extend(candidate.prescribe(context_tensor.to("mps")))

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


class ContextDataset(torch.utils.data.Dataset):
    """
    Simple PyTorch dataset that takes a pandas DataFrame, standardizes it, and converts it to a tensor.
    """
    def __init__(self, context_df: pd.DataFrame):
        scaler = StandardScaler()
        transformed = scaler.fit_transform(context_df)
        self.context_tensor = torch.Tensor(transformed)
        self.label_tensor = torch.zeros_like(self.context_tensor)

    def __len__(self):
        return len(self.context_tensor)
    
    def __getitem__(self, idx):
        return self.context_tensor[idx], self.label_tensor[idx]
