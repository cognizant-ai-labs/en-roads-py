"""
Custom problem for PyMoo to optimize En-ROADS.
"""
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem

from enroads_runner import EnroadsRunner
from evolution.outcomes.outcome_manager import OutcomeManager


class EnroadsProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for En-ROADS.
    All sliders have an upper and lower bound defined by inputSpecs. All switches range between 0 and 1.
        Switches are snapped to their correct values at evaluation.
    Constraints are set on start and stop years such that start year is less than stop year.
    All outcomes are minimized so we have to pre and post process them.
    """
    def __init__(self, actions: list[str], outcomes: dict[str, bool]):
        self.input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        xl = np.zeros(len(actions))
        xu = np.ones(len(actions))
        switch_idxs = []
        switchl = []
        switchu = []
        self.start_year_idxs = set()
        for i, action in enumerate(actions):
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                xl[i] = row["minValue"]
                xu[i] = row["maxValue"]
                if "start_time" in action and "stop_time" in actions[i+1]:
                    self.start_year_idxs.add(i)
            else:
                switch_idxs.append(i)
                switchl.append(row["offValue"])
                switchu.append(row["onValue"])

        super().__init__(n_var=len(actions), n_obj=len(outcomes), n_ieq_constr=len(self.start_year_idxs), xl=xl, xu=xu)

        # To evaluate candidate solutions
        self.runner = EnroadsRunner()
        self.actions = [action for action in actions]
        self.outcomes = {k: v for k, v in outcomes.items()}
        self.outcome_manager = OutcomeManager(list(self.outcomes.keys()))

        # To parse switches
        self.switch_idxs = switch_idxs
        self.switchl = switchl
        self.switchu = switchu

    def params_to_actions_dict(self, x) -> dict[str, float]:
        """
        Converts the optimized parameters into an actions dict readable by our EnroadsRunner.
        """
        parsed = x.copy()
        parsed[self.switch_idxs] = np.where(parsed[self.switch_idxs] < 0.5, self.switchl, self.switchu)
        actions_dict = dict(zip(self.actions, parsed))
        return actions_dict

    def _evaluate(self, x, out, *args, **kwargs):
        actions_dict = self.params_to_actions_dict(x)
        outcomes_df = self.runner.evaluate_actions(actions_dict)
        results_dict = self.outcome_manager.process_outcomes(actions_dict, outcomes_df)

        # Flip the objective if we are maximizing it.
        f = []
        for outcome, minimize in self.outcomes.items():
            val = results_dict[outcome]
            if not minimize:
                val *= -1
            f.append(val)

        g = [x[idx] - x[idx+1] for idx in self.start_year_idxs]

        out["F"] = f
        out["G"] = g
