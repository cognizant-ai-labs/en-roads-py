import importlib
import inspect
from pathlib import Path

import pandas as pd

from esp.outcomes.outcome import Outcome


class OutcomeManager:
    def __init__(self, outcomes: list[str]):
        self.outcomes = [outcome for outcome in outcomes]
        self.outcome_dict = {}
        for file in Path("esp/outcomes").glob("*.py"):
            if file.stem in self.outcomes:
                module = importlib.import_module(f"esp.outcomes.{file.stem}")
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Outcome) and obj is not Outcome:
                        self.outcome_dict[file.stem] = obj()

    def process_outcomes(self, actions_dict: dict[str, float], outcomes_df: pd.DataFrame) -> dict[str, float]:
        """
        Processes outcomes from outcomes_df with all outcomes in outcome_dict.
        """
        results_dict = {}
        for outcome in self.outcomes:
            results_dict[outcome] = self.outcome_dict[outcome].process_outcomes(actions_dict, outcomes_df)
            if results_dict[outcome] == None:
                assert False

        return results_dict
