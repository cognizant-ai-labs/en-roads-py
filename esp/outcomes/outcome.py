from abc import ABC, abstractmethod

import pandas as pd


class Outcome(ABC):
    @abstractmethod
    def process_outcomes(self, actions_dict: dict[str, float], outcomes_df: pd.DataFrame) -> float:
        """
        Processes actions and outcomes to get metric for candidate.
        """
