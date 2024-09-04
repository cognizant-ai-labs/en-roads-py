"""
Outcome interface.
"""
from abc import ABC, abstractmethod

import pandas as pd


class Outcome(ABC):
    """
    Outcome interface to be implemented by our custom outcomes.
    """
    @abstractmethod
    def process_outcomes(self, actions_dict: dict[str, float], outcomes_df: pd.DataFrame) -> float:
        """
        Takes in the actions dict which produced an outcomes dataframe returned by en-roads
        and processes them into a single float outcome.
        """
        raise NotImplementedError
