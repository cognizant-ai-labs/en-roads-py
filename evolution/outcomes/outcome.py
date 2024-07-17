from abc import ABC, abstractmethod

import pandas as pd

class Outcome(ABC):
    """
    Outcome interface to be implemented by our custom outcomes.
    """
    @abstractmethod
    def process_outcomes(self, outcomes_df: pd.DataFrame) -> float:
        """
        Takes in the outcomes dataframe returned by en-roads and processes it into a single float.
        """
        raise NotImplementedError
