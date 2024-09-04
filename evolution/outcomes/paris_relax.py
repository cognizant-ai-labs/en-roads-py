"""
A more relaxed Paris Agreement implementation.
"""
import pandas as pd

from evolution.outcomes.outcome import Outcome


class ParisRelaxOutcome(Outcome):
    """
    A more relaxed Paris Agreement goal.
    """
    def __init__(self):
        self.paris_goal_temp = 1.5

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the amount of emissions greater than the threshold we are at in 2030 and 2050 or 0 if we are below.
        """
        temp_change = outcomes_df["Temperature change from 1850"]
        penalty = max(temp_change.iloc[-1] - self.paris_goal_temp, 0)
        return penalty
