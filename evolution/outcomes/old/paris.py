"""
Paris agreement outcome implementation.
"""
import pandas as pd

from evolution.outcomes.outcome import Outcome


class ParisOutcome(Outcome):
    """
    Worst-case paris agreement goal.
    """
    def __init__(self):
        self.paris_start_emissions = 54.4789

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the amount of emissions greater than the threshold we are at in 2030 and 2050 or 0 if we are below.
        """
        emissions = outcomes_df["CO2 Equivalent Net Emissions"]
        mid_penalty = max(emissions.iloc[2030-1990] - self.paris_start_emissions * 0.55, 0)
        final_penalty = max(emissions.iloc[2050-1990], 0)
        return mid_penalty + final_penalty
