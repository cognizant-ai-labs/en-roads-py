import pandas as pd

from evolution.outcomes.outcome import Outcome

class ZeroEmissionsOutcome(Outcome):
    def process_outcomes(self, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the year zero emissions are reached or 2100 if they are never reached.
        """
        emissions = outcomes_df["CO2 Equivalent Net Emissions"]
        zero_emissions = emissions[emissions <= 0]
        if zero_emissions.empty:
            return 2100
        else:
            return 1990 + zero_emissions.index[0]