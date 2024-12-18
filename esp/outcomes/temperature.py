import pandas as pd

from esp.outcomes.outcome import Outcome


class Temperature(Outcome):
    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        return outcomes_df["Temperature change from 1850"].values[-1]
