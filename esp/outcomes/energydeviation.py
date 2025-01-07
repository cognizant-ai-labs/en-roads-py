import pandas as pd

from enroadspy.enroads_runner import EnroadsRunner
from esp.outcomes.outcome import Outcome


class EnergyDeviation(Outcome):
    """
    Computes the mean of square of difference between our energy and the baseline
    """
    def __init__(self):
        runner = EnroadsRunner()
        baseline_df = runner.evaluate_actions({})
        self.baseline_energy = baseline_df["Total Primary Energy Demand"]

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        return (outcomes_df["Total Primary Energy Demand"] - self.baseline_energy).pow(2).mean()
