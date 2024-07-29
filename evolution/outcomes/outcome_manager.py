import pandas as pd

from evolution.outcomes.actions import ActionsOutcome
from evolution.outcomes.average_cost import AverageCostOutcome
from evolution.outcomes.energy_change import EnergyChangeOutcome
from evolution.outcomes.enroads import EnroadsOutcome
from evolution.outcomes.near_cost import NearCostOutcome
from evolution.outcomes.paris import ParisOutcome
from evolution.outcomes.revenue import RevenueOutcome
from evolution.outcomes.zero_emissions import ZeroEmissionsOutcome


class OutcomeManager():
    """
    Manages many outcomes at once for the evaluator.
    """
    def __init__(self, outcomes: list[str]):
        outcome_dict = {}
        for outcome in outcomes:
            if outcome == "Actions taken":
                outcome_dict[outcome] = ActionsOutcome()
            elif outcome == "Average Adjusted cost of energy per GJ":
                outcome_dict[outcome] = AverageCostOutcome()
            elif outcome == "Average Percent Energy Change":
                outcome_dict[outcome] = EnergyChangeOutcome()
            elif outcome == "Cost of energy next 10 years":
                outcome_dict[outcome] = NearCostOutcome()
            elif outcome == "Year Zero Emissions Reached":
                outcome_dict[outcome] = ZeroEmissionsOutcome()
            elif outcome == "Government net revenue below zero":
                outcome_dict[outcome] = RevenueOutcome()
            elif outcome == "Emissions Above Paris Agreement":
                outcome_dict[outcome] = ParisOutcome()
            else:
                outcome_dict[outcome] = EnroadsOutcome(outcome)

        self.outcome_dict = outcome_dict

    def process_outcomes(self, actions_dict: dict[str, float], outcomes_df: pd.DataFrame) -> dict[str, float]:
        """
        Processes outcomes from outcomes_df with all outcomes in outcome_dict.
        """
        results_dict = {}
        for outcome, outcome_obj in self.outcome_dict.items():
            results_dict[outcome] = outcome_obj.process_outcomes(actions_dict, outcomes_df)

        return results_dict
