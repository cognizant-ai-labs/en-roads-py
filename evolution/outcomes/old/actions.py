"""
Outcome counting number of actions taken.
"""
from evolution.outcomes.outcome import Outcome

from enroadspy import load_input_specs


class ActionsOutcome(Outcome):
    """
    Counts number of actions taken by seeing which ones differ from the default.
    """
    def __init__(self):
        self.input_specs = load_input_specs()

    def process_outcomes(self, actions_dict: dict[str, float], _) -> float:
        """
        Returns the number of actions taken. We do so by finding how many actions are not the default value.
        """
        actions_taken = 0
        for action, value in actions_dict.items():
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if value != row["defaultValue"]:
                actions_taken += 1
        return actions_taken
