import pandas as pd

from evolution.outcomes.outcome import Outcome


class ActionsOutcome(Outcome):

    def __init__(self):
        self.input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)

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
