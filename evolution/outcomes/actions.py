"""
Outcome counting number of actions taken.
"""
from evolution.outcomes.outcome import Outcome

from enroadspy import load_input_specs


class ActionsOutcome(Outcome):
    """
    Counts number of actions taken by seeing which ones differ from the default.
    If a start time is after an end time, we reduce the action count by 2.
    Preprocesses the input specs to find the default values for each action.
    """
    def __init__(self):
        input_specs = load_input_specs()
        default_values = {}
        for action in input_specs["varId"]:
            row = input_specs[input_specs["varId"] == action].iloc[0]
            default_values[action] = row["defaultValue"]

        self.default_values = default_values

    def process_outcomes(self, actions_dict: dict[str, float], _) -> float:
        """
        Returns the number of actions taken. We do so by finding how many actions are not the default value.
        If we have a start time less than an end time, they must both not be default and therefore we can decrement our
        actions taken counter by 2 since the simulator interprets them as defaults.
        """
        actions_taken = 0
        for action, value in actions_dict.items():
            # Check if action is default
            if action in self.default_values and self.default_values[action] != value:
                actions_taken += 1
            # Check for end time < start time
            if "start_time" in action:
                end_time = action.replace("start_time", "end_time")
                if end_time in actions_dict and actions_dict[end_time] < value:
                    actions_taken -= 2
        return actions_taken
