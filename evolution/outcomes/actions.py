"""
Outcome counting number of actions taken.
"""
from evolution.outcomes.outcome import Outcome

from enroadspy import load_input_specs, id_to_name, name_to_id


class ActionsOutcome(Outcome):
    """
    Counts number of actions taken by seeing which ones differ from the default.
    If a start time is after an end time, we reduce the action count by 2.
    Preprocesses the input specs to find the default values for each action.
    """
    def __init__(self):
        self.input_specs = load_input_specs()
        default_values = {}
        for action in self.input_specs["id"]:
            row = self.input_specs[self.input_specs["id"] == action].iloc[0]
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
            # NOTE: This is a bit of a hack, we should find a better way to detect start/end times
            action_name = id_to_name(action, self.input_specs)
            if "start time" in action_name.lower():
                end_time_name = action_name.replace("start", "end")
                end_time_action = name_to_id(end_time_name, self.input_specs)
                if end_time_action in actions_dict and actions_dict[end_time_name] < value:
                    actions_taken -= 2
        return actions_taken
