from enroadspy import load_input_specs
from esp.outcomes.outcome import Outcome


class ExtremeActions(Outcome):
    """
    Counts the number of actions taken where the value is closer to the min or max value than the default value.
    """
    def __init__(self):
        self.input_specs = load_input_specs()
        self.thresholds = {}  # Cache the thresholds for each action rather than re-computing them
        self.seen = set()

    def process_outcomes(self, actions_dict: dict[str, float], _) -> float:
        count = 0
        for action, value in actions_dict.items():
            # If this action hasn't been seen, cache it
            if action not in self.seen:
                spec = self.input_specs[self.input_specs["varId"] == action].iloc[0]
                if spec["kind"] == "slider":
                    lower = (spec["minValue"] + spec["defaultValue"]) / 2
                    upper = (spec["maxValue"] + spec["defaultValue"]) / 2
                    self.thresholds[action] = lower, upper
                self.seen.add(action)

            if action in self.thresholds and (value < self.thresholds[action][0] or value > self.thresholds[action][1]):
                count += 1
        return count
