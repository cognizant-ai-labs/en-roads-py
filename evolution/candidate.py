"""
Candidate class to be used during evolution.
"""
import math
from pathlib import Path

import pandas as pd
import torch

class Candidate():
    """
    Candidate class that holds the model and stores evaluation and sorting information for evolution.
    Model can be persisted to disk.
    """
    def __init__(self, cand_id: str, parents: list[str], model_params: dict, actions: list[str], outcomes: dict[str, bool]):
        self.cand_id = cand_id
        self.actions = actions
        self.outcomes = outcomes
        self.metrics = {}

        self.parents = parents
        self.rank = None
        self.distance = None

        # Model
        self.model_params = model_params
        self.model = NNPrescriptor(actions=actions, **model_params).to("mps")
        self.model.eval()

        self.input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)

    @classmethod
    def from_seed(cls, path: Path, model_params: dict, actions, outcomes):
        """
        Loads PyTorch seed from disk.
        """
        cand_id = path.stem
        parents = []
        candidate = cls(cand_id, parents, model_params, actions, outcomes)
        candidate.model.load_state_dict(torch.load(path))
        return candidate
    
    def save(self, path: Path):
        """
        Saves PyTorch state dict to disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def truncate_actions(self, actions_dict: dict[str, float]):
        """
        Formats actions based on requirements by en-roads.
        Truncates floats to correct number of decimals.
        We do this here instead of in torch to make sure they aren't messed up by floating point conversion.
        Switches are snapped to 0 or 1 then set accordingly to the correct on/off value.
        TODO: This assumes steps are powers of 10
        TODO: This actually rounds, not truncates
        """
        for action in actions_dict:
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                # Round to correct number of decimals
                decimals = -1 * math.log10(row["step"])
                assert decimals == int(decimals), f"{row['step']} is not a power of 10"
                decimals = int(decimals)
                # If our step is 10 or more, we truncate it to next lowest step
                if decimals < 0:
                    actions_dict[action] = int(actions_dict[action] * row["step"]) // row["step"]
                # If our step is 1, we truncate to the nearest integer
                elif decimals == 0:
                    actions_dict[action] = int(actions_dict[action])
                # If our step is <1, we round to the correct number of decimals
                else:
                    actions_dict[action] = round(actions_dict[action], decimals)
                # Clip to max/min if we go over/under due to rounding
                if actions_dict[action] < row["minValue"]:
                    actions_dict[action] = row["minValue"]
                if actions_dict[action] > row["maxValue"]:
                    actions_dict[action] = row["maxValue"]
            elif row["kind"] == "switch":
                actions_dict[action] = int(actions_dict[action])
                # Switch values are not necessarily 0/1
                if actions_dict[action] == 1:
                    actions_dict[action] = row["onValue"]
                else:
                    actions_dict[action] = row["offValue"]
            else:
                raise ValueError(f"Unknown kind: {row['kind']}")

    def validate_actions(self, actions_dict: dict[str, float]):
        """
        Validates actions are valid.
        TODO: This is pretty inefficient right now.
        TODO: Not all switches are 0,1
        """
        for action in actions_dict:
            assert action in self.actions, f"{action} not found in actions."
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                assert row["minValue"] <= actions_dict[action] <= row["maxValue"], \
                    f"Value {actions_dict[action]} not in range [{row['minValue']}, {row['maxValue']}]."
                if "stop_time" in action:
                    stop_time = actions_dict[action]
                    start_time = actions_dict[action.replace("stop", "start")]
                    assert stop_time >= start_time, \
                        f"{action}: {stop_time} < {action.replace('stop', 'start')}: {start_time}."
            elif row["kind"] == "switch":
                assert actions_dict[action] in [row["offValue"], row["onValue"]], \
                    f"Value {actions_dict[action]} not in [{row['offValue']}, {row['onValue']}]."
            else:
                raise ValueError(f"Unknown kind: {row['kind']}")

        return True

    def prescribe(self, x: torch.Tensor) -> list[dict[str, float]]:
        """
        Parses the output of our model so that we can use it in en-roads model.
        """
        with torch.no_grad():
            outputs = self.model.forward(x).detach().cpu().tolist()
        actions_dicts = []
        for output in outputs:
            actions_dict = {action: value for action, value in zip(self.actions, output)}
            self.truncate_actions(actions_dict)
            self.validate_actions(actions_dict)
            actions_dicts.append(actions_dict)
        return actions_dicts

    def record_state(self):
        """
        Records metrics as well as seed and parents for reconstruction.
        """
        state = {
            "cand_id": self.cand_id,
            "parents": self.parents,
            "rank": self.rank,
            "distance": self.distance,
        }
        for outcome in self.outcomes:
            state[outcome] = self.metrics[outcome]
        return state

    def __str__(self):
        return f"Candidate({self.cand_id})"
    
    def __repr__(self):
        return f"Candidate({self.cand_id})"

class NNPrescriptor(torch.nn.Module):
    """
    Torch neural network that the candidate wraps around.
    """
    def __init__(self, in_size, hidden_size, out_size, actions):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size),
            torch.nn.Sigmoid()
        )

        # Orthogonal initialization
        for layer in self.nn:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)

        # Set up input specs so our inputs are forced into a format the model can use
        self.bias, self.scaler, self.binary_mask, self.end_date_idxs, self.steps, = self.initialize_input_specs(actions)

    def initialize_input_specs(self, actions):
        """
        Records information from inputSpecs that we need to parse our outputs.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        bias = []
        scaler = []
        binary_mask = []
        end_date_idxs = []
        steps = []
        for i, action in enumerate(actions):
            row = input_specs[input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                steps.append(row["step"])
                binary_mask.append(False)
                # TODO: This assumes the start time always comes immediately before stop time
                if "stop_time" in action:
                    end_date_idxs.append(i)
                    bias.append(0)
                    scaler.append(1)
                else:
                    bias.append(row["minValue"])
                    scaler.append(row["maxValue"] - row["minValue"])
            elif row["kind"] == "switch":
                steps.append(1)
                bias.append(0)
                scaler.append(1)
                binary_mask.append(True)
            else:
                raise ValueError(f"Unknown kind: {row['kind']}")
            
        bias = torch.tensor(bias, dtype=torch.float32)
        scaler = torch.tensor(scaler, dtype=torch.float32)
        steps = torch.tensor(steps, dtype=torch.float32)
        binary_mask = torch.tensor(binary_mask, dtype=torch.bool)
        return bias, scaler, binary_mask, end_date_idxs, steps

    def snap_to_zero_one(self, scaled):
        """
        Takes switches and makes them binary by sigmoiding then snapping to 0 or 1
        :param scaled: Tensor of shape (batch_size, num_actions)
        """
        scaled[:,self.binary_mask] = (scaled[:,self.binary_mask] > 0.5).float()
        return scaled
    
    def scale_end_times(self, output):
        """
        Scales end time based on start time's value.
        We do: end = start + end_logit * (scaler + bias - start)
        This has to be non in-place to not mess up the gradient calculations in seeding.
        """
        scaled = torch.zeros_like(output)
        for i in range(scaled.shape[1]):
            if i in self.end_date_idxs:
                scaled[:, i] = output[:, i-1] + output[:, i] * (self.scaler[i-1] + self.bias[i-1] - output[:, i-1])
            else:
                scaled[:, i] = output[:, i]
        return scaled

    # pylint: disable=consider-using-enumerate
    def forward(self, x):
        """
        Forward pass of neural network.
        Then scaled and biased.
        Then switches snapped to 0 or 1.
        Then scales end dates based on start dates.
        """
        nn_output = self.nn(x)

        self.scaler = self.scaler.to(x.device)
        self.bias = self.bias.to(x.device)
        self.steps = self.steps.to(x.device)
        self.binary_mask = self.binary_mask.to(x.device)

        scaled = nn_output * self.scaler + self.bias
        snapped = self.snap_to_zero_one(scaled)
        end = self.scale_end_times(snapped)
        return end
    
