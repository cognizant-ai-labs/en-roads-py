"""
Candidate class to be used during evolution.
"""
import math
from pathlib import Path

import numpy as np
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
        self.model = NNPrescriptor(**model_params).to("mps")
        self.model.eval()

        self.input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        self.scaling_params = self.initialize_scaling_params(actions)

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

    def initialize_scaling_params(self, actions):
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

    def snap_to_zero_one(self, scaled: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
        """
        Takes switches and makes them binary by sigmoiding then snapping to 0 or 1
        :param scaled: Tensor of shape (batch_size, num_actions)
        """
        scaled[:,binary_mask] = (scaled[:,binary_mask] > 0.5).float()
        return scaled
    
    def scale_end_times(self, output: torch.Tensor, end_date_idxs: list[int], scaler: torch.Tensor, bias: torch.Tensor):
        """
        Scales end time based on start time's value.
        We do: end = start + end_logit * (scaler + bias - start)
        This has to be non in-place to not mess up the gradient calculations in seeding.
        """
        scaled = torch.zeros_like(output)
        for i in range(scaled.shape[1]):
            if i in end_date_idxs:
                scaled[:, i] = output[:, i-1] + output[:, i] * (scaler[i-1] + bias[i-1] - output[:, i-1])
            else:
                scaled[:, i] = output[:, i]
        return scaled

    def decode_torch_output(self, nn_output: torch.Tensor) -> list[list[float]]:
        """
        Scales and snaps the output of the model to the correct format.
        output is a tensor of shape (batch_size, num_actions).
        """
        bias, scaler, binary_mask, end_date_idxs, steps = self.scaling_params
        bias = bias.to(nn_output.device)
        scaler = scaler.to(nn_output.device)
        binary_mask = binary_mask.to(nn_output.device)
        steps = steps.to(nn_output.device)

        scaled = nn_output * scaler + bias
        snapped = self.snap_to_zero_one(scaled, binary_mask)
        end = self.scale_end_times(snapped, end_date_idxs, scaler, bias)
        end = end.detach().cpu().tolist()
        return end

    def fix_switch_values(self, actions_dict: dict[str, float]):
        """
        Sets the switch values from 0 to 1 to offValue and onValue
        """
        for action in actions_dict:
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if row["kind"] == "switch":
                actions_dict[action] = int(actions_dict[action])
                # Switch values are not necessarily 0/1
                if actions_dict[action] == 1:
                    actions_dict[action] = row["onValue"]
                else:
                    actions_dict[action] = row["offValue"]

    def clip_min_max(self, actions_dict: dict[str, float]):
        """
        Clips the values of the actions to be within min and max.
        """
        for action in actions_dict:
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                if actions_dict[action] < row["minValue"]:
                    actions_dict[action] = row["minValue"]
                elif actions_dict[action] > row["maxValue"]:
                    actions_dict[action] = row["maxValue"]

    def prescribe(self, x: torch.Tensor) -> list[dict[str, float]]:
        """
        Parses the output of our model so that we can use it in en-roads model.
        NOTE: We actually can pass float values and are ok with it.
        """
        with torch.no_grad():
            nn_outputs = self.model.forward(x)
        outputs = self.decode_torch_output(nn_outputs)
        actions_dicts = []
        for output in outputs:
            actions_dict = {action: value for action, value in zip(self.actions, output)}
            self.fix_switch_values(actions_dict)
            self.clip_min_max(actions_dict)
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
    def __init__(self, in_size, hidden_size, out_size):
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

    def forward(self, x):
        """
        Forward pass of neural network.
        Returns a tensor of shape (batch_size, num_actions).
        Values are scaled between 0 and 1.
        """
        nn_output = self.nn(x)
        return nn_output
    
