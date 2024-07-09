from pathlib import Path

import pandas as pd
import torch

class Candidate():
    """
    Candidate class that points to a model on disk.
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

    @classmethod
    def from_seed(cls, path: Path, model_params: dict, actions, outcomes):
        cand_id = path.stem
        parents = []
        candidate = cls(cand_id, parents, model_params, actions, outcomes)
        candidate.model.load_state_dict(torch.load(path))
        return candidate
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def prescribe(self, x: torch.Tensor) -> dict:
        """
        Parses the output of our model so that we can use it in the AquaCrop model.
        """
        with torch.no_grad():
            outputs = self.model.forward(x).detach().cpu()
        return outputs

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
        self.bias, self.scaler, self.binary_mask, self.end_date_idxs, self.steps = self.initialize_input_specs(actions)

    def initialize_input_specs(self, actions):
        """
        Records information from inputSpecs that we need to parse our outputs.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True)
        bias = []
        scaler = []
        binary_mask = []
        end_date_idxs = []
        steps = []
        for i, action in enumerate(actions):
            row = input_specs[input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                steps.append(row["step"])
                if "stop_time" in action:
                    end_date_idxs.append(i)
                bias.append(row["minValue"])
                scaler.append(row["maxValue"] - row["minValue"])
                binary_mask.append(False)
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
        scaled[:,self.binary_mask] = (torch.sigmoid(scaled[:,self.binary_mask]) > 0.5).float()
        return scaled

    def swap_end_times(self, output):
        """
        Takes indices of end dates and swaps them with the start date if they're less.
        :param output: Tensor of shape (batch_size, num_actions)
        """
        for j in self.end_date_idxs:
            to_swap = output[:,j] < output[:,j-1]
            output[to_swap, j], output[to_swap, j-1] = output[to_swap, j-1], output[to_swap, j]
        return output

    def truncate_output_to_step(self, output):
        """
        Truncates outputs to the step thresholds from inputSpecs.
        :param output: Tensor of shape (batch_size, num_actions)
        """
        truncated = torch.floor(output / self.steps) * self.steps
        return truncated

    # pylint: disable=consider-using-enumerate
    def forward(self, x):
        """
        Forward pass of neural network.
        Then scaled and biased.
        Then switches snapped to 0 or 1.
        Then end dates swapped if they're less than start dates.
        Then truncated to step thresholds from inputSpecs.
        """
        nn_output = self.nn(x)

        self.scaler = self.scaler.to(x.device)
        self.bias = self.bias.to(x.device)
        self.steps = self.steps.to(x.device)
        self.binary_mask = self.binary_mask.to(x.device)

        scaled = nn_output * self.scaler + self.bias
        snapped = self.snap_to_zero_one(scaled)
        swapped = self.swap_end_times(snapped)
        return swapped
        # truncated = self.truncate_output_to_step(swapped)
        # return truncated
    
