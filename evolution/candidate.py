from pathlib import Path

import pandas as pd
import torch

class Candidate():
    """
    Candidate class that points to a model on disk.
    """
    def __init__(self, cand_id: str, parents: list[str], model_params: dict, actions, outcomes):
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
            outputs = self.model.forward(x).detach().cpu().tolist()
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
        self.bias, self.scaler, self.binary_idxs, self.end_date_idxs = self.initialize_input_specs(actions)

    def initialize_input_specs(self, actions):
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True)
        bias = []
        scaler = []
        binary_idxs = []
        end_date_idxs = []
        for i, action in enumerate(actions):
            row = input_specs[input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                if "stop_time" in action:
                    end_date_idxs.append(i)
                    bias.append(0)
                    scaler.append(1)
                else:
                    bias.append(row["minValue"])
                    scaler.append(row["maxValue"] - row["minValue"])
            elif row["kind"] == "switch":
                bias.append(0)
                scaler.append(1)
                binary_idxs.append(i)
            else:
                raise ValueError(f"Unknown kind: {row['kind']}")
            
        bias = torch.tensor(bias, dtype=torch.float32)
        scaler = torch.tensor(scaler, dtype=torch.float32)
        return bias, scaler, binary_idxs, end_date_idxs
        # self.bias = torch.tensor([-15, 2024, 2024, 0, 2024, 0, 2024, 2024, 0])
        # self.scaler = torch.tensor([115, 76, 76, 100, 76, 100, 76, 76, 10])


    def snap_to_zero_one(self, scaled, i, output):
        output[i] = scaled[i] > 0.5

    def scale_end_time(self, scaled, i, output):
        output[i] = scaled[i] * (self.scaler[i-1] - (output[i-1] - self.bias[i-1])) + output[i-1]
        assert output[i] >= output[i-1], f"End time {output[i]} is less than start time {output[i-1]}"
        assert output[i] <= self.scaler[i-1] + self.bias[i-1], f"End time {output[i]} is greater than max {self.scaler[i-1] + self.bias[i-1]}"

    # pylint: disable=consider-using-enumerate
    def forward(self, x):
        nn_output = self.nn(x)
        self.scaler = self.scaler.to(x.device)
        self.bias = self.bias.to(x.device)
        scaled = nn_output * self.scaler + self.bias
        output = torch.zeros_like(nn_output)
        for i in range(len(nn_output)):
            if i in self.binary_idxs:
                self.snap_to_zero_one(scaled, i, output)
            elif i in self.end_date_idxs:
                self.scale_end_time(scaled, i, output)
            else:
                output[i] = scaled[i]
            
        return output
    
