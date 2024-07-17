"""
Class to hold the context for the model. Called by the Evaluator during setup.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

class ContextDataset(Dataset):
    """
    Dataset holding the context for the model.
    Returns the context as a tensor and the corresponding dict to pass into the simulator.
    """
    def generate_default_df(self, context: list[str]) -> pd.DataFrame:
        """
        Generates the default context, which is just a single row of the default values for all the context variables.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        data = input_specs[["varId", "defaultValue"]]
        data = data[data["varId"].isin(context)]
        rotated = data.set_index("varId").T
        return rotated

    def generate_new_zero_carbon_df(self) -> pd.DataFrame:
        """
        Generates our new zero carbon breakthrough context. We simply have 3 values corresponding to the different
        breakthrough levels.
        """
        data = {
            "_new_tech_breakthrough_setting": [0, 1, 2]
        }
        return pd.DataFrame(data)

    def __init__(self, context: list[str]):
        if context == ["_new_tech_breakthrough_setting"]:
            df = self.generate_new_zero_carbon_df()
            self.context_vals = df.values
        else:
            df = self.generate_default_df(context)
            self.context_vals = np.array([[] for _ in range(len(df))])

        # Scale data down to [0, 1] for the PyTorch tensor
        for col in df.columns:
            scaler = MinMaxScaler()
            if len(df[col]) == 1:
                df[col] = scaler.fit_transform(df[col].values.reshape(1, -1))
            else:
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        self.tensor_context = torch.tensor(df.values, dtype=torch.float32)
        
    def __getitem__(self, idx):
        return self.tensor_context[idx], self.context_vals[idx]

    def __len__(self):
        return len(self.tensor_context)
