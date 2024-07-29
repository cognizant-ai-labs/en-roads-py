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
    
    def generate_renewable_df(self, context: list[str], n=10, seed=42) -> pd.DataFrame:
        """
        Generates our renewables breakthrough context.
        """
        rng = np.random.default_rng(seed)
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        data = {}
        for col in context:
            default_val = input_specs[input_specs["varId"] == col]["defaultValue"].iloc[0]
            min_val = input_specs[input_specs["varId"] == col]["minValue"].iloc[0]
            max_val = input_specs[input_specs["varId"] == col]["maxValue"].iloc[0]
            data[col] = [default_val] + rng.uniform(min_val, max_val, n-1).tolist()
        
        data_df = pd.DataFrame(data)
        return data_df

    def __init__(self, context: list[str]):
        breakthrough_cols = ["_breakthrough_cost_reduction_renewables",
                             "_breakthrough_success_year_renewables",
                             "_breakthrough_cost_reduction_elec_hydrogen",
                             "_breakthrough_success_year_elec_hydrogen",
                             "_breakthrough_cost_reduction_for_storage",
                             "_breakthrough_success_year_for_storage"]

        if context == ["_new_tech_breakthrough_setting"]:
            df = self.generate_new_zero_carbon_df()
            self.context_vals = df[context].values
        
        elif context == breakthrough_cols:
            df = self.generate_renewable_df(context)
            self.context_vals = df[context].values

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
