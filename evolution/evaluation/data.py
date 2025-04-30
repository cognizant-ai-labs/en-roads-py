"""
Data file containing various context datasets used within experiments.
"""
import itertools
from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataset import Dataset

from enroadspy import load_input_specs


class ContextDataset(Dataset):
    """
    Custom PyTorch dataset that takes a pandas DataFrame, standardizes it, then converts to tensor.
    """
    def __init__(self, context_df: pd.DataFrame, scaler: Optional[StandardScaler] = None):
        """
        NOTE: scaler doesn't have to be a StandardScaler, but sklearn does not have a base scaler class.
        """
        if scaler:
            self.scaler = scaler
            transformed = self.scaler.transform(context_df)
        else:
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(context_df)

        self.scaled_context = torch.Tensor(transformed)
        self.context = torch.Tensor(context_df.values)

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        return self.scaled_context[idx], self.context[idx]


class SSPDataset(Dataset):
    """
    Wrapper around ContextDataset that loads the SSP context data.
    """
    def __init__(self):
        context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
        context_df = context_df.drop(columns=["F", "scenario"])
        self.context_dataset = ContextDataset(context_df)

    def __len__(self):
        return len(self.context_dataset)

    def __getitem__(self, idx):
        return self.context_dataset[idx]


class ConstraintDataset(Dataset):
    """
    Dataset
    """
    def __init__(self, context: list[str]):
        input_specs = load_input_specs()
        context_constraints = []
        for cont in context:
            row = input_specs[input_specs["varId"] == cont].iloc[0]

            min_value, default_value, max_value = row["minValue"], row["defaultValue"], row["maxValue"]

            # Get the narrowest possible constraint + the full range
            constraint = ()
            dividers = row["rangeDividers"]
            for i, divider in enumerate(dividers):
                if divider > default_value:
                    low_norm = (dividers[i-1] - min_value) / (max_value - min_value)
                    high_norm = (divider - min_value) / (max_value - min_value)
                    constraint = (low_norm, high_norm)

            context_constraints.append([constraint, (0, 1)])

        # Get all combinations across contexts
        all_constraints = list(itertools.product(*context_constraints))
        # Convert constraints to tensor
        constraint_tensors = []
        for constraint in all_constraints:
            constraint_tensor = []
            for context_constraint in constraint:
                constraint_tensor.extend(context_constraint)
            constraint_tensor = torch.tensor(constraint_tensor, dtype=torch.float32)
            constraint_tensors.append(constraint_tensor)

        self.constraint_tensors = torch.stack(constraint_tensors)

    def __len__(self):
        return len(self.constraint_tensors)

    def __getitem__(self, idx):
        return self.constraint_tensors[idx]
