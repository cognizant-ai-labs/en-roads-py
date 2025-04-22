"""
Data file containing various context datasets used within experiments.
"""
from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from torch.utils.data.dataset import Dataset


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
