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

        self.scaled_context = torch.tensor(transformed, dtype=torch.float32)
        self.context = torch.tensor(context_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        return self.scaled_context[idx], self.context[idx]


class SSPDataset(ContextDataset):
    """
    Child of ContextDataset that loads the pre-generated SSP dataset.
    """
    def __init__(self):
        context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
        context_df = context_df.drop(columns=["F", "scenario"])
        super().__init__(context_df)
