"""
Trains seeds for the first generation of evolution using desired behavior.
"""
import argparse
import json
from pathlib import Path
import shutil
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from evolution.candidate import Candidate, NNPrescriptor, OutputParser
from evolution.evaluation.evaluator import Evaluator
from evolution.utils import modify_config
from enroadspy import load_input_specs
from enroadspy.generate_url import generate_actions_dict

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def train_seed(model_params: dict, dataset: Dataset, label: torch.Tensor, epochs=200, batch_size=1) -> NNPrescriptor:
    """
    Simple PyTorch training loop training a seed model with model_params using data from dataloader to match
    label label for epochs epochs.
    """
    label_tensor = label.to(DEVICE)
    model = NNPrescriptor(**model_params)
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with tqdm(range(epochs)) as pbar:
        for _ in pbar:
            avg_loss = 0
            n = 0
            for x, _ in dataloader:
                optimizer.zero_grad()
                x = x.to(DEVICE)
                output = model(x)
                loss = criterion(output, label_tensor.repeat(x.shape[0], 1))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                n += 1
            pbar.set_description(f"Avg Loss: {(avg_loss / n):.5f}")

    return model

def actions_to_label(actions: list[str], actions_dict: dict[str, float], output_parser: OutputParser) -> torch.Tensor:
    """
    Converts an actions dict to a label tensor. Actions is passed in to ensure the order is correct. The OutputParser
    unparses the actions dict into what the raw output of the model should be.
    """
    label = []
    for action in actions:
        value = actions_dict[action]
        label.append(value)
    parsed_label = torch.FloatTensor(label)
    unparsed_label = output_parser.unparse(parsed_label.unsqueeze(0))
    return unparsed_label


def create_default_labels(actions: list[str], output_parser: OutputParser) -> list[torch.Tensor]:
    """
    WARNING: Labels have to be added in the exact same order as the model.
    """
    input_specs = load_input_specs()
    categories = []
    for action in actions:
        possibilities = []
        row = input_specs[input_specs["varId"] == action].iloc[0]
        if row["kind"] == "slider":
            possibilities = [row["minValue"], row["maxValue"], row["defaultValue"]]
        elif row["kind"] == "switch":
            possibilities = [row["offValue"], row["onValue"], row["defaultValue"]]
        else:
            raise ValueError(f"Unknown kind {row['kind']}")
        categories.append(possibilities)

    combinations = [[possibilities[i] for possibilities in categories] for i in range(len(categories[0]))]
    labels = []
    for comb in combinations:
        actions_dict = dict(zip(actions, comb))
        label = actions_to_label(actions, actions_dict, output_parser)
        labels.append(label)
    return labels


def create_custom_labels(actions: list[str], seed_urls: list[str], output_parser: OutputParser) -> list[torch.Tensor]:
    """
    WARNING: Labels have to be added in the exact same order as the model.
    """
    input_specs = load_input_specs()
    actions_dicts = [generate_actions_dict(url) for url in seed_urls]
    labels = []
    for actions_dict in actions_dicts:
        # Fill actions dict with default values
        for action in actions:
            if action not in actions_dict:
                actions_dict[action] = input_specs[input_specs["varId"] == action].iloc[0]["defaultValue"]
        # Encode actions dict to tensor
        label = actions_to_label(actions, actions_dict, output_parser)
        labels.append(label)

    return labels


def create_seeds(model_params: dict,
                 context_ds: Dataset,
                 actions: list[str],
                 seed_urls: Optional[list[str]] = None,
                 epochs: Optional[int] = 1000) -> list[Candidate]:
    """
    Creates seed prescriptors for a given context dataset and actions. If seed_urls are provided, they are used to
    create custom labels for the seeds. Otherwise, we simply seed the default actions, min/off actions, and 
    max/on actions.
    """
    output_parser = OutputParser(actions, device=DEVICE)
    labels = create_default_labels(actions, output_parser)
    if seed_urls is not None:
        labels.extend(create_custom_labels(actions, seed_urls, output_parser))

    seeds = []
    for i, label in enumerate(labels):
        nn = train_seed(model_params, context_ds, label, epochs=epochs)
        candidate = Candidate(f"0_{i}", [], model_params, actions)
        candidate.model = nn
        seeds.append(candidate)

    return seeds
