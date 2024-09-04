"""
Trains seeds for the first generation of evolution using desired behavior.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evolution.candidate import NNPrescriptor
from evolution.evaluation.evaluator import Evaluator
from evolution.utils import modify_config
from enroadspy import load_input_specs
from enroadspy.generate_url import generate_actions_dict

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def train_seed(epochs: int, model_params: dict, seed_path: Path, dataloader: DataLoader, label: torch.Tensor):
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
    torch.save(model.state_dict(), seed_path)


def encode_action_labels(actions: list[str], actions_dict: dict[str, float]) -> torch.Tensor:
    """
    Encodes actions in en-roads format to torch format to be used in the model.
    Min/max scales slider variables and sets switches to 0 or 1 based on off/on.
    """
    input_specs = load_input_specs()
    label = []
    for action in actions:
        value = actions_dict[action]
        row = input_specs[input_specs["varId"] == action].iloc[0]
        if row["kind"] == "slider":
            value = (value - row["minValue"]) / (row["maxValue"] - row["minValue"])
            label.append(value)
        elif row["kind"] == "switch":
            label.append(1 if value == row["onValue"] else 0)
        else:
            raise ValueError(f"Unknown kind {row['kind']}")

    return torch.tensor(label, dtype=torch.float32)


def create_default_labels(actions: list[str]):
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
        label = encode_action_labels(actions, actions_dict)
        labels.append(label)
    return labels


def create_custom_labels(actions: list[str], seed_urls: list[str]):
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
        label = encode_action_labels(actions, actions_dict)
        labels.append(label)

    return labels


def main():
    """
    Main logic for training seeds.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    config = modify_config(config)
    seed_params = config["seed_params"]
    seed_dir = Path(seed_params["seed_path"])

    if seed_dir.exists():
        inp = input("Seed path already exists, do you want to overwrite? (y/n):")
        if inp.lower() == "y":
            shutil.rmtree(seed_dir)
        else:
            print("Exiting")
            exit()

    seed_dir.mkdir(parents=True, exist_ok=True)

    evaluator_params = config["eval_params"]
    evaluator = Evaluator(**evaluator_params)
    context_dataloader = evaluator.context_dataloader
    model_params = config["model_params"]
    print(model_params)

    labels = create_default_labels(config["actions"])
    # Add custom seed URLs
    if "seed_urls" in seed_params and len(seed_params["seed_urls"]) > 0:
        labels.extend(create_custom_labels(config["actions"], seed_params["seed_urls"]))

    torch.random.manual_seed(42)
    for i, label in enumerate(labels):
        print(f"Training seed 0_{i}.pt")
        train_seed(int(seed_params["epochs"]),
                   model_params,
                   seed_dir / f"0_{i}.pt",
                   context_dataloader,
                   label)


if __name__ == "__main__":
    main()
