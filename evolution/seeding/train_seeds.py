"""
Trains seeds for the first generation of evolution using desired behavior.
"""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from evolution.candidate import NNPrescriptor
from evolution.evaluation.evaluator import Evaluator
from evolution.utils import modify_config


def train_seed(epochs: int, model_params: dict, seed_path: Path, dataloader: DataLoader, label: torch.Tensor):
    """
    Simple PyTorch training loop training a seed model with model_params using data from dataloader to match
    label label for epochs epochs.
    """
    label_tensor = label.to("mps")
    model = NNPrescriptor(**model_params)
    model.to("mps")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()
    with tqdm(range(epochs)) as pbar:
        for _ in pbar:
            avg_loss = 0
            n = 0
            for x, _ in dataloader:
                optimizer.zero_grad()
                x = x.to("mps")
                output = model(x)
                loss = criterion(output, label_tensor.repeat(x.shape[0], 1))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                n += 1
            pbar.set_description(f"Avg Loss: {(avg_loss / n):.5f}")
    torch.save(model.state_dict(), seed_path)

def create_labels(actions: list[str]):
    """
    WARNING: Labels have to be added in the exact same order as the model.
    """
    input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
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
        torch_comb = torch.tensor(comb, dtype=torch.float32)
        labels.append(torch_comb)
    return labels

def main():
    """
    Main logic for training seeds.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument("--epochs", type=int, help="Epochs.", default=250)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    config = modify_config(config)

    if Path(config["seed_path"]).exists():
        inp = input("Seed path already exists, do you want to overwrite? (y/n):")
        if inp.lower() == "y":
            shutil.rmtree(config["seed_path"])
        else:
            print("Exiting")
            exit()

    evaluator_params = config["eval_params"]
    evaluator = Evaluator(**evaluator_params)
    context_dataloader = evaluator.context_dataloader
    model_params = config["model_params"]
    model_params["actions"] = config["actions"]
    print(model_params)
    seed_dir = Path(config["seed_path"])
    seed_dir.mkdir(parents=True, exist_ok=True)

    labels = create_labels(config["actions"])
    torch.random.manual_seed(42)
    for i, label in enumerate(labels):
        print(f"Training seed 0_{i}.pt")
        train_seed(args.epochs, model_params, seed_dir / f"0_{i}.pt", context_dataloader, label)

if __name__ == "__main__":
    main()
