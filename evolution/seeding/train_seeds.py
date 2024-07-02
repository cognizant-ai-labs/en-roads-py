import argparse
import itertools
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from evolution.candidate import NNPrescriptor
from evolution.evaluation.evaluator import Evaluator
from evolution.utils import modify_config

class CustomDS(Dataset):
    def __init__(self, torch_context: list[torch.Tensor]):
        self.x = torch_context

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

def train_seed(epochs: int, model_params: dict, seed_path: Path, torch_context: torch.tensor, label: torch.tensor):
    ds = CustomDS([torch_context])
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    label_tensor = label.to("mps")
    model = NNPrescriptor(**model_params)
    model.to("mps")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()
    with tqdm(range(epochs)) as pbar:
        for _ in pbar:
            avg_loss = 0
            for x in dl:
                optimizer.zero_grad()
                x = x.to("mps").squeeze()
                output = model(x)
                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            pbar.set_description(f"Avg Loss: {(avg_loss / len(ds)):.5f}")
    torch.save(model.state_dict(), seed_path)

def create_labels():
    """
    WARNING: Labels have to be added in the exact same order as the model.
    """
    categories = []
    categories.append([[-15], [100]])
    categories.append([[2024, 2024], [2024, 2100]])
    categories.append([[0], [100]])
    categories.append([[2024], [2100]])
    categories.append([[0], [100]])
    categories.append([[2024, 2024], [2024, 2100]])
    categories.append([[0], [10]])

    combinations = list(itertools.product(*categories))
    labels = []
    for comb in combinations:
        torch_comb = torch.tensor([item for category in comb for item in category], dtype=torch.float32)
        labels.append(torch_comb)
    return labels

def main():

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
    torch_context = evaluator.torch_context
    model_params = config["model_params"]
    print(model_params)
    seed_dir = Path(config["seed_path"])
    seed_dir.mkdir(parents=True, exist_ok=True)

    labels = create_labels()
    torch.random.manual_seed(42)
    for i, label in enumerate(labels):
        print(f"Training seed 0_{i}.pt")
        train_seed(args.epochs, model_params, seed_dir / f"0_{i}.pt", torch_context, label)

if __name__ == "__main__":
    main()
