"""
Trains seeds for the first generation of evolution using desired behavior.
"""
from argparse import ArgumentParser
from pathlib import Path
import shutil
from typing import Optional

from presp.prescriptor import NNPrescriptorFactory
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.candidates.direct import DirectPrescriptor, DirectFactory
from evolution.candidates.output_parser import OutputParser
from evolution.evaluation.data import SSPDataset
from enroadspy import load_input_specs
from enroadspy.generate_url import generate_actions_dict


def train_seed(model_params: dict,
               actions: list[str],
               dataset: Dataset,
               label: torch.Tensor,
               epochs: int = 200,
               batch_size: int = 1,
               device: str = "cpu") -> EnROADSPrescriptor:
    """
    Simple PyTorch training loop training a seed model with model_params using data from dataloader to match
    label label for epochs epochs.
    """
    label_tensor = label.to(device)
    presc = EnROADSPrescriptor(model_params, actions)
    model = presc.model
    model.to(device)
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
                x = x.to(device)
                output = model(x)
                loss = criterion(output, label_tensor.repeat(x.shape[0], 1))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                n += 1
            pbar.set_description(f"Avg Loss: {(avg_loss / n):.5f}")

    return presc


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
    Creates 3 seeds: one where all min actions are taken, one where all max actions are taken, and one where
    all default values are taken.
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


def create_seeds(model_params: list[dict],
                 context_ds: Dataset,
                 actions: list[str],
                 seed_urls: Optional[list[str]] = None,
                 epochs: Optional[int] = 1000,
                 device: str = "cpu") -> list[EnROADSPrescriptor]:
    """
    Creates seed prescriptors for a given context dataset and actions. If seed_urls are provided, they are used to
    create custom labels for the seeds. Otherwise, we simply seed the default actions, min/off actions, and
    max/on actions.
    """
    output_parser = OutputParser(actions, device=device)
    labels = create_default_labels(actions, output_parser)
    if seed_urls is not None:
        labels.extend(create_custom_labels(actions, seed_urls, output_parser))

    seeds = []
    for i, label in enumerate(labels):
        candidate = train_seed(model_params, actions, context_ds, label, epochs=epochs)
        candidate.cand_id = f"0_{i}"
        seeds.append(candidate)

    return seeds


def create_direct_seeds(actions: list[str], seed_urls: Optional[list[str]] = None) -> list[DirectPrescriptor]:
    """
    Creates seed models for direct evolution by directly turning the label into a genome.
    """
    output_parser = OutputParser(actions)
    labels = create_default_labels(actions, output_parser)
    if seed_urls is not None:
        labels.extend(create_custom_labels(actions, seed_urls, output_parser))

    seeds = []
    for i, label in enumerate(labels):
        candidate = DirectPrescriptor(actions)
        candidate.genome = label
        candidate.cand_id = f"1_{i}"
        seeds.append(candidate)

    return seeds


def main():
    """
    Main logic for seed training. Trains direct seeds or neural network seeds depending on the config file.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file to use to train seeds")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Handle if the seed directory already exists
    seed_dir = Path(config["evolution_params"]["seed_dir"])
    if seed_dir.exists():
        delete = input("Seed directory already exists. Would you like to overwrite it? (Y/n):")
        if delete.lower() != "y":
            print("Exiting...")
            return
        else:
            print(f"Overwiting seeds in {seed_dir}")
            shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Actually train seeds
    if "model_params" in config:
        context_ds = SSPDataset()
        seeds = create_seeds(config["model_params"],
                             context_ds,
                             config["actions"],
                             config.get("seed_urls", None),
                             config.get("epochs", 1000),
                             config["device"])

        factory = NNPrescriptorFactory(EnROADSPrescriptor,
                                       model_params=config["model_params"],
                                       device=config["device"],
                                       actions=config["actions"])

    elif len(config["context"]) == 0:
        seeds = create_direct_seeds(config["actions"], config["seed_urls"])
        # Save seeds to disk with factory
        factory = DirectFactory(config["actions"])

    else:
        raise ValueError("Model params must be present or we have no context")

    for seed in seeds:
        factory.save(seed, seed_dir / (seed.cand_id + ".pt"))


if __name__ == "__main__":
    main()
