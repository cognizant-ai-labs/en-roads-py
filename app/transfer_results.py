"""
Take the final pareto front out of the population so that we save file space and time loading.
"""
import argparse
from pathlib import Path
import shutil

import pandas as pd
from presp.prescriptor import NNPrescriptorFactory
import yaml

from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.utils import process_config


def copy_subset_population(results_dir: Path, output_dir: Path):
    """
    Copies the final pareto front from the population to the output directory.
    """
    with open(results_dir / "config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = process_config(config)

    factory = NNPrescriptorFactory(EnROADSPrescriptor, config["model_params"], actions=config["actions"])
    population = factory.load_population(results_dir / "population")

    results_df = pd.read_csv(results_dir / "results.csv")
    final_pareto_df = results_df[(results_df["gen"] == results_df["gen"].max()) & (results_df["rank"] == 1)]
    final_ids = final_pareto_df["cand_id"].tolist()

    final_population = []
    for cand_id, candidate in population.items():
        if cand_id in final_ids:
            final_population.append(candidate)

    # Save them back
    factory.save_population(final_population, output_dir / "population")


def transfer(results_dir: Path, output_dir: Path):
    """
    Transfers results from one directory to another. This includes the config, results csv, and population.
    We also subset the population to only include the final pareto so that we save space and time loading.
    """
    if output_dir.exists() and output_dir.is_dir() and len(list(output_dir.iterdir())) > 0:
        inp = input(f"Output directory {output_dir} already exists. Do you want to overwrite it? (y/n):")
        if inp.lower() != "y":
            print("Exiting...")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(results_dir / "config.yml", output_dir / "config.yml")
    shutil.copy2(results_dir / "results.csv", output_dir / "results.csv")
    copy_subset_population(results_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer results from one directory to another.")
    parser.add_argument("--results_dir", type=str, help="Path to the results directory.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")

    args = parser.parse_args()

    transfer(Path(args.results_dir), Path(args.output_dir))
