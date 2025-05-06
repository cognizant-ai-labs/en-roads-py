"""
Script to transfer the results from evolution to the app.
NOTE: You must reorder the outcomes in the config file to match the order manually for now:
    1. Temperature change 2. Cost of energy 3. Government spending 4. Energy use
"""
from argparse import ArgumentParser
from pathlib import Path
import shutil

import pandas as pd
import yaml


def main(results_dir: str, output_dir: str):
    """
    Copies the final Pareto front, config file, and model weights from the results directory to the output.
    Clears the output directory if it already exists.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        inp = input("Output directory already exists. Overwite? (Y/n):")
        if inp.lower() != "y":
            print("Exiting...")
            return
        shutil.rmtree(output_dir)
        print("Deleted existing output directory.")

    results_dir = Path(results_dir)
    with open(results_dir / "config.yml", 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    gens = config["evolution_params"]["n_generations"]

    pareto_df = pd.read_csv(results_dir / f"{gens}.csv")
    pareto_df = pareto_df[pareto_df["rank"] == 1]

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(results_dir / "config.yml", output_dir / "config.yml")
    shutil.copy(results_dir / f"{gens}.csv", output_dir / f"{gens}.csv")

    for cand_id in pareto_df["cand_id"]:
        gen_dir = output_dir / f"{cand_id.split('_')[0]}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(results_dir / f"{cand_id.split('_')[0]}" / f"{cand_id}", gen_dir / f"{cand_id}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory to copy from.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to copy to.")
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)
