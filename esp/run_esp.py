import argparse
from pathlib import Path
import shutil

from presp.evolution import Evolution
from presp.prescriptor import NNPrescriptorFactory
import yaml

from esp.evaluator import EnROADSEvaluator
from esp.prescriptor import EnROADSPrescriptor
from esp.seeds.seeding import create_seeds

def main():
    """
    Main logic for running neuroevolution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+", required=True)
    args = parser.parse_args()
    for config_path in args.config:
        print(f"Running evolution with config: {config_path}")
        with open(Path(config_path), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        evolution_params = config["evolution_params"]
        seed_params = config["seed_params"]
        prescriptor_params = config["prescriptor_params"]
        eval_params = config["eval_params"]

        # Check if the path exists
        save_path = Path(evolution_params["save_path"])
        if save_path.exists():
            inp = input(f"Save path {save_path} already exists. Do you want to overwrite? [Y|n].")
            if inp.lower() != "y":
                print("Exiting.")
                break
            shutil.rmtree(save_path)
        # Save config to save path
        save_path.mkdir(parents=True, exist_ok=False)
        with open(save_path / "config.yml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

        # Construct prescriptor and evaluator objects
        prescriptor_factory = NNPrescriptorFactory(EnROADSPrescriptor, **prescriptor_params)
        evaluator = EnROADSEvaluator(**eval_params)

        # Seeding
        seed_dir = config["evolution_params"]["seed_dir"]
        if seed_dir is not None:
            seed_dir = Path(seed_dir)
            if not seed_dir.exists():
                seed_dir.mkdir(parents=True)
                create_seeds(seed_dir,
                             prescriptor_params["model_params"],
                             evaluator.dataset,
                             eval_params["actions"],
                             seed_params["seed_urls"],
                             seed_params["epochs"])

        # Run evolution
        evolution = Evolution(prescriptor_factory=prescriptor_factory,
                                evaluator=evaluator,
                                **evolution_params)
        evolution.run_evolution()


if __name__ == "__main__":
    main()