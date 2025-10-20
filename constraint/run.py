from pathlib import Path
import shutil

from presp.evolution import Evolution
from presp.prescriptor import NNPrescriptorFactory
import yaml

from constraint.evaluator import ConstraintEvaluator
from constraint.prescriptor import ConstraintPrescriptor
from evolution.utils import process_config


def main():
    with open("constraint/config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    save_path = Path(config["evolution_params"]["save_path"])
    if save_path.exists():
        inp = input(f"Directory {save_path} already exists. Do you want to overwrite it? (y/n): ")
        if inp.lower() == "y":
            shutil.rmtree(save_path)
        else:
            print("Exiting...")
            return
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    config = process_config(config)

    factory = NNPrescriptorFactory(ConstraintPrescriptor,
                                   model_params=config["model_params"],
                                   context=config["context"],
                                   actions=config["actions"],
                                   device=config["device"])

    evaluator = ConstraintEvaluator(config["context"],
                                    config["actions"],
                                    config["outcomes"],
                                    config["n_jobs"],
                                    config["batch_size"],
                                    config["device"])

    evolution = Evolution(**config["evolution_params"],
                          prescriptor_factory=factory,
                          evaluator=evaluator)

    evolution.run_evolution()


if __name__ == "__main__":
    main()