import argparse
import json
from pathlib import Path
import shutil

from evolution.evolution import Evolution
from evolution.utils import modify_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    config = modify_config(config)
    print(config)

    if Path(config["save_path"]).exists():
        inp = input("Save path already exists, do you want to overwrite? (y/n):")
        if inp.lower() == "y":
            shutil.rmtree(config["save_path"])
        else:
            print("Exiting")
            exit()

    evolution = Evolution(config)
    evolution.neuroevolution()

if __name__ == "__main__":
    main()