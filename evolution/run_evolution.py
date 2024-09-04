"""
Script used to run the evolution process.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from evolution.evolution import Evolution
from evolution.utils import modify_config


def main():
    """
    Parses arguments, modifies config to reduce the amount of manual text added to it, then runs the evolution process.
    Prompts the user to overwrite the save path if it already exists.
    """
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
            sys.exit()

    evolution = Evolution(config)
    evolution.neuroevolution()


if __name__ == "__main__":
    main()
