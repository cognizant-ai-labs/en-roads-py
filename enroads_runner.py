"""
File containing functions used to compile and run en-roads with variable input arguments.
"""
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

class EnroadsRunner():
    """
    Class that handles the running of the En-ROADS simulator.
    """
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)

        self.compile_enroads()

    def compile_enroads(self):
        """
        Compiles the en-roads model.
        Make sure you extracted the zip file in the current directory.
        """
        subprocess.run(["make"], cwd="en-roads-sdk-v24.6.0-beta1/c", check=True)

    def format_string_input(self, value, decimal):
        """
        Formats a value to a string with the correct number of decimals.
        """
        return f"{value:.{decimal}f}"

    # pylint: disable=no-member
    def construct_enroads_input(self, inputs: dict[str, float]):
        """
        Constructs input file according to enroads.
        We want the index of the input and the value separated by a colon. Then separate those by spaces.
        TODO: This is pretty inefficient at the moment.
        """
        input_specs = self.input_specs.copy()
        input_specs["index"] = range(len(input_specs))

        # For switches we set the decimal to 0.
        # For steps of >= 1 we set the decimal to 0 as they should already be rounded integers.
        input_specs["step"] = input_specs["step"].fillna(1)
        # We do np.ceil to round up because in the case of 0.05 we want 2 decimals not 1.
        # We also know the default values will be in correct steps which means we don't have to worry about
        # truncating them to the nearest step.
        input_specs["decimal"] = np.ceil(-1 * np.log10(input_specs["step"])).astype(int)
        input_specs.loc[input_specs["decimal"] <= 0, "decimal"] = 0

        # Get all the values from the dict and replace NaNs with default values
        value = input_specs["varId"].map(inputs)
        value.fillna(input_specs["defaultValue"], inplace=True)
        input_specs["value"] = value

        # Format the values to strings with the correct number of decimals
        input_specs["value_str"] = input_specs.apply(lambda row: self.format_string_input(row["value"], row["decimal"]), axis=1)
        # input_specs["value_str"] = input_specs["value"].astype(str)
        
        # Format string for En-ROADS input
        input_specs["input_col"] = input_specs["index"].astype(str) + ":" + input_specs["value_str"]
        input_str = " ".join(input_specs["input_col"])
        with open(self.temp_dir / "enroads_input.txt", "w", encoding="utf-8") as f:
            f.write(input_str)

        return input_str
    # pylint: enable=no-member

    def run_enroads(self, output_path, input_path=None):
        """
        Simple function to run the enroads simulator.
        Possible variable values are stored in inputSpecs.jsonl.
        From the documentation:
            The input string is a space-delimited list of index-value pairs, where a colon separates the 
            index number from the value number with no spaces. Index numbers are zero-based.
        NOTE: The indices are the line numbers in inputSpecs.jsonl starting from 0, NOT the id column.
        """
        with open(output_path, "w", encoding="utf-8") as out_file:
            command = ["./en-roads-sdk-v24.6.0-beta1/c/enroads"]
            if input_path:
                command.append(input_path)
            # TODO: Gacky thing we do here where if a sim run fails we manually try again.
            code = subprocess.run(command, stdout=out_file, check=False)
            if code.returncode != 0:
                print("Failed")
                code = subprocess.run(command, stdout=out_file, check=True)

    def evaluate_actions(self, actions_dict: dict[str, str]):
        """
        Evaluates actions a candidate produced.
        TODO: There is a potential race condition where enroads hasn't written the full output file yet. We just
        handle this by sleeping for a second, then reading again. If the file still doesn't exist we throw an error.
        """
        if len(actions_dict) > 0:
            self.construct_enroads_input(actions_dict)
            self.run_enroads(self.temp_dir / "enroads_output.txt", self.temp_dir / "enroads_input.txt")
        else:
            self.run_enroads(self.temp_dir / "enroads_output.txt")
        try:
            outcomes_df = pd.read_csv(self.temp_dir / "enroads_output.txt", sep="\t")
        except pd.errors.EmptyDataError:
            print("Unable to read output file. Trying again.")
            time.sleep(1)
            outcomes_df = pd.read_csv(self.temp_dir / "enroads_output.txt", sep="\t")
        return outcomes_df
