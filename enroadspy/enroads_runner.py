"""
File containing functions used to compile and run en-roads with variable input arguments.
"""
import io
import subprocess
import tempfile

import numpy as np
import pandas as pd

from enroadspy import load_input_specs


class EnroadsRunner():
    """
    Class that handles the running of the En-ROADS simulator.
    """
    def __init__(self):
        self.input_specs = load_input_specs()
        self.compile_enroads()

    def compile_enroads(self):
        """
        Compiles the en-roads model.
        Make sure you extracted the zip file in the current directory.
        """
        subprocess.run(["make"], cwd="./enroadspy/en-roads-sdk-v24.6.0-beta1/c", check=True)

    def format_string_input(self, value, decimal):
        """
        Formats a value to a string with the correct number of decimals.
        """
        return f"{value:.{decimal}f}"

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

        # # Format the values to strings with the correct number of decimals
        input_specs["value_str"] = input_specs.apply(lambda row: self.format_string_input(row["value"],
                                                                                          row["decimal"]), axis=1)

        # Format string for En-ROADS input
        input_specs["input_col"] = input_specs["index"].astype(str) + ":" + input_specs["value_str"]
        input_str = " ".join(input_specs["input_col"])

        return input_str

    def check_input_string(self, input_str: str) -> bool:
        """
        Checks if the input string is valid for security purposes.
        1. Makes sure the input string is below a certain size in bytes (10,000).
        2. Makes sure the input string's values are numeric.
        """
        if len(input_str.encode('utf-8')) > 10000:
            return False
        for pair in input_str.split(" "):
            try:
                idx, val = pair.split(":")
                int(idx)
                float(val)
            except ValueError:
                return False
        return True

    def run_enroads(self, input_str=None):
        """
        Simple function to run the enroads simulator. A temporary file is created storing our input string as the
        En-ROADS main function requires a file as input.
        Possible variable values are stored in inputSpecs.jsonl.
        From the documentation:
            The input string is a space-delimited list of index-value pairs, where a colon separates the
            index number from the value number with no spaces. Index numbers are zero-based.
        NOTE: The indices are the order the variables appear in input_specs starting from 0, NOT the id column.
        """
        if input_str and not self.check_input_string(input_str):
            raise ValueError("Invalid input string")

        command = ["./enroadspy/en-roads-sdk-v24.6.0-beta1/c/enroads"]
        if input_str:
            with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
                temp_file.write(input_str)
                temp_file.flush()
                command.append(temp_file.name)
                result = subprocess.run(command, capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(command, capture_output=True, text=True, check=True)

        if result.returncode == 0:
            return result.stdout
        raise ValueError(f"Enroads failed with error code {result.returncode} and message {result.stderr}")

    def evaluate_actions(self, actions_dict: dict[str, str]):
        """
        Evaluates actions a candidate produced.
        Any actions not provided are replaced with the default value.
        Returns a DataFrame in the same format as the En-ROADS output with a column for each outcome and a row for each
        year.
        """
        if len(actions_dict) > 0:
            input_str = self.construct_enroads_input(actions_dict)
            raw_output = self.run_enroads(input_str)
        else:
            raw_output = self.run_enroads()
        outcomes_df = pd.read_table(io.StringIO(raw_output), sep="\t")
        return outcomes_df
