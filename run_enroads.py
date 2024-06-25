"""
File containing functions used to compile and run en-roads with variable input arguments.
"""
import subprocess
import time

import pandas as pd

def compile_enroads():
    """
    Compiles the en-roads model.
    Make sure you extracted the zip file in the current directory.
    """
    subprocess.run(["make"], cwd="en-roads-sdk-v24.6.0-beta1/c", check=True)

def run_enroads(output_path, input_str=None):
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
        if input_str:
            command.append(input_str)
        subprocess.run(command, stdout=out_file, check=True)

def main():
    """
    Dummy demo function to show how to use the functions.
    We read in input_specs and do some modification to create our input string, then run the model.
    This writes out to file which we can load as a dataframe.
    In the future this can be run directly through a buffer which eliminates the read/write before each run.
    """
    compile_enroads()
    input_specs = pd.read_json("inputSpecs.jsonl", lines=True)
    input_specs["index"] = range(len(input_specs))
    avg_vals = (input_specs["minValue"] + input_specs["maxValue"]) / 2
    input_col = input_specs["index"].astype(str) + ":" + avg_vals.astype(str)
    s = time.time()
    with open("test_input.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(input_col))

    run_enroads("output.txt", "test_input.txt")

    results_df = pd.read_csv("output.txt", sep="\t")
    e = time.time()
    print(results_df.head())

    print(f"Simulator took:{e-s:.4f} seconds")

if __name__ == "__main__":
    main()