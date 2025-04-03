"""
Universal way to gain access to the input specs for the enroads model.
Also stores the bad switch name because it has the same on and off value, messing up some tests.
"""
import json

import pandas as pd


BAD_SWITCH = "_qualifying_path_renewables"


def load_input_specs() -> pd.DataFrame:
    """
    Loads the input specs for the En-ROADS model. These specify default values, precision, and other information.
    We read in the source code file and extract the input specs from the JavaScript code.
    Strangely, this is actually faster than saving inputSpecs as a json file and reading it in with Pandas.
    TODO: Un hard-code the En-ROADS SDK version.
    """
    source_code = ""
    with open("enroadspy/en-roads-sdk-v24.6.0-beta1/packages/en-roads-core/dist/index.js", "r", encoding="utf-8") as f:
        source_code = f.read()

    # Locate var inputSpecs in the source code
    input_specs_start = source_code.find("var inputSpecs = [")
    input_specs_start = source_code.find("[", input_specs_start)

    # Find the end of inputSpecs by the next semicolon
    input_specs_end = source_code.find(";", input_specs_start)

    json_str = source_code[input_specs_start:input_specs_end]
    input_specs_json = json.loads(json_str)
    input_specs = pd.DataFrame(input_specs_json)

    # Convert column type for proper comparison later
    input_specs["id"] = input_specs["id"].astype(int)

    return input_specs
