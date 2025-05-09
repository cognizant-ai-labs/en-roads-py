"""
Universal way to gain access to the input specs for the enroads model.
Also stores the bad switch name because it has the same on and off value, messing up some tests.
"""
import json

import pandas as pd


BAD_SWITCH = 263

SDK_VERSION = "v25.4.0-beta1"


def load_input_specs() -> pd.DataFrame:
    """
    Loads the input specs for the En-ROADS model. These specify default values, precision, and other information.
    We read in the source code file and extract the input specs from the JavaScript code.
    Strangely, this is actually faster than saving inputSpecs as a json file and reading it in with Pandas.
    TODO: Un hard-code the En-ROADS SDK version.
    """
    source_code = ""
    with open(f"enroadspy/en-roads-sdk-{SDK_VERSION}/packages/en-roads-core/dist/index.js", "r", encoding="utf-8") as f:
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


def name_to_id(name: str, input_specs: pd.DataFrame) -> int:
    """
    Converts the En-ROADS pretty variable name to its unique integer ID.
    Returns -1 if the name is not found.
    """
    filtered = input_specs[input_specs["varName"] == name]
    if filtered.empty:
        return -1
    return filtered["id"].values[0]


def id_to_name(input_id: int, input_specs: pd.DataFrame) -> str:
    """
    Converts the En-ROADS unique integer ID to its nice variable name.
    Returns None if not found.
    """
    filtered = input_specs[input_specs["id"] == input_id]
    if filtered.empty:
        return None
    return filtered["varName"].values[0]


def varid_to_id(varid: str, input_specs: pd.DataFrame) -> int:
    """
    Converts the En-ROADS varId to its unique integer ID.
    Returns -1 if not found.
    """
    filtered = input_specs[input_specs["varId"] == varid]
    if filtered.empty:
        return -1
    return filtered["id"].values[0]


def id_to_varid(input_id: int, input_specs: pd.DataFrame) -> str:
    """
    Converts the En-ROADS unique integer ID to its varId from the inputSpecs.
    Returns None if not found.
    """
    filtered = input_specs[input_specs["id"] == input_id]
    if filtered.empty:
        return None
    return filtered["varId"].values[0]
