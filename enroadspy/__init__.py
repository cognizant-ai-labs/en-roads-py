"""
Universal way to gain access to the input specs for the enroads model.
Also stores the bad switch name because it has the same on and off value, messing up some tests.
"""
import json

import pandas as pd


BAD_SWITCH = "_qualifying_path_renewables"
SIMPLE_ACTIONS = [
    '_source_subsidy_delivered_coal_tce',
    '_source_subsidy_delivered_oil_boe',
    '_source_subsidy_delivered_gas_mcf',
    '_source_subsidy_renewables_kwh',
    '_source_subsidy_delivered_bio_boe',
    '_source_subsidy_nuclear_kwh',
    '_carbon_tax_initial_target',
    '_annual_improvement_to_energy_efficiency_of_new_capital_stationary',
    '_annual_improvement_to_energy_efficiency_of_new_capital_transport',
    '_electric_carrier_subsidy_with_required_comp_assets',
    '_electric_carrier_subsidy_stationary',
    '_target_change_in_other_ghgs_for_ag',
    '_land_cdr_percent_of_reference',
    '_target_change_other_ghgs_leakage_and_waste',
    '_deforestation_slider_setting',
    '_tech_cdr_percent_of_reference'
]


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


def name_to_id(name: str, input_specs: pd.DataFrame) -> str:
    """
    Converts the En-ROADS nice variable name to its unique ID (also a string).
    """
    return input_specs.loc[input_specs["varName"] == name, "varId"].values[0]


def id_to_name(var_id: str, input_specs: pd.DataFrame) -> str:
    """
    Converts the En-ROADS unique ID to its nice variable name.
    """
    return input_specs.loc[input_specs["varId"] == var_id, "varName"].values[0]
