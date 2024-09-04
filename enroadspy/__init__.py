"""
Universal way to gain access to the input specs for the enroads model.
"""
from pathlib import Path

import pandas as pd


def load_input_specs() -> pd.DataFrame:
    """
    Loads the input specs for the En-ROADS model from the inputSpecs.jsonl file.
    We make sure precise_float=True so that we get exact floats like 15 instead of 15.00001.
    """
    return pd.read_json(Path("enroadspy/inputSpecs.jsonl"), lines=True, precise_float=True)
