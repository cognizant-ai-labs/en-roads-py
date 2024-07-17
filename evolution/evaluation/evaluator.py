"""
Evaluates candidates in order for them to be sorted.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evolution.candidate import Candidate
from evolution.evaluation.data import ContextDataset
from evolution.outcomes.outcome_manager import OutcomeManager
from run_enroads import run_enroads, compile_enroads

class Evaluator:
    """
    Evaluates candidates by generating the actions and running the enroads model on them.
    Generates and stores context data based on config using ContextDataset.
    """
    def __init__(self, temp_dir: str, context: list[str], actions: list[str], outcomes: dict[str, bool]):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.actions = actions
        self.outcomes = outcomes
        self.outcome_manager = OutcomeManager(outcomes)

        # Precise float is required to load the enroads inputs properly
        self.input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)

        self.context = context
        self.context_dataset = ContextDataset(context)
        self.context_dataloader = DataLoader(self.context_dataset, batch_size=3, shuffle=False)

        compile_enroads()

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

    def validate_outcomes(self, outcomes_df: pd.DataFrame):
        """
        Ensures our outcome columns don't have NaNs or infs
        """
        outcome_keys = [key for key in self.outcomes if key in outcomes_df.columns]
        subset = outcomes_df[outcome_keys]
        assert not subset.isna().any().any(), "Outcomes contain NaNs."
        assert not np.isinf(subset.to_numpy()).any(), "Outcomes contain infs."
        return True

    def evaluate_actions(self, actions_dict: dict[str, str]):
        """
        Evaluates actions a candidate produced.
        """
        self.construct_enroads_input(actions_dict)
        run_enroads(self.temp_dir / "enroads_output.txt", self.temp_dir / "enroads_input.txt")
        outcomes_df = pd.read_csv(self.temp_dir / "enroads_output.txt", sep="\t")
        self.validate_outcomes(outcomes_df)
        return outcomes_df

    def reconstruct_context_dicts(self, batch_context: torch.Tensor) -> list[dict[str, float]]:
        """
        Takes a torch tensor and zips it with the context labels to create a list of dicts.
        """
        context_dicts = []
        for row in batch_context:
            context_dict = dict(zip(self.context, row.tolist()))
            context_dicts.append(context_dict)
        return context_dicts

    def evaluate_candidate(self, candidate: Candidate):
        """
        Evaluates a single candidate by running all the context through it and receiving all the batches of actions.
        Then evaluates all the actions and returns the average outcome.
        """
        outcomes_dfs = []
        cand_results = []
        # Iterate over batches of contexts
        for batch_tensor, batch_context in self.context_dataloader:
            context_dicts = self.reconstruct_context_dicts(batch_context)
            actions_dicts = candidate.prescribe(batch_tensor.to("mps"))
            for actions_dict, context_dict in zip(actions_dicts, context_dicts):
                # Add context to actions so we can pass it into the model
                actions_dict.update(context_dict)
                outcomes_df = self.evaluate_actions(actions_dict)
                outcomes_dfs.append(outcomes_df)
                cand_results.append(self.outcome_manager.process_outcomes(outcomes_df))

        candidate.metrics = {key: np.mean([result[key] for result in cand_results]) for key in cand_results[0]}
        return outcomes_dfs

    def evaluate_candidates(self, candidates: list[Candidate]):
        """
        Evaluates all candidates. Doesn't unnecessarily evaluate candidates that have already been evaluated.
        """
        for candidate in tqdm(candidates, leave=False):
            if len(candidate.metrics) == 0:
                self.evaluate_candidate(candidate)
