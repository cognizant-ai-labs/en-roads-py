import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from evolution.candidate import Candidate
from run_enroads import run_enroads, compile_enroads

class Evaluator:
    """
    Evaluates candidates by generating the actions and running the enroads model on them.
    """
    def __init__(self, temp_dir: str, context, actions, outcomes: dict[str, bool]):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.actions = actions
        self.outcomes = outcomes

        self.input_df = pd.read_json("inputSpecs.jsonl", lines=True)
        self.input_specs = self.input_df.copy() # For validation purposes
        self.input_df["index"] = range(len(self.input_df))
        self.input_df["value"] = self.input_df["defaultValue"]

        input_context = self.input_df[self.input_df["varId"].isin(context)]
        assert len(input_context) == len(context), f"Context is not the correct length. Expected {len(context)}, got {len(input_context)}."

        # self.torch_context = torch.tensor(input_context["value"].values, device="mps", dtype=torch.float32)
        tensor_context = self.generate_context(context, 1)
        dataset = TensorDataset(tensor_context)
        self.torch_context = DataLoader(dataset, batch_size=1, shuffle=False)

        compile_enroads()

    def generate_context(self, context: list[str], n: int):
        """
        Generates a uniform distribution of the data.
        TODO: Assumes that there are no switches in the context
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True)
        context_specs = input_specs[input_specs["varId"].isin(context)]
        context_specs.loc[:,'varId'] = pd.Categorical(context_specs['varId'], categories=context, ordered=True)
        # Sort the DataFrame by the 'id' column
        context_specs = context_specs.sort_values('varId')

        rng = np.random.default_rng(42)
        data = rng.uniform(context_specs["minValue"], context_specs["maxValue"], (n, len(context)))
        return torch.tensor(data, device="mps", dtype=torch.float32)

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
        input_specs["decimal"] = (-1 * np.log10(input_specs["step"])).astype(int)
        input_specs.loc[input_specs["decimal"] < 0, "decimal"] = 0

        value = self.input_df["varId"].map(inputs)
        value.fillna(self.input_df["defaultValue"], inplace=True)
        input_specs["value"] = value
        input_specs["value_str"] = input_specs.apply(lambda row: self.format_string_input(row["value"], row["decimal"]), axis=1)

        # self.input_df["value"] = self.input_df["defaultValue"]

        # # Update values in inputs
        # val_col = self.input_df['varId'].map(inputs)
        # self.input_df["value"] = val_col.fillna(self.input_df["value"])
        
        input_specs["input_col"] = input_specs["index"].astype(str) + ":" + input_specs["value_str"]
        input_str = " ".join(input_specs["input_col"])
        
        with open(self.temp_dir / "enroads_input.txt", "w", encoding="utf-8") as f:
            f.write(input_str)

        return input_str
    # pylint: enable=no-member

    def evaluate_actions(self, actions_dict: dict[str, str]):
        """
        Evaluates actions a candidate produced.
        """
        self.construct_enroads_input(actions_dict)
        run_enroads(self.temp_dir / "enroads_output.txt", self.temp_dir / "enroads_input.txt")
        outcomes_df = pd.read_csv(self.temp_dir / "enroads_output.txt", sep="\t")
        return outcomes_df

    def evaluate_candidate(self, candidate: Candidate):
        """
        Evaluates a single candidate by running all the context through it and receiving all the batches of actions.
        Then evaluates all the actions and returns the average outcome.
        This function also handles the custom outcome "Cost of energy next 10 years".
        """
        outcomes_dfs = []
        results_dict = {outcome: 0 for outcome in self.outcomes}
        for [batch] in self.torch_context:
            batch_action_dicts = candidate.prescribe(batch)
            for actions_dict in batch_action_dicts:
                outcomes_df = self.evaluate_actions(actions_dict)
                outcomes_dfs.append(outcomes_df)
                for outcome in self.outcomes:
                    # TODO: Technically this breaks if we iterate over this key first
                    if outcome == "Cost of energy next 10 years":
                        cost_col = outcomes_df["Total cost of energy"]
                        cost = cost_col.iloc[2025-1990:2035-1990].mean()
                        results_dict[outcome] += cost
                    else:
                        # We have to replace nans with either -inf if we're maximizing or inf if we're minimizing
                        measured_outcome = outcomes_df[outcome]
                        ascending = self.outcomes[outcome]
                        if ascending:
                            measured_outcome.fillna(float("inf"), inplace=True)
                        else:
                            measured_outcome.fillna(float("-inf"), inplace=True)
                        results_dict[outcome] += measured_outcome.iloc[-1]

        candidate.metrics = {k: v / len(outcomes_dfs) for k, v in results_dict.items()}
        
        # Return this for testing purposes
        for i in range(1, len(outcomes_dfs)):
            assert outcomes_dfs[i].iloc[:2024-1990].equals(outcomes_dfs[i-1].iloc[:2024-1990])
        concatenated = pd.concat(outcomes_dfs, axis=0)
        average_outcomes_df = concatenated.groupby(concatenated.index).mean()

        return average_outcomes_df

    def evaluate_candidates(self, candidates: list[Candidate]):
        """
        Evaluates all candidates. Doesn't unnecessarily evaluate candidates that have already been evaluated.
        """
        for candidate in tqdm(candidates, leave=False):
            if len(candidate.metrics) == 0:
                self.evaluate_candidate(candidate)
