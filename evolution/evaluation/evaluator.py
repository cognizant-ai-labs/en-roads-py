from pathlib import Path

import pandas as pd
import torch

from evolution.candidate import Candidate
from run_enroads import run_enroads, compile_enroads

class Evaluator:
    def __init__(self, temp_dir: str, context, actions, outcomes: list[str]):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.actions = actions
        self.outcomes = outcomes

        self.input_df = pd.read_json("inputSpecs.jsonl", lines=True)
        self.input_df["index"] = range(len(self.input_df))
        self.input_df["value"] = self.input_df["defaultValue"]

        input_context = self.input_df[self.input_df["varId"].isin(context)]
        assert len(input_context) == len(context), f"Context is not the correct length. Expected {len(context)}, got {len(input_context)}."

        self.torch_context = torch.tensor(input_context["value"].values, device="mps", dtype=torch.float32)

        compile_enroads()

    # pylint: disable=no-member
    def construct_enroads_input(self, inputs: dict[str, str]):
        self.input_df["value"] = self.input_df["defaultValue"]

        # Update values in inputs
        val_col = self.input_df['varId'].map(inputs)
        self.input_df["value"] = val_col.fillna(self.input_df["value"])
        
        self.input_df["input_col"] = self.input_df["index"].astype(str) + ":" + self.input_df["value"].astype(str)
        input_str = " ".join(self.input_df["input_col"])
        
        with open(self.temp_dir / "enroads_input.txt", "w", encoding="utf-8") as f:
            f.write(input_str)

        return input_str
    # pylint: enable=no-member

    def evaluate_actions(self, actions_dict: dict[str, str]):
        self.construct_enroads_input(actions_dict)
        run_enroads(self.temp_dir / "enroads_output.txt", self.temp_dir / "enroads_input.txt")
        outcomes_df = pd.read_csv(self.temp_dir / "enroads_output.txt", sep="\t")
        return outcomes_df

    def evaluate_candidate(self, candidate: Candidate):
        actions_arr = candidate.prescribe(self.torch_context)
        actions_dict = dict(zip(self.actions, actions_arr))
        outcomes_df = self.evaluate_actions(actions_dict)
        outcomes_df.fillna(99, inplace=True)
        
        for outcome in self.outcomes:
            candidate.metrics[outcome] = -1 * outcomes_df[outcome].iloc[-1]

    def evaluate_candidates(self, candidates: list[Candidate]):
        for candidate in candidates:
            if len(candidate.metrics) == 0:
                self.evaluate_candidate(candidate)
