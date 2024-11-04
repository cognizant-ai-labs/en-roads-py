"""
Utility functions for experimentation
"""
import json
from pathlib import Path

import torch

from evolution.candidate import Candidate


class Experimenter:
    """
    Helper functions to be used in experimentation.
    """
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        with open(results_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        self.context = config["context"]
        self.actions = config["actions"]

        self.model_params = config["model_params"]
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def get_candidate_actions(self,
                              candidate: Candidate,
                              torch_context: torch.Tensor,
                              context_vals: torch.Tensor) -> dict[str, float]:
        """
        Gets actions from a candidate given a context
        """
        [actions_dict] = candidate.prescribe(torch_context.to(self.device).unsqueeze(0))
        context_dict = dict(zip(self.context, context_vals.tolist()))
        actions_dict.update(context_dict)
        return actions_dict

    def get_candidate_from_id(self, cand_id: str) -> Candidate:
        """
        Loads a candidate from an id.
        """
        cand_path = self.results_dir / cand_id.split("_")[0] / f"{cand_id}.pt"
        return Candidate.from_seed(cand_path, self.model_params, self.actions)
