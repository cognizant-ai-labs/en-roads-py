"""
Utility functions for experimentation
"""
from pathlib import Path

from presp.prescriptor import NNPrescriptorFactory
import torch
import yaml

from evolution.candidate import EnROADSPrescriptor


class Experimenter:
    """
    Helper functions to be used in experimentation.
    """
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        with open(results_dir / "config.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.context = config["context"]
        self.device = config["device"]

        self.prescriptor_factory = NNPrescriptorFactory(EnROADSPrescriptor,
                                                        model_params=config["model_params"],
                                                        device=config["device"],
                                                        actions=config["actions"])

    def get_candidate_actions(self,
                              candidate: EnROADSPrescriptor,
                              torch_context: torch.Tensor,
                              context_vals: torch.Tensor) -> dict[str, float]:
        """
        Gets actions from a candidate given a context
        """
        [actions_dict] = candidate.forward(torch_context.to(self.device).unsqueeze(0))
        context_dict = dict(zip(self.context, context_vals.tolist()))
        actions_dict.update(context_dict)
        return actions_dict

    def get_candidate_from_id(self, cand_id: str) -> EnROADSPrescriptor:
        """
        Loads a candidate from an id.
        """
        cand_path = self.results_dir / cand_id.split("_")[0] / f"{cand_id}"
        return self.prescriptor_factory.load(cand_path)
