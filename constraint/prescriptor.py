from presp.prescriptor import NNPrescriptor
import torch

from evolution.candidates.output_parser import OutputParser


class ConstraintPrescriptor(NNPrescriptor):
    """
    NNPrescriptor wrapper that scales the output to the context.
    context is a subset of actions.
    """
    def __init__(self, model_params: dict, context: list[int], actions: list[int], device: str = "cpu"):
        super().__init__(model_params, device)
        self.actions = list(actions)
        # Get which indices of the actions are in the context
        self.context_idxs = [self.actions.index(c) for c in context]
        self.output_parser = OutputParser(self.actions, device=device)

    def forward(self, context: torch.Tensor) -> list[dict]:
        """
        Updated forward method that scales the output based on the context.
        We scale the actions that are in the context to be between the mins and maxes passed in using the torch context.
        """
        with torch.no_grad():
            nn_outputs = super().forward(context)

            # Scale the output to between the context
            mins = context[:, ::2]
            maxes = context[:, 1::2]
            nn_outputs = nn_outputs[:, self.context_idxs] * (maxes - mins) + mins

            outputs = self.output_parser.parse_output(nn_outputs).cpu().numpy()
        actions_dicts = [dict(zip(self.actions, output.tolist())) for output in outputs]
        return actions_dicts
