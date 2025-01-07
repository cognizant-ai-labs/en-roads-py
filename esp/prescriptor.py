from presp.prescriptor import NNPrescriptor
import torch


class EnROADSPrescriptor(NNPrescriptor):
    """
    A simple NNPrescriptor that sigmoids the outputs of the neural network.
    """
    def __init__(self, model_params: dict, device: str):
        super().__init__(model_params, device)
        self.model = torch.nn.Sequential(self.model, torch.nn.Sigmoid())

