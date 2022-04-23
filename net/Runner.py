from net.NeuralNetwork import NeuralNetwork

import torch


class Runner:
    r"""Performer of trained model."""

    def __init__(self, device, fname):
        self.device = device
        # Create the model
        self.model = NeuralNetwork().eval()
        self.model.load_state_dict(torch.load(fname))
        self.model.to(device)
        print(self.model)

    def infer(self, x):
        return self.model(x.to(self.device))