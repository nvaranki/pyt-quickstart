import torch
from torch import Tensor
from torch.nn.modules import Module


class Pow(Module):

    def __init__(self, t: float,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Pow, self).__init__()
        self.t = t

    def forward(self, input: Tensor) -> Tensor:
        return torch.pow(input, self.t)
