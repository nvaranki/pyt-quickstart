
from torch import nn, Tensor
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    r"""Definition of the math model"""

    grdB = Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, ])
    grdW = Tensor([
        [[[+0.5, -0.5],
          [ 0.0,  0.0]]],
        [[[ 0.0,  0.0],
          [+0.5, -0.5]]],
        [[[+0.5,  0.0],
          [-0.5,  0.0]]],
        [[[ 0.0, +0.5],
          [ 0.0, -0.5]]],
        [[[+0.5,  0.0],
          [ 0.0, -0.5]]],
        [[[ 0.0, +0.5],
          [-0.5,  0.0]]],
        [[[0.25, 0.25],
          [0.25, 0.25]]],
    ])

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7*27*27, 2512),
            # Pow(2), #  4m: nn.Mish() Accuracy: 75.0%, Avg loss: 0.769570
            nn.Tanh(),
            # -Linear, -Tanh as 4m # Accuracy: 73.5%, Avg loss: 1.341088
            nn.Linear(2512, 10),
            nn.Mish() # 4m: Accuracy: 75.1%, Avg loss: 0.767739;
            # nn.Tanh(), # Accuracy: 70.9%, Avg loss: 1.445483
            # nn.Sigmoid() # Accuracy: 69.3%, Avg loss: 1.979252
        )

    def forward(self, x):
        x = F.conv2d(x, self.grdW, self.grdB, stride=1, padding=0)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def to(self, *args, **kwargs):
        super(NeuralNetwork, self).to(*args, **kwargs)
        self.grdW = self.grdW.to(*args, **kwargs)
        self.grdB = self.grdB.to(*args, **kwargs)
        return self
