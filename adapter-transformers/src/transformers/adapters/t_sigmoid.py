import torch
from torch import nn


class Tsigmoid(nn.Module):
    def __init__(self):
        super(Tsigmoid, self).__init__()

    def forward(self, x: torch.Tensor, w: float = 1):
        # set w device == x device
        # w = w.to(x.device)
        out = 1 - torch.log2(torch.tensor(w + 1)) / (1 + torch.exp(x))  # 0<w<1
        return out

    # now with cuda support

