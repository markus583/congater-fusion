import torch
from torch import nn


class Tsigmoid(nn.Module):
    def __init__(self):
        super(Tsigmoid, self).__init__()

    def forward(self, x: torch.Tensor, w: float = 1):
        out = 1 - torch.log2(torch.tensor(w + 1)) / (1 + torch.exp(x))  # 0<w<1
        return out

class ReverseTsigmoid(nn.Module):
    # https://www.wolframalpha.com/input?i=plot+-1+%2B+log2%28w+%2B+1%29+%2F+%281+%2B+exp%28-x%29%29+%2B+1from+-10+to+10
    def __init__(self):
        super(Tsigmoid, self).__init__()

    def forward(self, x: torch.Tensor, w: float = 1):
        out = torch.log2(torch.tensor(w + 1)) / (1 + torch.exp(-x)) # 0<w<1
        return out

