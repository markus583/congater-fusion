import torch
from torch import nn


class Tsigmoid(nn.Module):
    def __init__(self, variable_omega: bool = False):
        super(Tsigmoid, self).__init__()
        self.variable_omega = variable_omega

    def forward(self, x: torch.Tensor, w: torch.tensor = 1):
        out = 1 - torch.log2(w + 1) / (1 + torch.exp(x))  # 0<w<1
        return out


class TTsigmoid(nn.Module):
    # https://www.wolframalpha.com/input?i=plot+-1+%2B+log2%28w+%2B+1%29+%2F+%281+%2B+exp%28-x%29%29+%2B+1from+-10+to+10
    def __init__(self, variable_omega: bool = False, omega_offset: float = 1.0):
        super(TTsigmoid, self).__init__()
        self.variable_omega = variable_omega
        self.omega_offset = omega_offset

    def forward(self, x: torch.Tensor, w: torch.tensor = 1, clamp_omega: bool = False):
        if clamp_omega:
            w = torch.clamp(w, 0, 1)
        out = torch.log2(w + self.omega_offset) / (1 + torch.exp(-x))  # 0<w<1
        return out


class TTTanh(nn.Module):
    def __init__(self, variable_omega: bool = False, omega_offset: float = 1.0):
        super(TTTanh, self).__init__()
        self.variable_omega = variable_omega

    def forward(self, x: torch.Tensor, w: torch.tensor = 1):
        out = w * torch.tanh(x)  # 0<w<1
        return out


class TTTanhV2(nn.Module):
    def __init__(self, variable_omega: bool = False, omega_offset: float = 1.0):
        super(TTTanhV2, self).__init__()
        self.variable_omega = variable_omega
        # self.omega = nn.Parameter(
        #     torch.ones(12, 768, requires_grad=True), requires_grad=True
        # )
        # replace with tensor
        # make trainable

    def forward(self, x, clamp_omega: bool = False):
        # inp = (w + 1) * x
        out = torch.tanh(x)  # 0<w<1
        return out
    


class TTSelu(nn.Module):
    def __init__(self, variable_omega: bool = False):
        super(TTSelu, self).__init__()
        self.variable_omega = variable_omega
        self.alpha = 1.6732632423543772848170429916717
        self.lambd = 1.0507009873554804934193349852946

    def forward(self, x: torch.Tensor, w: torch.tensor = 1):
        # max(0, -exp(-x) + 1) + min(exp(x)-1, 0)
        out = w * (
            torch.max(torch.tensor(0), -torch.exp(-x) + 1)
            + torch.min(torch.exp(x) - 1, torch.tensor(0))
        )  # 0<w<1
        return out


class TTSeluV2(nn.Module):
    def __init__(
        self,
        variable_omega: bool = False,
        alpha: float = 1.6732632423543772848170429916717,
        lambd: float = 1.0507009873554804934193349852946,
    ):
        super(TTSeluV2, self).__init__()
        self.variable_omega = variable_omega
        self.alpha = alpha
        self.lambd = lambd

    def forward(self, x: torch.Tensor, w: torch.tensor = 1):
        # max(0, -exp(-x) + 1) + min(exp(x)-1, 0)
        out = (
            self.alpha
            * self.lambd
            * w
            * (
                torch.max(torch.tensor(0), -torch.exp(-x) + 1)
                + torch.min(torch.exp(x) - 1, torch.tensor(0))
            )
        )  # 0<w<1
        return out
