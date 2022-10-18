import opt_einsum as oe
import torch
import torch.nn as nn


class OptEinsum(nn.Module):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
        self.exprs = {}

    def get_expr(self, *args):
        shapes = tuple([arg.shape for arg in args])
        if shapes not in self.exprs:
            self.exprs[shapes] = oe.contract_expression(
                self.equation, *shapes, optimize="optimal"
            )
        return self.exprs[shapes]

    def forward(self, *args):
        expr = self.get_expr(*args)
        return expr(*args)


class Einsum(nn.Module):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *args):
        return torch.einsum(self.equation, *args)
