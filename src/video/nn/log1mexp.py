import math
import torch

LOG2 = math.log(2.0)


class Log1mexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = torch.where(
            LOG2 <= x, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x))
        )

        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / torch.expm1(x)


exp1mexp = Log1mexp.apply
