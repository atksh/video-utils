import torch
from torch.utils.checkpoint import checkpoint


def ckpt_forward(func):
    def wrapper(*args, **kwargs):
        all_inputs = args + tuple(kwargs.values())
        if any(torch.is_tensor(x) and x.requires_grad for x in all_inputs):
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def ckpt_seq_forward(func):
    def wrapper(*args, **kwargs):
        all_inputs = args + tuple(kwargs.values())
        batch_size = None
        for a in all_inputs:
            if torch.is_tensor(a):
                batch_size = a.shape[0]
                break
        if batch_size is not None and any(
            torch.is_tensor(x) and x.requires_grad for x in all_inputs
        ):
            new_args = []
            new_kwargs = []
            for i in range(batch_size):
                new_args.append([a[[i]] if torch.is_tensor(a) else a for a in args])
                new_kwargs.append(
                    {k: v[[i]] if torch.is_tensor(v) else v for k, v in kwargs.items()}
                )
            out = []
            for a, k in zip(new_args, new_kwargs):
                out.append(checkpoint(func, *a, **k))
            return torch.cat(out, dim=0)
        else:
            return func(*args, **kwargs)

    return wrapper
