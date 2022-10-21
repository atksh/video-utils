import torch
from pytorch_lightning.callbacks.callback import Callback


class SetPrecisionCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.set_float32_matmul_precision("medium")
