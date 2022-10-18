import sys

import pytorch_lightning as pl

sys.path.append("src")

from config import *
from video.net import DataModule, Model

if __name__ == "__main__":
    cache_dir = "cache"
    path = ["video.mp4"]

    max_epochs = 1000

    dl = DataModule(
        path,
        max_len,
        n_steps,
        train_batch_size,
        num_workers,
        save_dir=cache_dir,
        resolution=resolution,
        fps=fps,
        skip_rate=skip_rate,
    )
    model = Model(n_steps=n_steps, last_dim=last_dim)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        max_epochs=max_epochs,
        log_every_n_steps=1,
        accumulate_grad_batches=total_batch_size // train_batch_size,
        benchmark=True,
        deterministic=False,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(model=model, datamodule=dl)
