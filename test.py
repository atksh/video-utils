import sys

import pytorch_lightning as pl

sys.path.append("src")

from video.net import DataModule, Model

if __name__ == "__main__":
    cache_dir = "cache"
    path = ["video2.mp4"]
    total_batch_size = 32
    batch_size = 1
    num_workers = 16

    n_layers = 2
    max_epochs = 1000

    resolusion = "720:480"
    fps = 30
    max_len = 16
    n_steps = 4
    skip_rate = 5

    dl = DataModule(
        path,
        max_len,
        n_steps,
        batch_size,
        num_workers,
        save_dir=cache_dir,
        resolution=resolusion,
        fps=fps,
        skip_rate=skip_rate,
    )
    model = Model(n_layers=n_layers, n_steps=n_steps)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        max_epochs=max_epochs,
        log_every_n_steps=1,
        accumulate_grad_batches=total_batch_size // batch_size,
        benchmark=True,
        deterministic=False,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
    )
    trainer.fit(model=model, datamodule=dl)
