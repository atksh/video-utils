import sys

import pytorch_lightning as pl

sys.path.append("src")

from video.net import DataModule, Model

if __name__ == "__main__":
    cache_dir = "cache"
    path = ["video.mp4"]
    total_batch_size = 30
    batch_size = 6
    num_workers = 16

    max_epochs = 1000

    resolution = "256:144"
    fps = 30
    skip_rate = 1
    max_len = 16
    n_steps = 4
    num_mix = 4
    last_dim = 64

    dl = DataModule(
        path,
        max_len,
        n_steps,
        batch_size,
        num_workers,
        save_dir=cache_dir,
        resolution=resolution,
        fps=fps,
        skip_rate=skip_rate,
    )
    model = Model(n_steps=n_steps, last_dim=last_dim, num_mix=num_mix)

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
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(model=model, datamodule=dl)
