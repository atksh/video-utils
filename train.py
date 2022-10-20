import glob
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append("src")

from config import *
from video.net import DataModule, Model

if __name__ == "__main__":
    cache_dir = "cache"
    path = list(glob.glob("data/*.mp4"))
    print(path)

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
    model = Model(
        backbone_feat_dims,
        front_feat_dims,
        enc_num_heads,
        enc_num_layers,
        dec_num_heads,
        dec_num_layers,
        last_dim,
        n_steps,
    )

    checkpoint_callback = ModelCheckpoint(save_last=True, every_n_train_steps=100)
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
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        limit_train_batches=1.0 / skip_rate,
        limit_val_batches=1.0 / skip_rate,
        limit_test_batches=1.0 / skip_rate,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=dl)
