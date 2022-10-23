import argparse
import glob
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append("src")

from config import *
from video.callback import SetPrecisionCallback
from video.net import DataModule, Model


def create_dm(video_path_list):
    dl = DataModule(
        video_path_list,
        max_len,
        n_steps,
        train_batch_size,
        num_workers,
        save_dir=cache_dir,
        resolution=resolution,
        fps=fps,
        skip_rate=skip_rate,
    )
    return dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cache_dir = "cache"
    path = list(glob.glob("data/*.mp4"))
    if args.debug:
        path = ["video2.mp4"]
    print(path)

    max_epochs = 1000

    model = Model(
        in_ch=in_ch,
        out_ch=out_ch,
        widths=widths,
        depths=depths,
        heads=heads,
        head_widths=head_widths,
        block_sizes=block_sizes,
        kernel_sizes=kernel_sizes,
        dec_depths=dec_depths,
        resolution_scale=resolution_scale,
    )

    precision_callback = SetPrecisionCallback()
    checkpoint_callback = ModelCheckpoint(save_last=True, every_n_train_steps=32)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        max_epochs=max_epochs,
        log_every_n_steps=1,
        benchmark=True,
        deterministic=False,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        limit_train_batches=1.0 / skip_rate,
        limit_val_batches=1.0 / skip_rate,
        limit_test_batches=1.0 / skip_rate,
        callbacks=[precision_callback, checkpoint_callback],
        accumulate_grad_batches=accumulate_grad_batchs,
    )

    train_dl = create_dm(path)
    trainer.fit(model, datamodule=train_dl)
