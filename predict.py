import argparse
import os
import shutil
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

sys.path.append("src")

from config import *
from video.dataset import VideoDatasetForInference
from video.net import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=int)
    parser.add_argument("video_path", type=str)
    args = parser.parse_args()

    dataset = VideoDatasetForInference(
        args.video_path, max_len, n_steps, resolution, fps=fps, skip_rate=skip_rate
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=predict_batch_size,
        shuffle=False,
        num_workers=2,
    )

    version = args.version
    ckpt_path = f"lightning_logs/version_{version}/checkpoints/last.ckpt"
    model = Model.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        in_dim=in_dim,
        stem_dim=stem_dim,
        widths=widths,
        depths=depths,
        heads=heads,
    )
    model.eval()

    trainer = pl.Trainer(accelerator="gpu", devices=[1])
    out = trainer.predict(model, dataloader)
    preds = torch.cat([p[0] for p in out], dim=0)
    golds = torch.cat([p[1] for p in out], dim=0)
    preds = preds.permute(0, 1, 3, 4, 2).numpy()
    golds = golds.permute(0, 1, 3, 4, 2).numpy()

    shutil.rmtree("out", ignore_errors=True)
    os.makedirs("out", exist_ok=True)
    for i, (pred, gold) in enumerate(zip(preds, golds)):
        imgs = []
        for t in range(n_steps):
            # stack with height
            img = np.concatenate([pred[t], gold[t]], axis=0)
            # to gray scale and to rgb
            prev = golds[max(0, i - t - 1)][t]
            diff = np.abs(prev - pred[t])
            diff = np.mean(diff, axis=2).astype(np.uint8)
            diff = np.stack([diff, diff, diff], axis=2)
            diffprev = np.concatenate([prev, diff], axis=0)
            img = np.concatenate([img, diffprev], axis=1)
            imgs.append(img)
        # stack with width
        img = np.concatenate(imgs, axis=1)
        img = Image.fromarray(img)
        img.save(f"out/{i}.png")
