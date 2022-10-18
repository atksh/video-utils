import argparse
import glob
import os
import shutil
import sys

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
    ckpt_path = glob.glob(f"lightning_logs/version_{version}/checkpoints/*.ckpt")[-1]
    model = Model.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        n_steps=n_steps,
        last_dim=last_dim,
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
    for t in range(n_steps):
        os.makedirs(f"out/{t}", exist_ok=True)
        for i, (pred, gold) in enumerate(zip(preds, golds)):
            pred_img = Image.fromarray(pred[t].astype("uint8"))
            gold_img = Image.fromarray(gold[t].astype("uint8"))
            pred_img.save(f"out/{t}/{i}_pred.png")
            gold_img.save(f"out/{t}/{i}_gold.png")
