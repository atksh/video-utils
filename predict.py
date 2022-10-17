import sys
import os
import shutil
import torch
import glob

import pytorch_lightning as pl
from PIL import Image

sys.path.append("src")

from video.net import Model
from video.dataset import VideoDatasetForInference

if __name__ == "__main__":
    resolusion = "640:360"
    batch_size = 1
    fps = 30
    skip_rate = 3
    max_len = 8
    n_steps = 4
    num_mix = 4
    last_dim = 64

    dataset = VideoDatasetForInference(
        "video2.mp4", max_len, n_steps, resolusion, fps=fps, skip_rate=skip_rate
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    model = Model(n_steps=n_steps, last_dim=last_dim, num_mix=num_mix)

    version = 23
    ckpt_path = glob.glob(f"lightning_logs/version_{version}/checkpoints/*.ckpt")[-1]
    model = Model.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        n_steps=n_steps,
        last_dim=last_dim,
        num_mix=num_mix,
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
