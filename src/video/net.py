import os
import hashlib

import pytorch_lightning as pl
import torch
from tqdm import tqdm

from .dataset import VideoDataset
from .nn.model import Decoder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision("medium")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        video_path_list,
        max_len,
        n_steps,
        batch_size,
        num_workers,
        resolution,
        fps,
        save_dir=None,
    ):
        super().__init__()
        self.video_path_list = video_path_list
        self.max_len = max_len
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.fps = fps

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), ".cache")
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def create_dataset(self, path):
        cache_path = self.to_serialized_path(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                path = f.read()

        return VideoDataset(
            path,
            self.max_len,
            self.n_steps,
            resolution=self.resolution,
            fps=self.fps,
        )

    def to_serialized_path(self, path):
        # calc md5 of path
        m = hashlib.md5()
        m.update(path.encode("utf-8"))
        md5 = m.hexdigest()
        return os.path.join(self.save_dir, f"{md5}.bin")

    def prepare_data(self):
        datasets = []
        for path in tqdm(self.video_path_list):
            ds = self.create_dataset(path)
            serialized = ds.serialize()
            cache_path = self.to_serialized_path(path)
            with open(cache_path, "wb") as f:
                f.write(serialized)
            datasets.append(ds)
        self.ds = torch.utils.data.ConcatDataset(datasets)

    def setup(self, stage=None):
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.ds,
            [int(len(self.ds) * 0.8), len(self.ds) - int(len(self.ds) * 0.8)],
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class Model(pl.LightningModule):
    def __init__(self, n_layers=1, output_dim=3, last_dim=32, n_steps=1):
        super().__init__()
        self.model = Decoder(n_layers, output_dim, last_dim, n_steps)
        self.loss = self.model.loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
