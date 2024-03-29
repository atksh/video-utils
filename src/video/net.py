import hashlib
import os

import pytorch_lightning as pl
import torch
from adabelief_pytorch import AdaBelief
from tqdm import tqdm

from .dataset import VideoDataset
from .nn.loss import MSSSIML1Loss
from .nn.module import VideoModel, make_fused_model_loss


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
        skip_rate=1,
    ):
        super().__init__()
        self.video_path_list = video_path_list
        self.max_len = max_len
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.fps = fps
        self.skip_rate = skip_rate

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
            skip_rate=self.skip_rate,
        )

    def to_serialized_path(self, path):
        # calc md5 of path content
        m = hashlib.md5()
        with open(path, "rb") as f:
            m.update(f.read())
        md5 = m.hexdigest()
        res = self.resolution.replace(":", "x")
        return os.path.join(self.save_dir, md5 + f"_{res}.bin")

    def prepare_data(self):
        datasets = []
        for path in tqdm(self.video_path_list):
            ds = self.create_dataset(path)
            cache_path = self.to_serialized_path(path)
            if not os.path.exists(cache_path):
                serialized = ds.serialize()
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
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )


class Model(pl.LightningModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        widths,
        depths,
        heads,
        head_widths,
        block_sizes,
        kernel_sizes,
        dec_depths,
        resolution_scale,
    ):
        super().__init__()
        model = VideoModel(
            in_ch,
            out_ch,
            widths,
            depths,
            heads,
            head_widths,
            block_sizes,
            kernel_sizes,
            dec_depths,
            resolution_scale,
        )
        loss = MSSSIML1Loss()
        self.model_loss = make_fused_model_loss(model, loss)

    def forward(self, x):
        return self.model_loss(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.model_loss(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.model_loss(x, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        video, y = batch
        with torch.inference_mode():
            preds = self.forward(video)
            preds = (preds * 255).to(torch.uint8)
            y = (y * 255).to(torch.uint8)
        return preds, y

    def configure_optimizers(self):
        return AdaBelief(
            self.parameters(),
            lr=4e-5,
            weight_decay=1e-4,
            print_change_log=False,
            eps=1e-12,
        )
