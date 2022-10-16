import io
import os
import pickle
import random
import tempfile

import av
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .reader import Video


class VideoDataset(Dataset):
    def __init__(
        self,
        video_path_or_serialized: str,
        max_len: int,
        n_steps: int,
        resolution: str,
        crf=23,
        fps=30,
        sc_thre=40,
        use_gpu=True,
        gpu_id="2",
        buf_sec=1,
        skip_rate=6,
    ):
        self.max_len = max_len
        self.n_steps = n_steps
        self.skip_rate = skip_rate
        if isinstance(video_path_or_serialized, bytes):
            self.videos = self.deserialize(video_path_or_serialized)
        else:
            video_path = video_path_or_serialized
            video = Video(
                video_path, resolution, crf, fps, sc_thre, use_gpu, gpu_id, buf_sec
            )
            self.videos = video.split()

        self.to_tensor = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
            ]
        )

        self.n, self.to_index = self.calc_len()

    def __len__(self):
        return self.n

    @staticmethod
    def deserialize(video_bytes):
        return pickle.loads(video_bytes)

    def serialize(self):
        return pickle.dumps(self.videos)

    def calc_len(self):
        n = 0
        to_index = {}
        for i in range(len(self.videos)):
            frames = self.get_frames(i)
            d = max(1, len(frames) - self.max_len - self.n_steps + 1)
            for j in range(d):
                m = n + j
                to_index[m] = i
            n += d
        return n, to_index

    def get_frames(self, index):
        video_bytes = self.videos[index]
        frames = []
        with io.BytesIO(video_bytes) as f:
            with av.open(f, "r") as container:
                for frame in container.decode(video=0):
                    frames.append(frame.to_image())
        return frames[:: self.skip_rate]

    def __getitem__(self, index):
        frames = self.get_frames(self.to_index[index])
        max_len = self.max_len + self.n_steps
        if len(frames) > max_len:
            idx = random.randint(0, len(frames) - max_len)
            frames = frames[idx : idx + max_len]
        elif len(frames) < max_len:
            frames = [frames[0]] * (max_len - len(frames)) + frames
        assert len(frames) == max_len
        frames = torch.stack([self.to_tensor(frame) for frame in frames])
        input_frames = frames[: self.max_len]
        target_frames = frames[self.max_len :]
        return input_frames, target_frames
