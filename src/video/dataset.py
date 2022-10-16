import glob
import os

from memory_tempfile import MemoryTempfile
from torch.utils.data import Dataset

from .io import Video


class VideoDataset(Dataset):
    def __init__(
        self,
        video_path,
        resolution="720:480",
        crf=23,
        fps=30,
        sc_thre=20,
        use_gpu=True,
        gpu_id="2",
        buf_sec=2,
    ):
        video = Video(
            video_path, resolution, crf, fps, sc_thre, use_gpu, gpu_id, buf_sec
        )
        self.videos = video.split()
        self._tempfile = MemoryTempfile()
        self._tempdir = self._tempfile.TemporaryDirectory()

    @property
    def dname(self):
        return self._tempdir.name

    def __del__(self):
        self._tempdir.cleanup()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_bytes = self.videos[index]
        path = os.path.join(self.dname, f"video_{index}.mp4")
        with open(path, "wb") as f:
            f.write(video_bytes)
        frames: "List[PIL.Image]" = Video(path).get_all_frames()
        n_frames = len(frames)
