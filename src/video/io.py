import multiprocessing as mp
import os
import shutil
from typing import List

import av
import cv2
import ffmpeg
from memory_tempfile import MemoryTempfile
from PIL import Image
from tqdm import tqdm

MAXINT = 2147483647


class Video:
    def __init__(
        self,
        path: str,
        resolution: str = "720:480",
        crf: int = 23,
        fps: int = 30,
        sc_thre: int = 20,
        use_gpu: bool = False,
        gpu_id: int = "any",
        buf_sec: int = 2,
    ) -> None:
        self.original_path = path
        self.resolution = resolution
        self.fps = fps
        self.crf = crf + 4 if use_gpu else crf
        self.sc_thre = sc_thre
        self.use_gpu = use_gpu
        self.buf_sec = buf_sec
        self.gpu_id = gpu_id

        self._tempfile = MemoryTempfile()
        self._tempdir = self._tempfile.TemporaryDirectory()

        self.path = os.path.join(self._tempdir.name, f"video.mp4")
        self.pre_compile(path)

    @property
    def encoder(self):
        if self.use_gpu:
            return "h264_nvenc"
        else:
            return "libx264"

    @property
    def preset(self):
        if self.use_gpu:
            return "slow"
        else:
            return "veryfast"

    def get_options(self, **kwargs):
        options = {
            "c:v": f"{self.encoder}",
            "c:a": "aac",
            "profile:v": "high",
            "b:v": "0",
            "b:a": "128k",
            "pix_fmt": "yuv420p",
            "r": f"{self.fps}",
            "g": f"{MAXINT}",
            "keyint_min": f"{self.fps * self.buf_sec}",
            "sc_threshold": f"{self.sc_thre}",
            "preset": f"{self.preset}",
            "threads": f"{mp.cpu_count()}",
            "subq": "7",
            "qcomp": "0.6",
            "qmin": "10",
            "qmax": "51",
            "trellis": "2",
            "coder": "1",
            "refs": "16",
            "me_range": "16",
            "rc-lookahead": "50",
        }
        if self.use_gpu:
            gpu_options = {
                "rc:v": "vbr_hq",
                "gpu": f"{self.gpu_id}",
                "cq": f"{self.crf}",
                "surfaces": "64",
                "b_adapt": True,
                "b_ref_mode": "middle",
                "i_qfactor": "0.75",
                "b_qfactor": "1.1",
                "aq-strength": "15",
                "2pass": True,
            }

            options.update(gpu_options)
        else:
            cpu_options = {
                "crf": f"{self.crf}",
                "i_qfactor": "0.71",
                "me_method": "umh",
                "b_strategy": "1",
                "b_sensitivity": f"{self.sc_thre}",
                "bf": "16",
            }
            options.update(cpu_options)

        options.update(kwargs)
        return options

    def pre_compile(self, path):
        options = {
            "vf": f"yadif=0:-1:1,scale={self.resolution}",
            "sws_flags": "lanczos+accurate_rnd",
        }
        options = self.get_options(**options)
        ffmpeg.input(path).output(self.path, **options).run(overwrite_output=True)

    def __del__(self):
        self._tempdir.cleanup()
        del self._tempfile

    def find_keyframe_timestamps(self) -> List[int]:
        """Find indices of keyframes .

        Returns:
            List[int]: List of keyframe indices
        """
        out = []
        with av.open(self.path) as container:
            for frame in container.decode(video=0):
                if frame.key_frame:
                    out.append(frame.pts * frame.time_base)
            out.append(frame.pts * frame.time_base)
        return list(map(float, out))

    def _split(self, start, end, output_path):
        options = {
            "keyint_min": f"{MAXINT}",
        }
        options = self.get_options(**options)
        stream = ffmpeg.input(self.path)
        vid = stream.video.trim(start=start, end=end).setpts("PTS-STARTPTS")
        aud = stream.audio.filter_("atrim", start=start, end=end).filter_(
            "asetpts", "PTS-STARTPTS"
        )
        joined = ffmpeg.concat(vid, aud, v=1, a=1).node
        output = ffmpeg.output(joined[0], joined[1], output_path, **options)
        output.run(overwrite_output=True)
        with open(output_path, "rb") as f:
            return f.read()

    def split(self) -> List[av.container.OutputContainer]:
        """Split video by keyframes

        Returns:
            List[bytes]: List of bytes of video segments
        """
        keyframes_indices = self.find_keyframe_timestamps()

        out = []
        with self._tempfile.TemporaryDirectory() as temp_dir:
            for i in tqdm(range(len(keyframes_indices) - 1)):
                start_frame = keyframes_indices[i]
                end_frame = keyframes_indices[i + 1]
                output_path = f"{temp_dir}/{i}.mp4"
                out.append(self._split(start_frame, end_frame, output_path))
        return out

    def get_all_frames(self):
        """Return a list of all PIL Images.

        Returns:
            [List[PIL Image]]:  a list of PIL Image w/o audio
        """
        frames = []
        with av.open(self.path) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_image())
        return frames


if __name__ == "__main__":
    video = Video("video.mp4", use_gpu=True, gpu_id=2)
    containers = video.split()

    shutil.rmtree("output", ignore_errors=True)
    os.mkdir("output")
    for i, container in enumerate(containers):
        with open(f"output/{i}.mp4", "wb") as f:
            f.write(container)
