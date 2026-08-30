from __future__ import annotations
from contextlib import contextmanager
from fractions import Fraction
import os
import tempfile
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
from scenedetect import FrameTimecode, SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector
from scenedetect.video_stream import VideoStream


class TensorVideoStream(VideoStream):
    BACKEND_NAME = "tensor"

    def __init__(self, frames: torch.Tensor, fps: float):
        self._frames = frames.detach()
        self._frame_rate = Fraction(float(fps)).limit_denominator(1_000_000)
        self._next_frame = 0

        if (
            frames.shape[1] in (1, 3, 4)
            and frames.shape[2] > 4
            and frames.shape[3] > 4
        ):
            self._channel_first = True
            self._height, self._width = int(frames.shape[2]), int(frames.shape[3])
        elif frames.shape[-1] in (1, 3, 4):
            self._channel_first = False
            self._height, self._width = int(frames.shape[1]), int(frames.shape[2])
        else:
            raise ValueError(
                "image cannot be interpreted as (B,C,H,W) or (B,H,W,C)."
            )

        self._normalized = float(self._frames.max().item()) <= 1.0 + 1e-6

    @property
    def path(self) -> str:
        return ""

    @property
    def name(self) -> str:
        return "tensor"

    @property
    def is_seekable(self) -> bool:
        return True

    @property
    def frame_rate(self) -> Fraction:
        return self._frame_rate

    @property
    def duration(self) -> FrameTimecode:
        return FrameTimecode(len(self._frames), self._frame_rate)

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def aspect_ratio(self) -> float:
        return 1.0

    @property
    def position(self) -> FrameTimecode:
        return FrameTimecode(max(0, self._next_frame - 1), self._frame_rate)

    @property
    def position_ms(self) -> float:
        return self.position.seconds * 1000.0

    @property
    def frame_number(self) -> int:
        return self._next_frame

    def frame_at(self, index: int) -> np.ndarray:
        frame = self._frames[index].to(device="cpu")
        if self._channel_first:
            frame = frame.permute(1, 2, 0)
        frame_rgb = frame.numpy()
        if self._normalized:
            frame_rgb = frame_rgb * 255.0
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        if frame_rgb.shape[-1] == 4:
            frame_rgb = frame_rgb[..., :3]
        elif frame_rgb.shape[-1] == 1:
            frame_rgb = np.repeat(frame_rgb, 3, axis=-1)
        return frame_rgb[..., ::-1].copy()

    def read(self, decode: bool = True) -> np.ndarray | bool:
        if self._next_frame >= len(self._frames):
            return False
        index = self._next_frame
        self._next_frame += 1
        return self.frame_at(index) if decode else True

    def reset(self) -> None:
        self._next_frame = 0

    def seek(self, target: Any) -> None:
        frame = FrameTimecode(target, self._frame_rate).frame_num
        if frame < 0:
            raise ValueError("target must be greater than or equal to 0")
        self._next_frame = min(frame, len(self._frames))


def choose_detector(
    method: str, threshold: float, min_scene_len: int | float, luma_only: bool
):
    if method == "adaptive":
        return AdaptiveDetector(min_scene_len=min_scene_len, luma_only=luma_only)
    if method == "threshold":
        # ThresholdDetector does not support luma_only in PySceneDetect 0.7.x.
        return ThresholdDetector(threshold=threshold, min_scene_len=min_scene_len)
    return ContentDetector(
        threshold=threshold, min_scene_len=min_scene_len, luma_only=luma_only
    )


def timecodes_to_dict(
    scene_list: List[Tuple[FrameTimecode, FrameTimecode]],
    fps: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, (start, end) in enumerate(scene_list, start=1):
        s, e = start.frame_num, end.frame_num
        d = max(0, e - s)
        rows.append(
            {
                "index": i,
                "start_frame": s,
                "end_frame": e,
                "duration_frames": d,
                "fps": fps,
                "start_time": start.get_timecode(),
                "end_time": end.get_timecode(),
                "duration_sec": max(0.0, end.seconds - start.seconds),
            }
        )
    return rows


def pick_frame_index(row: Dict[str, Any], representative: str) -> int:
    s, e = row["start_frame"], row["end_frame"]
    if representative == "end":
        return max(s, e - 1)
    if representative == "middle":
        return s + max(0, (e - s) // 2)
    return s


def resize_keep_ar(w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    if max_w <= 0 and max_h <= 0:
        return w, h
    scale = 1.0
    if max_w > 0:
        scale = min(scale, max_w / w)
    if max_h > 0:
        scale = min(scale, max_h / h)
    if scale >= 1.0:
        return w, h
    nw, nh = int(w * scale), int(h * scale)
    return max(nw, 1), max(nh, 1)


def frame_to_tensor_bhwc(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return torch.from_numpy(arr)[None, ...]  # (1,H,W,C)


def detect_scenes(
    video_path: str,
    method: str,
    threshold: float,
    min_scene_len_sec: float,
    min_scene_len_frames: int,
    luma_only: bool,
    start_time: float = 0.0,
    duration: float = 0.0,
):
    video = open_video(video_path)
    if start_time > 0:
        video.seek(start_time)
    return detect_scenes_from_video(
        video,
        method,
        threshold,
        min_scene_len_sec,
        min_scene_len_frames,
        luma_only,
        duration,
    )


def detect_scenes_from_video(
    video: VideoStream,
    method: str,
    threshold: float,
    min_scene_len_sec: float,
    min_scene_len_frames: int,
    luma_only: bool,
    duration: float = 0.0,
):
    fps = float(getattr(video, "frame_rate", 0.0))
    min_scene_len_seconds = max(0.0, float(min_scene_len_sec))
    min_scene_len = (
        min_scene_len_seconds
        if min_scene_len_seconds > 0
        else max(0, int(min_scene_len_frames))
    )
    manager = SceneManager()

    try:
        detector = choose_detector(method, threshold, min_scene_len, luma_only)
        manager.add_detector(detector)
        manager.detect_scenes(
            video=video,
            duration=duration if duration > 0 else None,
            show_progress=False,
        )
        scene_list = manager.get_scene_list()
    finally:
        # Ensure file-backed streams are released even if detection raises.
        release = getattr(video, "release", None)
        if callable(release):
            release()

    return scene_list, fps


@contextmanager
def video_source_path(source: Any):
    if isinstance(source, (str, os.PathLike)):
        yield os.fspath(source)
        return

    with tempfile.TemporaryDirectory(prefix="scenedetect_source_") as tmpdir:
        path = os.path.join(tmpdir, "input.video")
        source.seek(0)
        with open(path, "wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
        source.seek(0)
        yield path


def read_video_frames(video_path: str, frame_indices: List[int]) -> Dict[int, np.ndarray]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: Dict[int, np.ndarray] = {}
    try:
        for index in sorted(set(frame_indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                frames[index] = frame
    finally:
        capture.release()
    return frames
