from __future__ import annotations
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
from scenedetect import SceneManager, open_video
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector


def choose_detector(method: str, threshold: float, min_scene_len: int, luma_only: bool):
    if method == "adaptive":
        return AdaptiveDetector(min_scene_len=min_scene_len, luma_only=luma_only)
    if method == "threshold":
        # ThresholdDetector does not support luma_only in PySceneDetect 0.6.x.
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
        s, e = int(start.get_frames()), int(end.get_frames())
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
                "duration_sec": (d / fps) if fps > 0 else None,
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
):
    video = open_video(video_path)
    fps = float(getattr(video, "frame_rate", 0.0))
    min_scene_len = (
        max(0, int(round(min_scene_len_sec * fps)))
        if (min_scene_len_sec and fps > 0)
        else max(0, int(min_scene_len_frames))
    )
    manager = SceneManager()

    try:
        detector = choose_detector(method, threshold, min_scene_len, luma_only)
        manager.add_detector(detector)
        manager.detect_scenes(video=video, show_progress=False)
        scene_list = manager.get_scene_list()
    finally:
        # Ensure the temp video file is released even if detection raises.
        if hasattr(video, "release"):
            video.release()
        if hasattr(video, "close"):
            video.close()
        del video
    
    return scene_list, fps
