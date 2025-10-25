# pyscenedetect_nodes.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from scenedetect import SceneManager, open_video
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ------------ ユーティリティ ------------

def _choose_detector(method: str, threshold: float, min_scene_len: int, luma_only: bool):
    if method == "adaptive":
        return AdaptiveDetector(min_scene_len=min_scene_len, luma_only=luma_only)
    if method == "threshold":
        return ThresholdDetector(threshold=threshold, min_scene_len=min_scene_len, luma_only=luma_only)
    return ContentDetector(threshold=threshold, min_scene_len=min_scene_len, luma_only=luma_only)

def _timecodes_to_dict(scene_list: List[Tuple[FrameTimecode, FrameTimecode]], fps: float) -> List[Dict[str, Any]]:
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

def _pick_frame_index(row: Dict[str, Any], representative: str) -> int:
    s, e = row["start_frame"], row["end_frame"]
    if representative == "end":
        return max(s, e - 1)
    if representative == "middle":
        return s + max(0, (e - s) // 2)
    return s  # start

def _resize_keep_ar(w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int]:
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

def _frame_to_tensor_bchw(frame_bgr: np.ndarray) -> torch.Tensor:
    # BGR -> RGB, HWC[0..255] -> CHW[0..1], add batch dim
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    if arr.ndim == 2:  # safety
        arr = np.repeat(arr[..., None], 3, axis=2)
    chw = np.transpose(arr, (2, 0, 1))  # C,H,W
    ten = torch.from_numpy(chw)[None, ...]  # 1,C,H,W
    return ten

# ------------ ノード本体 ------------

class PySceneDetectToImages:
    """
    PySceneDetectでシーン境界を検出し、各シーンの代表フレームをIMAGEバッチとして返すノード。
    併せてメタデータJSON（STRING）とシーン数（INT）も返す。
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "video_path": ("STRING", {"multiline": False, "placeholder": "/path/to/video.mp4"}),
                "method": (["content", "adaptive", "threshold"], {"default": "content"}),
                "threshold": ("FLOAT", {"default": 27.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "min_scene_len_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.05}),
                "min_scene_len_frames": ("INT", {"default": 15, "min": 0, "step": 1}),
                "luma_only": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "representative": (["start", "middle", "end"], {"default": "start"}),
                "max_width": ("INT", {"default": 0, "min": 0, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "step": 1}),
                "limit_scenes": ("INT", {"default": 0, "min": 0, "step": 1}),  # 0なら制限なし
                "write_thumbs": ("BOOLEAN", {"default": False}),
                "thumbs_dir": ("STRING", {"default": "", "placeholder": "空なら video と同ディレクトリ/scene_thumbs"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "scenes_json", "scene_count")
    FUNCTION = "run"
    CATEGORY = "Video/PySceneDetect"

    def run(
        self,
        video_path: str,
        method: str,
        threshold: float,
        min_scene_len_sec: float,
        min_scene_len_frames: int,
        luma_only: bool,
        representative: str = "start",
        max_width: int = 0,
        max_height: int = 0,
        limit_scenes: int = 0,
        write_thumbs: bool = False,
        thumbs_dir: str = "",
    ):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"video_path が存在しません: {video_path}")

        # 1) シーン検出
        video = open_video(video_path)
        fps = float(getattr(video, "frame_rate", 0.0))
        if min_scene_len_sec and fps > 0:
            min_scene_len = max(0, int(round(min_scene_len_sec * fps)))
        else:
            min_scene_len = max(0, int(min_scene_len_frames))

        detector = _choose_detector(method, threshold, min_scene_len, luma_only)
        manager = SceneManager()
        manager.add_detector(detector)
        manager.detect_scenes(video=video, show_progress=False)
        scene_list = manager.get_scene_list()
        rows = _timecodes_to_dict(scene_list, fps)

        if limit_scenes and limit_scenes > 0:
            rows = rows[:limit_scenes]

        # 2) 代表フレーム抽出 -> IMAGEバッチ
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV が動画を開けませんでした。")

        images_bchw: List[torch.Tensor] = []
        if write_thumbs:
            if not thumbs_dir:
                base_dir = os.path.dirname(os.path.abspath(video_path))
                thumbs_dir = os.path.join(base_dir, "scene_thumbs")
            os.makedirs(thumbs_dir, exist_ok=True)

        for row in rows:
            fidx = _pick_frame_index(row, representative)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok or frame is None:
                # 読み込めない場合はスキップ（行を残しつつダミーは作らない）
                continue

            # 任意縮小
            h, w = frame.shape[:2]
            nw, nh = _resize_keep_ar(w, h, max_width, max_height)
            if (nw, nh) != (w, h):
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

            # 画像テンソルへ
            ten = _frame_to_tensor_bchw(frame)  # 1,C,H,W
            images_bchw.append(ten)

            # 任意保存
            if write_thumbs:
                out_name = f"scene_{row['index']:03d}_f{fidx}.jpg"
                out_path = os.path.join(thumbs_dir, out_name)
                cv2.imwrite(out_path, frame)

        cap.release()

        # 画像が一枚も取れなかった場合のフォールバック
        if len(images_bchw) == 0:
            # 空の1x1黒画像を返す（ComfyUIの型整合のため）
            black = np.zeros((1, 1, 3), dtype=np.uint8)
            ten = _frame_to_tensor_bchw(black)
            images_bchw = [ten]

        # (B,C,H,W) に結合
        images = torch.cat(images_bchw, dim=0)

        scenes_json = json.dumps(
            {
                "video_path": os.path.abspath(video_path),
                "fps": fps,
                "method": method,
                "threshold": threshold,
                "min_scene_len_frames": min_scene_len,
                "representative": representative,
                "scenes": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        return (images, scenes_json, len(rows))


NODE_CLASS_MAPPINGS["PySceneDetectToImages"] = PySceneDetectToImages
NODE_DISPLAY_NAME_MAPPINGS["PySceneDetectToImages"] = "PySceneDetect: Scenes → Images"
