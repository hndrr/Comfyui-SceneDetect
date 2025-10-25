from __future__ import annotations
from typing import Any, Dict
import os, json, cv2, torch

from ..utils.video_ops import (
    detect_scenes,
    timecodes_to_dict,
    pick_frame_index,
    resize_keep_ar,
    frame_to_tensor_bchw,
)

NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}


class PySceneDetectToImages:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {"multiline": False, "placeholder": "/path/to/video.mp4"},
                ),
                "method": (
                    ["content", "adaptive", "threshold"],
                    {"default": "content"},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 27.0, "min": 0.0, "max": 1000.0, "step": 0.1},
                ),
                "min_scene_len_sec": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "step": 0.05},
                ),
                "min_scene_len_frames": ("INT", {"default": 15, "min": 0, "step": 1}),
                "luma_only": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "representative": (["start", "middle", "end"], {"default": "start"}),
                "max_width": ("INT", {"default": 0, "min": 0, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "step": 1}),
                "limit_scenes": ("INT", {"default": 0, "min": 0, "step": 1}),
                "write_thumbs": ("BOOLEAN", {"default": False}),
                "thumbs_dir": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "空なら video と同ディレクトリ/scene_thumbs",
                    },
                ),
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

        scene_list, fps = detect_scenes(
            video_path,
            method,
            threshold,
            min_scene_len_sec,
            min_scene_len_frames,
            luma_only,
        )
        rows = timecodes_to_dict(scene_list, fps)
        if limit_scenes and limit_scenes > 0:
            rows = rows[:limit_scenes]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV が動画を開けませんでした。")

        images = []
        if write_thumbs:
            if not thumbs_dir:
                base_dir = os.path.dirname(os.path.abspath(video_path))
                thumbs_dir = os.path.join(base_dir, "scene_thumbs")
            os.makedirs(thumbs_dir, exist_ok=True)

        for row in rows:
            fidx = pick_frame_index(row, representative)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            h, w = frame.shape[:2]
            nw, nh = resize_keep_ar(w, h, max_width, max_height)
            if (nw, nh) != (w, h):
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

            images.append(frame_to_tensor_bchw(frame))  # (1,C,H,W)

            if write_thumbs:
                out_name = f"scene_{row['index']:03d}_f{fidx}.jpg"
                cv2.imwrite(os.path.join(thumbs_dir, out_name), frame)

        cap.release()

        if not images:
            # 型整合のためのフォールバック：1x1黒
            import numpy as np

            black = np.zeros((1, 1, 3), dtype=np.uint8)
            images = [frame_to_tensor_bchw(black)]

        batch = torch.cat(images, dim=0)  # (B,C,H,W)

        scenes_json = json.dumps(
            {
                "video_path": os.path.abspath(video_path),
                "fps": fps,
                "method": method,
                "threshold": threshold,
                "min_scene_len_frames": (
                    int(round(min_scene_len_sec * fps))
                    if (min_scene_len_sec and fps > 0)
                    else int(min_scene_len_frames)
                ),
                "representative": representative,
                "scenes": rows,
            },
            ensure_ascii=False,
            indent=2,
        )

        return (batch, scenes_json, len(rows))


NODE_CLASS_MAPPINGS["PySceneDetectToImages"] = PySceneDetectToImages
NODE_DISPLAY_NAME_MAPPINGS["PySceneDetectToImages"] = "PySceneDetect: Scenes → Images"

