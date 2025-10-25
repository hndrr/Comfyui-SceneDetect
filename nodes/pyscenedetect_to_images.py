from __future__ import annotations
from typing import Any, Dict
import os, json, cv2, torch, tempfile
import numpy as np

from ..utils.video_ops import (
    detect_scenes,
    timecodes_to_dict,
    pick_frame_index,
    resize_keep_ar,
    frame_to_tensor_bhwc,
)


class _MultiInput(str):
    def __new__(cls, name: str, allowed_types="*"):
        res = super().__new__(cls, name)
        res.allowed_types = allowed_types
        return res

    def __ne__(self, other: Any) -> bool:
        allowed = getattr(self, "allowed_types", "*")
        if allowed == "*" or other == "*":
            return False
        return other not in allowed


IMAGE_OR_LATENT = _MultiInput("IMAGE", ["IMAGE", "LATENT"])

NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}


class PySceneDetectToImages:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": (IMAGE_OR_LATENT, {}),
                "video_info": ("VHS_VIDEOINFO", {}),
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
                "thumbs_dir": ("STRING", {"default": "", "placeholder": "Leave empty to use ./scene_thumbs"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "scenes_json", "scene_count")
    FUNCTION = "run"
    CATEGORY = "Video/PySceneDetect"

    def run(
        self,
        image: torch.Tensor,
        video_info: Dict[str, Any],
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
        if isinstance(image, dict) and "samples" in image:
            raise ValueError("LATENT tensors from VAE outputs are not supported. Disconnect the VAE from the Load Video node.")
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] == 0:
            raise ValueError("image must be a tensor with shape (B,H,W,C) or (B,C,H,W).")

        if not isinstance(video_info, dict):
            raise ValueError("Connect the fourth output from Load Video (VHS) to video_info.")
        fps = float(video_info.get("loaded_fps", 0.0) or 0.0)
        if fps <= 0:
            fps = float(video_info.get("source_fps", 0.0) or 0.0)
        if fps <= 0:
            raise ValueError("video_info does not contain a valid FPS value.")

        def _jsonable(val: Any):
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                return float(val)
            return val

        video_info_json = {k: _jsonable(v) for k, v in video_info.items()}

        frames_cpu = image.detach().cpu()
        if frames_cpu.shape[1] in (1, 3, 4) and frames_cpu.shape[2] > 4 and frames_cpu.shape[3] > 4:
            frames_np = frames_cpu.numpy().transpose(0, 2, 3, 1)
        elif frames_cpu.shape[-1] in (1, 3, 4):
            frames_np = frames_cpu.numpy()
        else:
            raise ValueError("image cannot be interpreted as (B,C,H,W) or (B,H,W,C).")

        max_val = float(frames_cpu.max().item())
        if max_val <= 1.0 + 1e-6:
            frames_rgb = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
        else:
            frames_rgb = np.clip(frames_np, 0, 255).astype(np.uint8)

        if frames_rgb.shape[-1] == 4:
            frames_rgb = frames_rgb[..., :3]
        if frames_rgb.shape[-1] == 1:
            frames_rgb = np.repeat(frames_rgb, 3, axis=-1)

        height, width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
        frames_bgr = [frame[..., ::-1].copy() for frame in frames_rgb]

        with tempfile.TemporaryDirectory(prefix="scenedetect_") as tmpd:
            tmp_video = os.path.join(tmpd, "input.avi")

            def _open_writer(path: str):
                for code in ("MJPG", "mp4v", "XVID"):
                    writer = cv2.VideoWriter(
                        path,
                        cv2.VideoWriter_fourcc(*code),
                        float(fps),
                        (int(width), int(height)),
                    )
                    if writer.isOpened():
                        return writer
                return None

            writer = _open_writer(tmp_video)
            if writer is None:
                raise RuntimeError("Failed to create temporary video: no compatible codec found.")

            try:
                for frame_bgr in frames_bgr:
                    writer.write(frame_bgr)
            finally:
                writer.release()

            scene_list, fps_detected = detect_scenes(
                tmp_video,
                method,
                threshold,
                min_scene_len_sec,
                min_scene_len_frames,
                luma_only,
            )

        if fps_detected > 0:
            fps = fps_detected

        rows = timecodes_to_dict(scene_list, fps)
        if limit_scenes and limit_scenes > 0:
            rows = rows[:limit_scenes]

        image_tensors = []
        if write_thumbs:
            if not thumbs_dir:
                thumbs_dir = os.path.join(os.getcwd(), "scene_thumbs")
            os.makedirs(thumbs_dir, exist_ok=True)

        for row in rows:
            fidx = pick_frame_index(row, representative)
            if fidx < 0 or fidx >= len(frames_bgr):
                continue
            frame = frames_bgr[fidx]

            h, w = frame.shape[:2]
            nw, nh = resize_keep_ar(w, h, max_width, max_height)
            if (nw, nh) != (w, h):
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

            image_tensors.append(frame_to_tensor_bhwc(frame))  # (1,H,W,C)

            if write_thumbs:
                out_name = f"scene_{row['index']:03d}_f{fidx}.jpg"
                cv2.imwrite(os.path.join(thumbs_dir, out_name), frame)

        if not image_tensors:
            # Fallback for type consistency: 1x1 black frame
            black = np.zeros((1, 1, 3), dtype=np.uint8)
            image_tensors = [frame_to_tensor_bhwc(black)]

        batch = torch.cat(image_tensors, dim=0)  # (B,H,W,C)

        scenes_json = json.dumps(
            {
                "video_path": "",
                "video_info": video_info_json,
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
