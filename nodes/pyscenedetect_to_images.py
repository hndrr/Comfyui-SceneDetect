from __future__ import annotations
from typing import Any, Dict
import os, json, cv2, torch
import numpy as np

from ..utils.video_ops import (
    DETECTION_METHODS,
    DetectorSettings,
    TensorVideoStream,
    detect_scenes_from_video,
    detector_optional_input_types,
    format_scenes_for_llm,
    timecodes_to_dict,
    pick_frame_index,
    resize_keep_ar,
    frame_to_tensor_bhwc,
    unpack_method_input,
)
from .schema_v3 import _NODE_BASE, common_scene_inputs, io


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


class PySceneDetectToImages(_NODE_BASE):
    if io is None:
        FUNCTION = "run"
        RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "STRING")
        RETURN_NAMES = (
            "images",
            "scenes_json",
            "scene_count",
            "all_scenes_text",
            "per_scene_prompt_list",
        )
        OUTPUT_IS_LIST = (False, False, False, False, True)
        CATEGORY = "Video/PySceneDetect"

        @classmethod
        def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
            optional = {
                "representative": (["start", "middle", "end"], {"default": "start"}),
                "max_width": ("INT", {"default": 0, "min": 0, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "step": 1}),
                "limit_scenes": ("INT", {"default": 0, "min": 0, "step": 1}),
                "write_thumbs": ("BOOLEAN", {"default": False}),
                "thumbs_dir": (
                    "STRING",
                    {"default": "", "placeholder": "Leave empty to use ./scene_thumbs"},
                ),
            }
            optional.update(detector_optional_input_types())
            return {
                "required": {
                    "image": (IMAGE_OR_LATENT, {}),
                    "video_info": ("VHS_VIDEOINFO", {}),
                    "method": (DETECTION_METHODS, {"default": "content"}),
                    "threshold": (
                        "FLOAT",
                        {"default": 27.0, "min": 0.0, "max": 1000.0, "step": 0.1},
                    ),
                    "min_scene_len_sec": (
                        "FLOAT",
                        {"default": 0.0, "min": 0.0, "step": 0.05},
                    ),
                    "min_scene_len_frames": (
                        "INT",
                        {"default": 15, "min": 0, "step": 1},
                    ),
                    "luma_only": ("BOOLEAN", {"default": True}),
                },
                "optional": optional,
            }

    @classmethod
    def define_schema(cls):
        if io is None:
            raise RuntimeError("ComfyUI V3 API is required for DynamicCombo.")
        return io.Schema(
            node_id="PySceneDetectToImages",
            display_name="PySceneDetect: Scenes → Images (Legacy VHS)",
            category="Video/PySceneDetect",
            inputs=[
                io.Image.Input("image"),
                io.Custom("VHS_VIDEOINFO").Input("video_info"),
                *common_scene_inputs(include_split=False),
            ],
            outputs=[
                io.Image.Output("images"),
                io.String.Output("scenes_json"),
                io.Int.Output("scene_count"),
                io.String.Output("all_scenes_text"),
                io.String.Output("per_scene_prompt_list", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, image, video_info, method, **kwargs):
        method, threshold, luma_only, detector_options = unpack_method_input(
            method,
            kwargs.pop("threshold", 27.0),
            kwargs.pop("luma_only", True),
        )
        kwargs.update(detector_options)
        results = cls().run(
            image,
            video_info,
            method,
            threshold=threshold,
            min_scene_len_sec=kwargs.pop("min_scene_len_sec", 0.0),
            min_scene_len_frames=kwargs.pop("min_scene_len_frames", 15),
            luma_only=luma_only,
            **kwargs,
        )
        return io.NodeOutput(*results)

    def run(
        self,
        image: torch.Tensor,
        video_info: Dict[str, Any],
        method: Any,
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
        prompt_template: str = "",
        **detector_options,
    ):
        method, threshold, luma_only, detector_options = unpack_method_input(
            method, threshold, luma_only, detector_options
        )
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
        settings = DetectorSettings.from_mapping(detector_options)

        video = TensorVideoStream(image, fps)
        scene_list, fps_detected = detect_scenes_from_video(
            video,
            method,
            threshold,
            min_scene_len_sec,
            min_scene_len_frames,
            luma_only,
            settings=settings,
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
            if fidx < 0 or fidx >= image.shape[0]:
                continue
            frame = video.frame_at(fidx)

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
        all_scenes_text, per_scene_prompt_list = format_scenes_for_llm(
            rows, prompt_template
        )

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

        return (batch, scenes_json, len(rows), all_scenes_text, per_scene_prompt_list)


NODE_CLASS_MAPPINGS["PySceneDetectToImages"] = PySceneDetectToImages
NODE_DISPLAY_NAME_MAPPINGS["PySceneDetectToImages"] = "PySceneDetect: Scenes → Images (Legacy VHS)"
