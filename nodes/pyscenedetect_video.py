from __future__ import annotations
from typing import Any, Dict
import json, ntpath, os, cv2, torch
import numpy as np
import folder_paths

from ..utils.video_ops import (
    DETECTION_METHODS,
    DetectorSettings,
    detect_scenes,
    detector_optional_input_types,
    format_scenes_for_llm,
    frame_to_tensor_bhwc,
    load_video_from_file,
    pick_frame_index,
    read_video_frames,
    resize_keep_ar,
    sanitize_clip_name,
    split_scene_clips,
    timecodes_to_dict,
    video_source_path,
)


def _resolve_output_path(output_root: str, relative_path: str) -> str:
    if not relative_path or os.path.isabs(relative_path) or ntpath.isabs(relative_path):
        raise ValueError(
            "Path must be relative to ComfyUI's output directory."
        )
    if ".." in relative_path.replace("\\", "/").split("/"):
        raise ValueError("Path must not contain '..'.")

    path = os.path.realpath(os.path.abspath(os.path.join(output_root, relative_path)))
    if not folder_paths.is_within_directory(output_root, path):
        raise ValueError("Path must stay inside ComfyUI's output directory.")
    return path


# Keep the previous helper name for existing tests.
_resolve_thumbnail_path = _resolve_output_path


class PySceneDetectVideo:
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
                {
                    "default": "",
                    "placeholder": "Relative to ComfyUI output; default: scene_thumbs",
                },
            ),
            "split_clips": ("BOOLEAN", {"default": False}),
            "split_dir": (
                "STRING",
                {
                    "default": "",
                    "placeholder": "Relative to ComfyUI output; default: scene_clips",
                },
            ),
            "split_reencode": ("BOOLEAN", {"default": False}),
        }
        optional.update(detector_optional_input_types())
        return {
            "required": {
                "video": ("VIDEO", {}),
                "method": (DETECTION_METHODS, {"default": "content"}),
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
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = (
        "images",
        "scenes_json",
        "scene_count",
        "all_scenes_text",
        "per_scene_prompt",
        "videos",
    )
    OUTPUT_IS_LIST = (False, False, False, False, True, True)
    FUNCTION = "run"
    CATEGORY = "Video/PySceneDetect"

    def run(
        self,
        video: Any,
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
        split_clips: bool = False,
        split_dir: str = "",
        split_reencode: bool = False,
        prompt_template: str = "",
        **detector_options,
    ):
        fps = float(video.get_frame_rate())
        if fps <= 0:
            raise ValueError("video does not contain a valid FPS value.")

        width, height = video.get_dimensions()
        frame_count = video.get_frame_count()
        video_duration = video.get_duration()
        start_time, trim_duration = video.get_active_trim_window()
        settings = DetectorSettings.from_mapping(detector_options)
        output_root = folder_paths.get_output_directory()

        videos = []
        with video_source_path(video.get_stream_source()) as video_path:
            scene_list, fps_detected = detect_scenes(
                video_path,
                method,
                threshold,
                min_scene_len_sec,
                min_scene_len_frames,
                luma_only,
                start_time,
                trim_duration,
                settings=settings,
            )
            if fps_detected > 0:
                fps = fps_detected

            if limit_scenes and limit_scenes > 0:
                scene_list = scene_list[:limit_scenes]
            rows = timecodes_to_dict(scene_list, fps)

            frame_indices = [pick_frame_index(row, representative) for row in rows]
            frames = read_video_frames(video_path, frame_indices)

            if split_clips and scene_list:
                clip_subdir = split_dir.strip() or "scene_clips"
                clip_dir = _resolve_output_path(output_root, clip_subdir)
                clip_paths = split_scene_clips(
                    video_path,
                    scene_list,
                    clip_dir,
                    video_name=sanitize_clip_name(video_path),
                    reencode=split_reencode,
                )
                for row, clip_path in zip(rows, clip_paths):
                    row["clip_path"] = clip_path
                videos = [load_video_from_file(path) for path in clip_paths]

        image_tensors = []
        thumbnail_subdir = thumbs_dir.strip() or "scene_thumbs"
        if write_thumbs:
            thumbnail_dir = _resolve_output_path(output_root, thumbnail_subdir)
            os.makedirs(thumbnail_dir, exist_ok=True)

        for row, frame_index in zip(rows, frame_indices):
            frame = frames.get(frame_index)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            new_width, new_height = resize_keep_ar(w, h, max_width, max_height)
            if (new_width, new_height) != (w, h):
                frame = cv2.resize(
                    frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

            image_tensors.append(frame_to_tensor_bhwc(frame))

            if write_thumbs:
                out_name = f"scene_{row['index']:03d}_f{frame_index}.jpg"
                out_path = _resolve_output_path(
                    output_root, os.path.join(thumbnail_subdir, out_name)
                )
                cv2.imwrite(out_path, frame)

        if not image_tensors:
            black = np.zeros((1, 1, 3), dtype=np.uint8)
            image_tensors = [frame_to_tensor_bhwc(black)]

        batch = torch.cat(image_tensors, dim=0)
        video_info = {
            "source_fps": fps,
            "source_frame_count": frame_count,
            "source_duration": video_duration,
            "source_width": width,
            "source_height": height,
            "loaded_fps": fps,
            "loaded_frame_count": frame_count,
            "loaded_duration": video_duration,
            "loaded_width": width,
            "loaded_height": height,
            "trim_start_sec": start_time,
            "trim_duration_sec": trim_duration,
        }
        all_scenes_text, per_scene_prompt = format_scenes_for_llm(
            rows, prompt_template
        )
        scenes_json = json.dumps(
            {
                "video_path": "",
                "video_info": video_info,
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

        return (
            batch,
            scenes_json,
            len(rows),
            all_scenes_text,
            per_scene_prompt,
            videos,
        )


NODE_CLASS_MAPPINGS = {"PySceneDetectVideo": PySceneDetectVideo}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PySceneDetectVideo": "PySceneDetect: Video → Scenes"
}
