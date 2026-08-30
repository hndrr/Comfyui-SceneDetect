from __future__ import annotations
from typing import Any, Dict, List
import os
import shutil
import uuid

import folder_paths


def _as_video_list(video: Any) -> List[Any]:
    if video is None:
        return []
    if isinstance(video, list):
        return video
    return [video]


def _file_path_from_video(video: Any) -> str | None:
    source = getattr(video, "get_stream_source", None)
    if callable(source):
        value = source()
        if isinstance(value, (str, os.PathLike)) and os.path.isfile(value):
            return os.fspath(value)
    path = getattr(video, "path", None)
    if isinstance(path, (str, os.PathLike)) and os.path.isfile(path):
        return os.fspath(path)
    if isinstance(video, (str, os.PathLike)) and os.path.isfile(video):
        return os.fspath(video)
    return None


def preview_entry_for_path(path: str) -> Dict[str, str]:
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        raise ValueError(f"Video file was not found: {path}")

    temp_root = os.path.realpath(folder_paths.get_temp_directory())
    if folder_paths.is_within_directory(temp_root, path):
        relative = os.path.relpath(path, temp_root)
        subfolder, filename = os.path.split(relative)
        return {
            "filename": filename,
            "subfolder": "" if subfolder == "." else subfolder.replace("\\", "/"),
            "type": "temp",
        }

    dest_dir = os.path.join(temp_root, "scenedetect_preview")
    os.makedirs(dest_dir, exist_ok=True)
    dest_name = f"{uuid.uuid4().hex}{os.path.splitext(path)[1] or '.mp4'}"
    shutil.copy2(path, os.path.join(dest_dir, dest_name))
    return {
        "filename": dest_name,
        "subfolder": "scenedetect_preview",
        "type": "temp",
    }


def preview_entry_for_video(video: Any) -> Dict[str, str]:
    path = _file_path_from_video(video)
    if path is not None:
        return preview_entry_for_path(path)

    save_to = getattr(video, "save_to", None)
    if not callable(save_to):
        raise ValueError("Preview Videos requires a file-backed VIDEO input.")

    dest_dir = os.path.join(folder_paths.get_temp_directory(), "scenedetect_preview")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{uuid.uuid4().hex}.mp4")
    save_to(dest_path)
    return preview_entry_for_path(dest_path)


class PySceneDetectPreviewVideos:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "Connect a VIDEO or the scene_videos list from PySceneDetect: Video → Scenes. Files stay in temp; nothing is written to output.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "Video/PySceneDetect"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    DESCRIPTION = (
        "Preview VIDEO clips without saving them to the output directory. "
        "Connect `scene_videos` from PySceneDetect: Video → Scenes after enabling split_clips."
    )

    def preview(self, video: Any):
        results = [
            preview_entry_for_video(item)
            for item in _as_video_list(video)
            if item is not None
        ]
        # Do not send `images`/`animated`: ComfyUI's native video preview
        # preloads every URL, which is too heavy for large scene counts.
        # The frontend loads a single <video> and switches by scene number.
        return {"ui": {"videos": results}}


NODE_CLASS_MAPPINGS = {"PySceneDetectPreviewVideos": PySceneDetectPreviewVideos}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PySceneDetectPreviewVideos": "PySceneDetect: Preview Videos"
}
