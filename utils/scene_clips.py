from __future__ import annotations
import os
import re
import shutil
from typing import Any, List, Tuple
from scenedetect import FrameTimecode, is_ffmpeg_available, split_video_ffmpeg

FFMPEG_COPY_ARGS = "-map 0:v:0 -map 0:a? -map 0:s? -c copy"
FFMPEG_REENCODE_ARGS = (
    "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 22 -c:a aac"
)


def sanitize_clip_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "scene"


def _collect_scene_clip_paths(
    output_dir: str, clip_name: str, scene_count: int
) -> List[str]:
    pattern = re.compile(rf"^{re.escape(clip_name)}-Scene-(\d+)\.mp4$")
    found: List[Tuple[int, str]] = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(output_dir, name)))
    found.sort(key=lambda item: item[0])
    if len(found) != scene_count:
        raise RuntimeError(
            f"Expected {scene_count} scene clips in {output_dir}, found {len(found)}."
        )
    return [path for _, path in found]


def split_scene_clips(
    video_path: str,
    scene_list: List[Tuple[FrameTimecode, FrameTimecode]],
    output_dir: str,
    video_name: str = "scene",
    reencode: bool = True,
) -> List[str]:
    if not scene_list:
        return []
    if not is_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to split scene clips. Install ffmpeg and ensure it is on PATH."
        )

    os.makedirs(output_dir, exist_ok=True)
    clip_name = sanitize_clip_name(video_name)
    attempts = [True] if reencode else [False, True]
    last_error = "ffmpeg failed to split scene clips."
    for attempt_reencode in attempts:
        ret = split_video_ffmpeg(
            video_path,
            scene_list,
            output_dir=output_dir,
            output_file_template="$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
            video_name=clip_name,
            arg_override=(
                FFMPEG_REENCODE_ARGS if attempt_reencode else FFMPEG_COPY_ARGS
            ),
            show_progress=False,
            show_output=False,
        )
        if ret != 0:
            last_error = f"ffmpeg failed to split scene clips (exit code {ret})."
            continue
        try:
            return _collect_scene_clip_paths(
                output_dir, clip_name, len(scene_list)
            )
        except RuntimeError as exc:
            last_error = str(exc)
    raise RuntimeError(last_error)


class FileBackedVideo:
    """Minimal VIDEO duck type when `comfy_api.latest.InputImpl` is unavailable."""

    def __init__(self, path: str):
        self._path = os.fspath(path)

    @property
    def path(self) -> str:
        return self._path

    def get_stream_source(self) -> str:
        return self._path

    def save_to(self, dest_path: str, **_kwargs) -> None:
        dest = os.fspath(dest_path)
        if os.path.realpath(dest) == os.path.realpath(self._path):
            return
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copy2(self._path, dest)


def load_video_from_file(path: str):
    try:
        from comfy_api.latest import InputImpl
    except ImportError:
        return FileBackedVideo(path)
    return InputImpl.VideoFromFile(path)


def stamp_scene_duration(clip: Any, duration_sec: float) -> Any:
    """Attach detected scene length so preview can hide the copy-split tail."""
    try:
        setattr(clip, "scene_duration_sec", float(duration_sec))
    except (AttributeError, TypeError):
        pass
    return clip
