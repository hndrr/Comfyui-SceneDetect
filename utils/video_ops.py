from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, fields
from fractions import Fraction
import os
import re
import tempfile
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
from scenedetect import (
    FrameTimecode,
    SceneManager,
    is_ffmpeg_available,
    open_video,
    split_video_ffmpeg,
)
from scenedetect.detectors import (
    AdaptiveDetector,
    ContentDetector,
    HashDetector,
    HistogramDetector,
    ThresholdDetector,
)
from scenedetect.video_stream import VideoStream

DETECTION_METHODS = ["content", "adaptive", "threshold", "hash", "histogram"]
DEFAULT_SCENE_PROMPT = (
    "Scene {index}/{scene_count}: {start_time}–{end_time} ({duration_sec}s). Describe this shot."
)
FFMPEG_COPY_ARGS = "-map 0:v:0 -map 0:a? -map 0:s? -c copy"
FFMPEG_REENCODE_ARGS = (
    "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 22 -c:a aac"
)


@dataclass(frozen=True)
class DetectorSettings:
    adaptive_threshold: float = 3.0
    window_width: int = 2
    min_content_val: float = 15.0
    delta_hue: float = 1.0
    delta_sat: float = 1.0
    delta_lum: float = 1.0
    delta_edges: float = 0.0
    kernel_size: int = 0
    hash_threshold: float = 0.395
    hash_size: int = 16
    hash_lowpass: int = 2
    hist_threshold: float = 0.05
    hist_bins: int = 256
    fade_bias: float = 0.0
    add_final_scene: bool = False
    threshold_method: str = "floor"
    start_in_scene: bool = False
    downscale: int = 0

    @classmethod
    def from_mapping(cls, data: Dict[str, Any] | None) -> "DetectorSettings":
        if not data:
            return cls()
        names = {item.name for item in fields(cls)}
        return cls(**{key: data[key] for key in names if key in data})


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


def unpack_method_input(
    method: Any,
    threshold: float = 27.0,
    luma_only: bool = True,
    extra: Dict[str, Any] | None = None,
) -> Tuple[str, float, bool, Dict[str, Any]]:
    """Flatten a V3 DynamicCombo `method` dict, or pass through a plain method name."""
    extras = dict(extra or {})
    extras.pop("show_all_settings", None)
    if isinstance(method, dict):
        selected = method.get("method", "content")
        extras.update({key: value for key, value in method.items() if key != "method"})
        extras.pop("show_all_settings", None)
        method = selected
    if "threshold" in extras:
        threshold = extras.pop("threshold")
    if "luma_only" in extras:
        luma_only = extras.pop("luma_only")
    return str(method or "content"), float(threshold), bool(luma_only), extras


def unpack_toggle_combo(value: Any, name: str) -> Tuple[bool, Dict[str, Any]]:
    """Flatten a V3 DynamicCombo used as an on/off toggle with nested fields."""
    if isinstance(value, dict):
        raw = value.get(name, False)
        extras = {key: item for key, item in value.items() if key != name}
        return _is_truthy(raw), extras
    return _is_truthy(value), {}


def _is_truthy(value: Any) -> bool:
    return value is True or value == 1 or str(value).lower() == "true"


def detector_optional_input_types() -> Dict[str, Any]:
    return {
        "adaptive_threshold": (
            "FLOAT",
            {"default": 3.0, "min": 0.0, "max": 1000.0, "step": 0.1},
        ),
        "window_width": ("INT", {"default": 2, "min": 1, "step": 1}),
        "min_content_val": (
            "FLOAT",
            {"default": 15.0, "min": 0.0, "max": 1000.0, "step": 0.1},
        ),
        "delta_hue": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
        "delta_sat": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
        "delta_lum": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
        "delta_edges": (
            "FLOAT",
            {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05},
        ),
        "kernel_size": ("INT", {"default": 0, "min": 0, "step": 2}),
        "hash_threshold": (
            "FLOAT",
            {"default": 0.395, "min": 0.0, "max": 1.0, "step": 0.001},
        ),
        "hash_size": ("INT", {"default": 16, "min": 1, "step": 1}),
        "hash_lowpass": ("INT", {"default": 2, "min": 1, "step": 1}),
        "hist_threshold": (
            "FLOAT",
            {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001},
        ),
        "hist_bins": ("INT", {"default": 256, "min": 2, "max": 256, "step": 1}),
        "fade_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
        "add_final_scene": ("BOOLEAN", {"default": False}),
        "threshold_method": (["floor", "ceiling"], {"default": "floor"}),
        "start_in_scene": ("BOOLEAN", {"default": False}),
        "downscale": ("INT", {"default": 0, "min": 0, "step": 1}),
        "prompt_template": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "placeholder": DEFAULT_SCENE_PROMPT,
            },
        ),
    }


def normalized_kernel_size(value: int) -> int | None:
    size = int(value)
    if size < 3:
        return None
    if size % 2 == 0:
        size += 1
    return size


def choose_detector(
    method: str,
    threshold: float,
    min_scene_len: int | float,
    luma_only: bool,
    settings: DetectorSettings | None = None,
):
    options = settings or DetectorSettings()
    weights = ContentDetector.Components(
        delta_hue=options.delta_hue,
        delta_sat=options.delta_sat,
        delta_lum=options.delta_lum,
        delta_edges=options.delta_edges,
    )
    kernel_size = normalized_kernel_size(options.kernel_size)

    if method == "adaptive":
        return AdaptiveDetector(
            adaptive_threshold=options.adaptive_threshold,
            min_scene_len=min_scene_len,
            window_width=options.window_width,
            min_content_val=options.min_content_val,
            weights=weights,
            luma_only=luma_only,
            kernel_size=kernel_size,
        )
    if method == "threshold":
        fade_method = (
            ThresholdDetector.Method.CEILING
            if str(options.threshold_method).lower() == "ceiling"
            else ThresholdDetector.Method.FLOOR
        )
        return ThresholdDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
            fade_bias=options.fade_bias,
            add_final_scene=options.add_final_scene,
            method=fade_method,
        )
    if method == "hash":
        return HashDetector(
            threshold=options.hash_threshold,
            size=options.hash_size,
            lowpass=options.hash_lowpass,
            min_scene_len=min_scene_len,
        )
    if method == "histogram":
        return HistogramDetector(
            threshold=options.hist_threshold,
            bins=options.hist_bins,
            min_scene_len=min_scene_len,
        )
    if method != "content":
        raise ValueError(f"Unsupported detection method: {method}")
    return ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len,
        weights=weights,
        luma_only=luma_only,
        kernel_size=kernel_size,
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
    settings: DetectorSettings | None = None,
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
        settings=settings,
    )


def detect_scenes_from_video(
    video: VideoStream,
    method: str,
    threshold: float,
    min_scene_len_sec: float,
    min_scene_len_frames: int,
    luma_only: bool,
    duration: float = 0.0,
    settings: DetectorSettings | None = None,
):
    options = settings or DetectorSettings()
    fps = float(getattr(video, "frame_rate", 0.0))
    min_scene_len_seconds = max(0.0, float(min_scene_len_sec))
    min_scene_len = (
        min_scene_len_seconds
        if min_scene_len_seconds > 0
        else max(0, int(min_scene_len_frames))
    )
    manager = SceneManager()
    if options.downscale and options.downscale > 0:
        manager.auto_downscale = False
        manager.downscale = int(options.downscale)

    try:
        detector = choose_detector(
            method, threshold, min_scene_len, luma_only, settings=options
        )
        manager.add_detector(detector)
        manager.detect_scenes(
            video=video,
            duration=duration if duration > 0 else None,
            show_progress=False,
        )
        scene_list = manager.get_scene_list(start_in_scene=options.start_in_scene)
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


def load_video_from_file(path: str):
    from comfy_api.latest import InputImpl

    return InputImpl.VideoFromFile(path)


def stamp_scene_duration(clip: Any, duration_sec: float) -> Any:
    """Attach detected scene length so preview can hide the copy-split tail."""
    try:
        setattr(clip, "scene_duration_sec", float(duration_sec))
    except (AttributeError, TypeError):
        pass
    return clip


class _TemplateMap(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_scenes_for_llm(
    rows: List[Dict[str, Any]],
    template: str = "",
) -> Tuple[str, List[str]]:
    prompt_template = (template or "").strip() or DEFAULT_SCENE_PROMPT
    scene_count = len(rows)
    prompts: List[str] = []
    lines = [f"# Scenes ({scene_count})"]
    for row in rows:
        values = {
            "index": row["index"],
            "scene_count": scene_count,
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration_sec": float(row["duration_sec"]),
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "duration_frames": row["duration_frames"],
            "clip_path": row.get("clip_path") or "",
        }
        try:
            prompts.append(prompt_template.format_map(_TemplateMap(values)))
        except (ValueError, IndexError):
            prompts.append(prompt_template)
        lines.append(
            f"{row['index']}. {row['start_time']} – {row['end_time']} | "
            f"{float(row['duration_sec']):.3f}s | frames {row['start_frame']}-{row['end_frame']}"
        )
    return "\n".join(lines), prompts
