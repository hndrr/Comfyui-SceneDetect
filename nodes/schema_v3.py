from __future__ import annotations

try:
    from comfy_api.latest import io
except ImportError:
    io = None

_NODE_BASE = io.ComfyNode if io is not None else object

# First 11 widgets match master v1.3.x (positional widgets_values).
# New fields must be appended after this prefix.
LEGACY_WIDGET_NAMES = (
    "method",
    "threshold",
    "min_scene_len_sec",
    "min_scene_len_frames",
    "luma_only",
    "representative",
    "max_width",
    "max_height",
    "limit_scenes",
    "write_thumbs",
    "thumbs_dir",
)


def method_dynamic_combo():
    """Detector combo. `content` has no nested widgets so old graphs stay aligned.

    Extra detectors nest only their unique fields. `threshold` and `luma_only`
    stay at top level (same slots as v1.3.x). Content weights live after the
    original 11 widgets, not under `method`, so they do not shift saved values.
    """
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required for DynamicCombo.")

    return io.DynamicCombo.Input(
        "method",
        tooltip=(
            "content|adaptive|threshold|hash|histogram. "
            "content keeps the original widget order. Other detectors show "
            "their extra fields under this combo."
        ),
        options=[
            io.DynamicCombo.Option("content", []),
            io.DynamicCombo.Option(
                "adaptive",
                [
                    io.Float.Input(
                        "adaptive_threshold",
                        default=3.0,
                        min=0.0,
                        max=1000.0,
                        step=0.1,
                    ),
                    io.Int.Input("window_width", default=2, min=1, step=1),
                    io.Float.Input(
                        "min_content_val", default=15.0, min=0.0, max=1000.0, step=0.1
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                "threshold",
                [
                    io.Float.Input(
                        "fade_bias", default=0.0, min=-1.0, max=1.0, step=0.05
                    ),
                    io.Boolean.Input("add_final_scene", default=False),
                    io.Combo.Input(
                        "threshold_method",
                        options=["floor", "ceiling"],
                        default="floor",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                "hash",
                [
                    io.Float.Input(
                        "hash_threshold", default=0.395, min=0.0, max=1.0, step=0.001
                    ),
                    io.Int.Input("hash_size", default=16, min=1, step=1),
                    io.Int.Input("hash_lowpass", default=2, min=1, step=1),
                ],
            ),
            io.DynamicCombo.Option(
                "histogram",
                [
                    io.Float.Input(
                        "hist_threshold", default=0.05, min=0.0, max=1.0, step=0.001
                    ),
                    io.Int.Input("hist_bins", default=256, min=2, max=256, step=1),
                ],
            ),
        ],
    )


def split_clips_combo():
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required.")
    return io.DynamicCombo.Input(
        "split_clips",
        tooltip="false: no clips. true: ffmpeg scene clips in temp. Re-encode is on by default so cuts match scene boundaries.",
        options=[
            io.DynamicCombo.Option("false", []),
            io.DynamicCombo.Option(
                "true",
                [io.Boolean.Input("split_reencode", default=True)],
            ),
        ],
    )


def _legacy_scene_widgets_after_method():
    """Slots 2–11 of the original node (threshold … thumbs_dir)."""
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required.")
    return [
        io.Float.Input("threshold", default=27.0, min=0.0, max=1000.0, step=0.1),
        io.Float.Input("min_scene_len_sec", default=0.0, min=0.0, step=0.05),
        io.Int.Input("min_scene_len_frames", default=15, min=0, step=1),
        io.Boolean.Input("luma_only", default=True),
        io.Combo.Input(
            "representative", options=["start", "middle", "end"], default="start"
        ),
        io.Int.Input("max_width", default=0, min=0, step=1),
        io.Int.Input("max_height", default=0, min=0, step=1),
        io.Int.Input("limit_scenes", default=0, min=0, step=1),
        io.Boolean.Input("write_thumbs", default=False),
        io.String.Input(
            "thumbs_dir",
            default="",
            placeholder="Relative to ComfyUI output; default: scene_thumbs",
        ),
    ]


def _content_weight_widgets():
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required.")
    return [
        io.Float.Input("delta_hue", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_sat", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_lum", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_edges", default=0.0, min=0.0, max=10.0, step=0.05),
        io.Int.Input("kernel_size", default=0, min=0, step=2),
    ]


def _prompt_and_decode_widgets():
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required.")
    return [
        io.String.Input(
            "prompt_template",
            default="",
            multiline=True,
            placeholder="Scene {index}/{scene_count}: {start_time}–{end_time} ({duration_sec}s). Describe this shot.",
        ),
        io.Boolean.Input("start_in_scene", default=False),
        io.Int.Input("downscale", default=0, min=0, step=1),
    ]


def common_scene_inputs(*, include_split: bool):
    """Widget list after the `video` (or `image`/`video_info`) sockets.

    Order: original 11 widgets, then new fields. `content` DynamicCombo children
    are empty so existing workflows keep positional `widgets_values`.
    """
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required.")
    inputs = [
        method_dynamic_combo(),
        *_legacy_scene_widgets_after_method(),
        *_content_weight_widgets(),
    ]
    if include_split:
        inputs.append(split_clips_combo())
    inputs.extend(_prompt_and_decode_widgets())
    return inputs
