from __future__ import annotations

try:
    from comfy_api.latest import io
except ImportError:
    io = None

_NODE_BASE = io.ComfyNode if io is not None else object


def method_dynamic_combo():
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required for DynamicCombo.")

    content_weights = [
        io.Float.Input("delta_hue", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_sat", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_lum", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_edges", default=0.0, min=0.0, max=10.0, step=0.05),
        io.Int.Input("kernel_size", default=0, min=0, step=2),
    ]

    return io.DynamicCombo.Input(
        "method",
        tooltip="All fields for the selected detector. Other detectors stay hidden.",
        options=[
            io.DynamicCombo.Option(
                "content",
                [
                    io.Float.Input(
                        "threshold", default=27.0, min=0.0, max=1000.0, step=0.1
                    ),
                    io.Boolean.Input("luma_only", default=True),
                    *content_weights,
                ],
            ),
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
                    io.Boolean.Input("luma_only", default=True),
                    *content_weights,
                ],
            ),
            io.DynamicCombo.Option(
                "threshold",
                [
                    io.Float.Input(
                        "threshold", default=27.0, min=0.0, max=1000.0, step=0.1
                    ),
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


def show_all_settings_combo():
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required.")
    extras = [
        io.Int.Input("max_width", default=0, min=0, step=1),
        io.Int.Input("max_height", default=0, min=0, step=1),
        io.Int.Input("limit_scenes", default=0, min=0, step=1),
        io.Boolean.Input("write_thumbs", default=False),
        io.String.Input(
            "thumbs_dir",
            default="",
            placeholder="Relative to ComfyUI output; default: scene_thumbs",
        ),
        io.String.Input(
            "prompt_template",
            default="",
            multiline=True,
            placeholder="Scene {index}/{scene_count}: {start_time}–{end_time} ({duration_sec}s). Describe this shot.",
        ),
        io.Boolean.Input("start_in_scene", default=False),
        io.Int.Input("downscale", default=0, min=0, step=1),
    ]
    return io.DynamicCombo.Input(
        "show_all_settings",
        tooltip="false: detection essentials. true: resize, scene limit, thumbs, prompt, downscale, and related node options.",
        options=[
            io.DynamicCombo.Option("false", []),
            io.DynamicCombo.Option("true", extras),
        ],
    )


def common_scene_inputs(*, include_split: bool):
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required.")
    inputs = [
        io.Float.Input("min_scene_len_sec", default=0.0, min=0.0, step=0.05),
        io.Int.Input("min_scene_len_frames", default=15, min=0, step=1),
        io.Combo.Input(
            "representative", options=["start", "middle", "end"], default="start"
        ),
    ]
    if include_split:
        inputs.append(split_clips_combo())
    inputs.append(show_all_settings_combo())
    return inputs
