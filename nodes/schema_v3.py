from __future__ import annotations

try:
    from comfy_api.latest import io
except ImportError:
    io = None

_NODE_BASE = io.ComfyNode if io is not None else object


def method_dynamic_combo():
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required for DynamicCombo.")

    def content_weights():
        return [
            io.Float.Input("delta_hue", default=1.0, min=0.0, max=10.0, step=0.05),
            io.Float.Input("delta_sat", default=1.0, min=0.0, max=10.0, step=0.05),
            io.Float.Input("delta_lum", default=1.0, min=0.0, max=10.0, step=0.05),
            io.Float.Input("delta_edges", default=0.0, min=0.0, max=10.0, step=0.05),
            io.Int.Input("kernel_size", default=0, min=0, step=2),
        ]

    return io.DynamicCombo.Input(
        "method",
        tooltip="Nested fields follow the selected detector.",
        options=[
            io.DynamicCombo.Option(
                "content",
                [
                    io.Float.Input(
                        "threshold", default=27.0, min=0.0, max=1000.0, step=0.1
                    ),
                    io.Boolean.Input("luma_only", default=True),
                    *content_weights(),
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
                    *content_weights(),
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


def all_detector_fields():
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required.")
    return [
        io.Combo.Input(
            "method",
            options=["content", "adaptive", "threshold", "hash", "histogram"],
            default="content",
        ),
        io.Float.Input("threshold", default=27.0, min=0.0, max=1000.0, step=0.1),
        io.Boolean.Input("luma_only", default=True),
        io.Float.Input(
            "adaptive_threshold", default=3.0, min=0.0, max=1000.0, step=0.1
        ),
        io.Int.Input("window_width", default=2, min=1, step=1),
        io.Float.Input(
            "min_content_val", default=15.0, min=0.0, max=1000.0, step=0.1
        ),
        io.Float.Input("delta_hue", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_sat", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_lum", default=1.0, min=0.0, max=10.0, step=0.05),
        io.Float.Input("delta_edges", default=0.0, min=0.0, max=10.0, step=0.05),
        io.Int.Input("kernel_size", default=0, min=0, step=2),
        io.Float.Input(
            "hash_threshold", default=0.395, min=0.0, max=1.0, step=0.001
        ),
        io.Int.Input("hash_size", default=16, min=1, step=1),
        io.Int.Input("hash_lowpass", default=2, min=1, step=1),
        io.Float.Input(
            "hist_threshold", default=0.05, min=0.0, max=1.0, step=0.001
        ),
        io.Int.Input("hist_bins", default=256, min=2, max=256, step=1),
        io.Float.Input("fade_bias", default=0.0, min=-1.0, max=1.0, step=0.05),
        io.Boolean.Input("add_final_scene", default=False),
        io.Combo.Input(
            "threshold_method", options=["floor", "ceiling"], default="floor"
        ),
    ]


def show_all_settings_combo():
    if io is None:
        raise RuntimeError("ComfyUI V3 API is required for DynamicCombo.")
    return io.DynamicCombo.Input(
        "show_all_settings",
        tooltip="false shows only the selected method. true shows every detector field.",
        options=[
            io.DynamicCombo.Option("false", [method_dynamic_combo()]),
            io.DynamicCombo.Option("true", all_detector_fields()),
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
        io.Int.Input("max_width", default=0, min=0, step=1, optional=True),
        io.Int.Input("max_height", default=0, min=0, step=1, optional=True),
        io.Int.Input("limit_scenes", default=0, min=0, step=1, optional=True),
        io.Boolean.Input("write_thumbs", default=False, optional=True),
        io.String.Input(
            "thumbs_dir",
            default="",
            optional=True,
            placeholder="Relative to ComfyUI output; default: scene_thumbs",
        ),
    ]
    if include_split:
        inputs.extend(
            [
                io.Boolean.Input("split_clips", default=False, optional=True),
                io.Boolean.Input("split_reencode", default=False, optional=True),
            ]
        )
    inputs.extend(
        [
            io.String.Input(
                "prompt_template",
                default="",
                optional=True,
                multiline=True,
                placeholder="Scene {index}/{scene_count}: {start_time}–{end_time} ({duration_sec}s). Describe this shot.",
            ),
            io.Boolean.Input("start_in_scene", default=False, optional=True),
            io.Int.Input("downscale", default=0, min=0, step=1, optional=True),
        ]
    )
    return inputs
