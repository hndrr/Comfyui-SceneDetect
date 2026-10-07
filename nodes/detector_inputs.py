def detector_input_types():
    return {
        "detector_settings": (
            ["default", "custom"],
            {"default": "default", "tooltip": "default uses PySceneDetect defaults. custom shows and applies the detailed detector settings."},
        ),
        "adaptive_threshold": (
            "FLOAT",
            {"default": 3.0, "min": 0.0, "step": 0.1,
             "tooltip": "custom / adaptive: change relative to surrounding frames needed for a cut. Lower values detect more cuts."},
        ),
        "window_width": (
            "INT",
            {"default": 2, "min": 1, "step": 1,
             "tooltip": "custom / adaptive: number of frames before and after each frame used for comparison."},
        ),
        "min_content_val": (
            "FLOAT",
            {"default": 15.0, "min": 0.0, "step": 0.1,
             "tooltip": "custom / adaptive: minimum absolute change needed for a cut, even if the relative change is large."},
        ),
        "delta_hue": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "step": 0.05,
             "tooltip": "custom / content or adaptive: weight of hue changes. Ignored when luma_only is true."},
        ),
        "delta_sat": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "step": 0.05,
             "tooltip": "custom / content or adaptive: weight of saturation changes. Ignored when luma_only is true."},
        ),
        "delta_lum": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "step": 0.05,
             "tooltip": "custom / content or adaptive: weight of brightness changes. Ignored when luma_only is true."},
        ),
        "delta_edges": (
            "FLOAT",
            {"default": 0.0, "min": 0.0, "step": 0.05,
             "tooltip": "custom / content or adaptive: weight of edge changes. Ignored when luma_only is true."},
        ),
        "kernel_size": (
            "INT",
            {"default": 0, "min": 0, "step": 1,
             "tooltip": "custom / content or adaptive: edge expansion size. 0–2 = automatic; larger even values round up to the next odd size."},
        ),
        "fade_bias": (
            "FLOAT",
            {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
             "tooltip": "custom / threshold: -1 places the cut at fade-out, 0 midway, +1 at fade-in."},
        ),
        "add_final_scene": (
            "BOOLEAN",
            {"default": False,
             "tooltip": "custom / threshold: add a cut at the final fade-out when the video ends without fading back in."},
        ),
        "threshold_method": (
            ["floor", "ceiling"],
            {"default": "floor",
             "tooltip": "custom / threshold: floor detects fades to black; ceiling detects fades to white."},
        ),
    }
