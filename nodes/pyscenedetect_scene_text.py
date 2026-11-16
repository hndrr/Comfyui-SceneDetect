from __future__ import annotations
import json

try:
    from comfy_api.latest import io
except ImportError:
    io = None

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:
    from comfy_execution.graph import ExecutionBlocker

from ..utils.scene_text import DEFAULT_SCENE_PROMPT, format_scenes_for_llm


_NODE_BASE = io.ComfyNode if io is not None else object
_DESCRIPTION = (
    "Apply a template to each scene and return the prompts as a list and joined text. "
    "Connect either SceneDetect node. No AI model is run."
)
_JSON_TOOLTIP = "Connect scenes_json from Video → Scenes or Scenes → Images (Legacy VHS)."
_TEMPLATE_TOOLTIP = (
    "One prompt per scene. Supports {index}, {scene_count}, {start_time}, "
    "{end_time}, {duration_sec}, {start_frame}, {end_frame}, {duration_frames}, "
    "and {clip_path}. Empty uses the default shot description prompt."
)


class PySceneDetectSceneText(_NODE_BASE):
    if io is None:
        FUNCTION = "format_text"
        RETURN_TYPES = ("STRING", "STRING")
        RETURN_NAMES = ("all_scene_prompts", "per_scene_prompt_list")
        OUTPUT_IS_LIST = (False, True)
        CATEGORY = "Video/PySceneDetect"
        DESCRIPTION = _DESCRIPTION

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "scenes_json": ("STRING", {"forceInput": True, "tooltip": _JSON_TOOLTIP}),
                    "prompt_template": ("STRING", {
                        "default": DEFAULT_SCENE_PROMPT, "multiline": True,
                        "tooltip": _TEMPLATE_TOOLTIP,
                    }),
                }
            }

    @classmethod
    def define_schema(cls):
        if io is None:
            raise RuntimeError("ComfyUI V3 API is required.")
        return io.Schema(
            node_id="PySceneDetectSceneText",
            display_name="PySceneDetect: Scene Text",
            category="Video/PySceneDetect",
            description=_DESCRIPTION,
            inputs=[
                io.String.Input("scenes_json", force_input=True, tooltip=_JSON_TOOLTIP),
                io.String.Input(
                    "prompt_template", default=DEFAULT_SCENE_PROMPT,
                    multiline=True, tooltip=_TEMPLATE_TOOLTIP,
                ),
            ],
            outputs=[
                io.String.Output("all_scene_prompts", tooltip="All formatted scene prompts joined with newlines, for processing together."),
                io.String.Output("per_scene_prompt_list", is_output_list=True, tooltip="One formatted prompt per scene, in scene order."),
            ],
        )

    @classmethod
    def execute(cls, scenes_json: str, prompt_template: str = ""):
        return io.NodeOutput(*cls().format_text(scenes_json, prompt_template))

    def format_text(self, scenes_json: str, prompt_template: str = ""):
        try:
            data = json.loads(scenes_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Invalid scenes_json. Connect scenes_json from a SceneDetect node.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
            raise ValueError("scenes_json must contain a 'scenes' list from a SceneDetect node.")
        try:
            text, prompts = format_scenes_for_llm(data["scenes"], prompt_template)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid scene metadata in scenes_json.") from exc
        # Empty lists crash ComfyUI consumers that process each prompt separately.
        return (text, prompts or [ExecutionBlocker(None)])


NODE_CLASS_MAPPINGS = {"PySceneDetectSceneText": PySceneDetectSceneText}
NODE_DISPLAY_NAME_MAPPINGS = {"PySceneDetectSceneText": "PySceneDetect: Scene Text"}
