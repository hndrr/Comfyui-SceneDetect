# Top-level: ComfyUI が import する入口
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    from comfy_api.latest import ComfyExtension
except ImportError:
    ComfyExtension = None

if ComfyExtension is not None:
    class SceneDetectExtension(ComfyExtension):
        async def get_node_list(self):
            from .nodes.pyscenedetect_preview import PySceneDetectPreviewVideos
            return [PySceneDetectPreviewVideos]

    async def comfy_entrypoint():
        return SceneDetectExtension()

    __all__.append("comfy_entrypoint")
