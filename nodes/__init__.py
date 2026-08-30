# nodes パッケージ内で、個々のノードのマッピングを集約
from .pyscenedetect_to_images import (
    NODE_CLASS_MAPPINGS as _M1,
    NODE_DISPLAY_NAME_MAPPINGS as _D1,
)
from .pyscenedetect_video import (
    NODE_CLASS_MAPPINGS as _M2,
    NODE_DISPLAY_NAME_MAPPINGS as _D2,
)

from .pyscenedetect_preview import (
    NODE_CLASS_MAPPINGS as _M3,
    NODE_DISPLAY_NAME_MAPPINGS as _D3,
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(_M1)
NODE_CLASS_MAPPINGS.update(_M2)
NODE_CLASS_MAPPINGS.update(_M3)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(_D1)
NODE_DISPLAY_NAME_MAPPINGS.update(_D2)
NODE_DISPLAY_NAME_MAPPINGS.update(_D3)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
