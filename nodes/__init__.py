# nodes パッケージ内で、個々のノードのマッピングを集約
from .pyscenedetect_to_images import (
    NODE_CLASS_MAPPINGS as _M1,
    NODE_DISPLAY_NAME_MAPPINGS as _D1,
)

# 必要に応じて他ノードもここで import してマージしていく
# from .another_node import NODE_CLASS_MAPPINGS as _M2, NODE_DISPLAY_NAME_MAPPINGS as _D2

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(_M1)
# NODE_CLASS_MAPPINGS.update(_M2)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(_D1)
# NODE_DISPLAY_NAME_MAPPINGS.update(_D2)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
