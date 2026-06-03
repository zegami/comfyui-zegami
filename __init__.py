"""ComfyUI-Zegami — node-pack entry point.

ComfyUI imports this package from `custom_nodes/` and reads these two mappings
to register the node.
"""

# ComfyUI loads this `__init__.py` by file location without adding the pack
# directory to sys.path, so the sibling `comfyui_zegami` / `zegami_client`
# packages aren't importable by default. Put this directory on sys.path so
# they resolve as top-level packages — matching how they're installed via
# `pip install` and imported in the test suite.
import os as _os
import sys as _sys

_PACK_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _PACK_DIR not in _sys.path:
    _sys.path.insert(0, _PACK_DIR)

from comfyui_zegami import ZegamiBatchExport  # noqa: E402  (must follow sys.path setup above)

NODE_CLASS_MAPPINGS = {
    "ZegamiBatchExport": ZegamiBatchExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZegamiBatchExport": "Zegami Batch Export",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
