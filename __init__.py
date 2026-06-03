"""ComfyUI-Zegami — node-pack entry point.

ComfyUI imports this package from `custom_nodes/` and reads these two mappings
to register the node.
"""

from comfyui_zegami import ZegamiBatchExport

NODE_CLASS_MAPPINGS = {
    "ZegamiBatchExport": ZegamiBatchExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZegamiBatchExport": "Zegami Batch Export",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
