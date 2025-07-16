from typing import Any as any_type
from comfy import model_management

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False
any_type = AlwaysEqualProxy("*")
class ReservedVRAMSetter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (any_type, {}),
                "reserved": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.6,
                    "step": 0.1,
                    "display": "reserved (GB)" 
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "set_vram"
    CATEGORY = "VRAM"
    def set_vram(self, anything, reserved, unique_id=None, extra_pnginfo=None):
        model_management.EXTRA_RESERVED_VRAM = int(reserved * 1024 * 1024 * 1024)
        print(f'set EXTRA_RESERVED_VRAM={reserved}GB')
        return (anything,)

NODE_CLASS_MAPPINGS = {
    "ReservedVRAMSetter": ReservedVRAMSetter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReservedVRAMSetter": "Set Reserved VRAM(GB) ⚙️"
}
