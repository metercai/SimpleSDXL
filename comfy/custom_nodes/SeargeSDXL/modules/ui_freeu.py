"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 Searge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from .data_utils import retrieve_parameter
from .ui import UI


# ====================================================================================================
# UI: Loras Input
# ====================================================================================================

class SeargeFreeU:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "freeu_mode": (UI.FREEU_MODES,),
                "b1": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 1.4, "step": 0.01},),
                "b2": ("FLOAT", {"default": 1.4, "min": 1.2, "max": 1.6, "step": 0.01},),
                "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},),
                "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05},),
                "freeu_version": (UI.FREEU_VERSION,),
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM",)
    RETURN_NAMES = ("data",)
    FUNCTION = "get"

    CATEGORY = UI.CATEGORY_UI_INPUTS

    @staticmethod
    def create_dict(freeu_mode, b1, b2, s1, s2, freeu_version):
        return {
            UI.F_FREEU_MODE: freeu_mode,
            UI.F_FREEU_B1: b1,
            UI.F_FREEU_B2: b2,
            UI.F_FREEU_S1: s1,
            UI.F_FREEU_S2: s2,
            UI.F_FREEU_VERSION: freeu_version,
        }

    def get(self, freeu_mode, b1, b2, s1, s2, freeu_version, data=None):
        if data is None:
            data = {}

        data[UI.S_FREEU] = self.create_dict(
            freeu_mode,
            b1,
            b2,
            s1,
            s2,
            freeu_version,
        )

        return (data,)
