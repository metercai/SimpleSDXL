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

from .ui import UI


# ====================================================================================================
# UI: Operating Mode Input
# ====================================================================================================

class SeargeOperatingMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_mode": (UI.WORKFLOW_MODES, {"default": UI.WF_MODE_TEXT_TO_IMAGE},),
                "prompting_mode": (UI.PROMPTING_MODES, {"default": UI.PROMPTING_DEFAULT},),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1},),
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
    def create_dict(workflow_mode, prompting_mode, batch_size):
        return {
            UI.F_WORKFLOW_MODE: workflow_mode,
            UI.F_PROMPTING_MODE: prompting_mode,
            UI.F_BATCH_SIZE: batch_size,
        }

    def get(self, workflow_mode, prompting_mode, batch_size, data=None):
        if data is None:
            data = {}

        data[UI.S_OPERATING_MODE] = self.create_dict(
            workflow_mode,
            prompting_mode,
            batch_size,
        )

        return (data,)
