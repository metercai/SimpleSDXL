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
# UI: Model Selector Input
# ====================================================================================================

class SeargeModelSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_checkpoint": (UI.CHECKPOINTS(),),
                "refiner_checkpoint": (UI.CHECKPOINTS_WITH_NONE(),),
                "vae_checkpoint": (UI.VAE_WITH_EMBEDDED(),),
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
    def create_dict(base_checkpoint, refiner_checkpoint, vae_checkpoint):
        return {
            UI.F_BASE_CHECKPOINT: base_checkpoint,
            UI.F_REFINER_CHECKPOINT: refiner_checkpoint,
            UI.F_VAE_CHECKPOINT: vae_checkpoint,
        }

    def get(self, base_checkpoint, refiner_checkpoint, vae_checkpoint, data=None):
        if data is None:
            data = {}

        data[UI.S_CHECKPOINTS] = self.create_dict(
            base_checkpoint,
            refiner_checkpoint,
            vae_checkpoint,
        )

        return (data,)
