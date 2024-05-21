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
# UI: Image to Image and Inpainting
# ====================================================================================================

class SeargeImage2ImageAndInpainting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},),
                "inpaint_mask_blur": ("INT", {"default": 16, "min": 0, "max": 24, "step": 4},),
                "inpaint_mask_mode": (UI.MASK_MODES,),
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
    def create_dict(denoise, inpaint_mask_blur, inpaint_mask_mode):
        return {
            UI.F_DENOISE: round(denoise, 3),
            UI.F_INPAINT_MASK_BLUR: inpaint_mask_blur,
            UI.F_INPAINT_MASK_MODE: inpaint_mask_mode,
        }

    def get(self, denoise, inpaint_mask_blur, inpaint_mask_mode, data=None):
        if data is None:
            data = {}

        data[UI.S_IMG2IMG_INPAINTING] = self.create_dict(
            denoise,
            inpaint_mask_blur,
            inpaint_mask_mode,
        )

        return (data,)
