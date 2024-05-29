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
# Adapter for image inputs
# ====================================================================================================

class SeargeImageAdapterV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
                "source_image": ("IMAGE",),
                "image_mask": ("MASK",),
                "uploaded_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM", "SRG_DATA_STREAM",)
    RETURN_NAMES = ("data", UI.S_IMAGE_INPUTS,)
    FUNCTION = "get_value"

    CATEGORY = UI.CATEGORY_UI_PROMPTING

    @staticmethod
    def create_dict(source_image, image_mask, uploaded_mask):
        return {
            UI.F_SOURCE_IMAGE_CHANGED: True,
            UI.F_SOURCE_IMAGE: source_image,
            UI.F_IMAGE_MASK_CHANGED: True,
            UI.F_IMAGE_MASK: image_mask,
            UI.F_UPLOADED_MASK_CHANGED: True,
            UI.F_UPLOADED_MASK: uploaded_mask,
        }

    def get_value(self, source_image=None, image_mask=None, uploaded_mask=None, data=None):
        if data is None:
            data = {}

        data[UI.S_IMAGE_INPUTS] = self.create_dict(
            source_image,
            image_mask,
            uploaded_mask,
        )

        return (data, data[UI.S_IMAGE_INPUTS],)
