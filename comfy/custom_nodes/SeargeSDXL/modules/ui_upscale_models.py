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
# UI: Upscale Models Input
# ====================================================================================================

class SeargeUpscaleModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detail_processor": (UI.UPSCALERS_1x_WITH_NONE(),),
                "high_res_upscaler": (UI.UPSCALERS_4x_WITH_NONE(),),
                "primary_upscaler": (UI.UPSCALERS_4x_WITH_NONE(),),
                "secondary_upscaler": (UI.UPSCALERS_4x_WITH_NONE(),),
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
    def create_dict(detail_processor, high_res_upscaler, primary_upscaler, secondary_upscaler):
        return {
            UI.F_DETAIL_PROCESSOR: detail_processor,
            UI.F_HIGH_RES_UPSCALER: high_res_upscaler,
            UI.F_PRIMARY_UPSCALER: primary_upscaler,
            UI.F_SECONDARY_UPSCALER: secondary_upscaler,
        }

    def get(self, detail_processor, high_res_upscaler, primary_upscaler, secondary_upscaler, data=None):
        if data is None:
            data = {}

        data[UI.S_UPSCALE_MODELS] = self.create_dict(
            detail_processor,
            high_res_upscaler,
            primary_upscaler,
            secondary_upscaler,
        )

        return (data,)
