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
# UI: Advanced Parameters Input
# ====================================================================================================

class SeargeAdvancedParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dynamic_cfg_method": (UI.DYNAMIC_CFG_METHODS, {"default": UI.NONE},),
                "dynamic_cfg_factor": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},),
                "refiner_detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "contrast_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "saturation_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "latent_detailer": (UI.LATENT_DETAILERS, {"default": UI.NONE},),
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
    def create_dict(dynamic_cfg_method, dynamic_cfg_factor, refiner_detail_boost, contrast_factor, saturation_factor,
                    latent_detailer):
        return {
            UI.F_DYNAMIC_CFG_METHOD: dynamic_cfg_method,
            UI.F_DYNAMIC_CFG_FACTOR: round(dynamic_cfg_factor, 3),
            UI.F_REFINER_DETAIL_BOOST: round(refiner_detail_boost, 3),
            UI.F_CONTRAST_FACTOR: round(contrast_factor, 3),
            UI.F_SATURATION_FACTOR: round(saturation_factor, 3),
            UI.F_LATENT_DETAILER: latent_detailer,
        }

    def get(self, dynamic_cfg_method, dynamic_cfg_factor, refiner_detail_boost, contrast_factor, saturation_factor,
            latent_detailer, data=None):
        if data is None:
            data = {}

        data[UI.S_ADVANCED_PARAMETERS] = self.create_dict(
            dynamic_cfg_method,
            dynamic_cfg_factor,
            refiner_detail_boost,
            contrast_factor,
            saturation_factor,
            latent_detailer,
        )

        return (data,)
