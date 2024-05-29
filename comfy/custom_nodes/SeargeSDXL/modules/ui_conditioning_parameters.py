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
# UI: Conditioning Parameters Input
# ====================================================================================================

class SeargeConditioningParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_conditioning_scale": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.25},),
                "refiner_conditioning_scale": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.25},),
                "target_conditioning_scale": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.25},),
                "positive_conditioning_scale": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 2.0, "step": 0.25},),
                "negative_conditioning_scale": ("FLOAT", {"default": 0.75, "min": 0.25, "max": 2.0, "step": 0.25},),
                "positive_aesthetic_score": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 10.0, "step": 0.5},),
                "negative_aesthetic_score": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.5},),
                "precondition_mode": (UI.PRECONDITION_MODES,),
                "precondition_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},),
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
    def create_dict(base_conditioning_scale, refiner_conditioning_scale, target_conditioning_scale,
                    positive_conditioning_scale, negative_conditioning_scale,
                    positive_aesthetic_score, negative_aesthetic_score,
                    precondition_mode, precondition_strength):
        return {
            UI.F_BASE_CONDITIONING_SCALE: round(base_conditioning_scale, 3),
            UI.F_REFINER_CONDITIONING_SCALE: round(refiner_conditioning_scale, 3),
            UI.F_TARGET_CONDITIONING_SCALE: round(target_conditioning_scale, 3),
            UI.F_POSITIVE_CONDITIONING_SCALE: round(positive_conditioning_scale, 3),
            UI.F_NEGATIVE_CONDITIONING_SCALE: round(negative_conditioning_scale, 3),
            UI.F_POSITIVE_AESTHETIC_SCORE: round(positive_aesthetic_score, 3),
            UI.F_NEGATIVE_AESTHETIC_SCORE: round(negative_aesthetic_score, 3),
            UI.F_PRECONDITION_MODE: precondition_mode,
            UI.F_PRECONDITION_STRENGTH: round(precondition_strength, 3),
        }

    def get(self, base_conditioning_scale, refiner_conditioning_scale, target_conditioning_scale,
            positive_conditioning_scale, negative_conditioning_scale,
            positive_aesthetic_score, negative_aesthetic_score,
            precondition_mode, precondition_strength, data=None):
        if data is None:
            data = {}

        data[UI.S_CONDITIONING_PARAMETERS] = self.create_dict(
            base_conditioning_scale,
            refiner_conditioning_scale,
            target_conditioning_scale,
            positive_conditioning_scale,
            negative_conditioning_scale,
            positive_aesthetic_score,
            negative_aesthetic_score,
            precondition_mode,
            precondition_strength,
        )

        return (data,)
