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
# UI: High Resolution Input
# ====================================================================================================

class SeargeHighResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hires_mode": (UI.HIRES_MODES, {"default": UI.NONE},),
                "hires_scale": (UI.HIRES_SCALE_FACTORS, {"default": UI.HIRES_SCALE_1_5},),
                "hires_denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},),
                "hires_softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},),
                "hires_detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "hires_contrast_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "hires_saturation_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "hires_latent_detailer": (UI.LATENT_DETAILERS, {"default": UI.NONE},),
                "final_upscale_size": (UI.UPSCALE_FACTORS, {"default": UI.NONE},),
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
    def create_dict(hires_mode, hires_scale, hires_denoise, hires_softness, hires_detail_boost, hires_contrast_factor,
                    hires_saturation_factor, hires_latent_detailer, final_upscale_size):
        return {
            UI.F_HIRES_MODE: hires_mode,
            UI.F_HIRES_SCALE: hires_scale,
            UI.F_HIRES_DENOISE: round(hires_denoise, 3),
            UI.F_HIRES_SOFTNESS: round(hires_softness, 3),
            UI.F_HIRES_DETAIL_BOOST: round(hires_detail_boost, 3),
            UI.F_HIRES_CONTRAST_FACTOR: round(hires_contrast_factor, 3),
            UI.F_HIRES_SATURATION_FACTOR: round(hires_saturation_factor, 3),
            UI.F_HIRES_LATENT_DETAILER: hires_latent_detailer,
            UI.F_FINAL_UPSCALE_SIZE: final_upscale_size,
        }

    def get(self, hires_mode, hires_scale, hires_denoise, hires_softness, hires_detail_boost, hires_contrast_factor,
            hires_saturation_factor, hires_latent_detailer, final_upscale_size, data=None):
        if data is None:
            data = {}

        data[UI.S_HIGH_RESOLUTION] = self.create_dict(
            hires_mode,
            hires_scale,
            hires_denoise,
            hires_softness,
            hires_detail_boost,
            hires_contrast_factor,
            hires_saturation_factor,
            hires_latent_detailer,
            final_upscale_size,
        )

        return (data,)
