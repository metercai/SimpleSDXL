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

from .names import Names
from .ui import UI


# ====================================================================================================
# Inputs for SDXL Sampler
# ====================================================================================================

class SeargeSDXLSamplerV4Inputs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": ("MODEL",),
                "base_positive": ("CONDITIONING",),
                "base_negative": ("CONDITIONING",),
                "refiner_model": ("MODEL",),
                "refiner_positive": ("CONDITIONING",),
                "refiner_negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0},),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200},),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5},),
                "sampler_name": (UI.SAMPLERS, {"default": "dpmpp_2m"},),
                "scheduler": (UI.SCHEDULERS, {"default": "karras"},),
                "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "dynamic_cfg_method": ("FLOAT", {"default": UI.NONE},),
                "dynamic_base_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "dynamic_refiner_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "refiner_detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM", "SRG_DATA_STREAM",)
    RETURN_NAMES = ("data", "inputs",)
    FUNCTION = "get"

    CATEGORY = UI.CATEGORY_SAMPLING

    @staticmethod
    def create_dict(base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
                    latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise,
                    dynamic_cfg_method, dynamic_base_cfg, dynamic_refiner_cfg, refiner_detail_boost):
        return {
            Names.BASE_MODEL: base_model,
            Names.BASE_POSITIVE: base_positive,
            Names.BASE_NEGATIVE: base_negative,
            Names.REFINER_MODEL: refiner_model,
            Names.REFINER_POSITIVE: refiner_positive,
            Names.REFINER_NEGATIVE: refiner_negative,
            Names.LATENT_IMAGE: latent_image,
            Names.NOISE_SEED: noise_seed,
            Names.STEPS: steps,
            Names.CFG: cfg,
            Names.SAMPLER_NAME: sampler_name,
            Names.SCHEDULER: scheduler,
            Names.BASE_RATIO: base_ratio,
            Names.DENOISE: denoise,
            Names.DYNAMIC_CFG_METHOD: dynamic_cfg_method,
            Names.DYNAMIC_BASE_CFG: dynamic_base_cfg,
            Names.DYNAMIC_REFINER_CFG: dynamic_refiner_cfg,
            Names.REFINER_DETAIL_BOOST: refiner_detail_boost,
        }

    def get(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
            latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise,
            dynamic_cfg_method, dynamic_base_cfg, dynamic_refiner_cfg, refiner_detail_boost, data=None):
        if data is None:
            data = {}

        data["sampler_input"] = self.create_dict(
            base_model,
            base_positive,
            base_negative,
            refiner_model,
            refiner_positive,
            refiner_negative,
            latent_image,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            base_ratio,
            denoise,
            dynamic_cfg_method,
            dynamic_base_cfg,
            dynamic_refiner_cfg,
            refiner_detail_boost
        )

        return (data, data[Names.SAMPLER_INPUT],)
