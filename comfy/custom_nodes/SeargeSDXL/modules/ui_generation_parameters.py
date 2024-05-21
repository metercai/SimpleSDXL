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
# UI: Generation Parameters Input
# ====================================================================================================

class SeargeGenerationParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0},),
                "image_size_preset": (UI.RESOLUTION_PRESETS,),
                "image_width": ("INT", {"default": 1024, "min": 0, "max": UI.MAX_RESOLUTION, "step": 8},),
                "image_height": ("INT", {"default": 1024, "min": 0, "max": UI.MAX_RESOLUTION, "step": 8},),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200},),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.5, "max": 30.0, "step": 0.5},),
                "sampler_preset": (UI.SAMPLER_PRESETS,),
                "sampler_name": (UI.SAMPLERS, {"default": "dpmpp_2m"},),
                "scheduler": (UI.SCHEDULERS, {"default": "karras"},),
                "base_vs_refiner_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},),
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
    def create_dict(seed, image_size_preset, image_width, image_height, steps, cfg,
                    sampler_preset, sampler_name, scheduler, base_vs_refiner_ratio):

        # TODO: move to pre-processor
        if sampler_preset == UI.SAMPLER_PRESET_DPMPP_2M_KARRAS:
            (sampler_name, scheduler) = ("dpmpp_2m", "karras")
        elif sampler_preset == UI.SAMPLER_PRESET_EULER_A:
            (sampler_name, scheduler) = ("euler_ancestral", "normal")
        elif sampler_preset == UI.SAMPLER_PRESET_DPMPP_2M_SDE_KARRAS:
            (sampler_name, scheduler) = ("dpmpp_2m_sde", "karras")
        elif sampler_preset == UI.SAMPLER_PRESET_DPMPP_3M_SDE_EXPONENTIAL:
            (sampler_name, scheduler) = ("dpmpp_3m_sde", "exponential")
        elif sampler_preset == UI.SAMPLER_PRESET_DDIM_UNIFORM:
            (sampler_name, scheduler) = ("ddim", "ddim_uniform")

        return {
            UI.F_SEED: seed,
            UI.F_IMAGE_SIZE_PRESET: image_size_preset,
            UI.F_IMAGE_WIDTH: image_width,
            UI.F_IMAGE_HEIGHT: image_height,
            UI.F_STEPS: steps,
            UI.F_CFG: round(cfg, 3),
            UI.F_SAMPLER_PRESET: sampler_preset,
            UI.F_SAMPLER_NAME: sampler_name,
            UI.F_SCHEDULER: scheduler,
            UI.F_BASE_VS_REFINER_RATIO: round(base_vs_refiner_ratio, 3),
        }

    def get(self, seed, image_size_preset, image_width, image_height, steps, cfg,
            sampler_preset, sampler_name, scheduler, base_vs_refiner_ratio, data=None):
        if data is None:
            data = {}

        data[UI.S_GENERATION_PARAMETERS] = self.create_dict(
            seed,
            image_size_preset,
            image_width,
            image_height,
            steps,
            cfg,
            sampler_preset,
            sampler_name,
            scheduler,
            base_vs_refiner_ratio,
        )

        return (data,)
