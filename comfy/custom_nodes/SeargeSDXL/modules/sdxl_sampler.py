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

from .custom_sdxl_ksampler import sdxl_ksampler
from .data_utils import retrieve_parameter
from .names import Names
from .ui import UI


# ====================================================================================================
# SDXL Sampler with base and refiner support
# ====================================================================================================

class SeargeSDXLSamplerV4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
                "sampler_input": ("SRG_DATA_STREAM",),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM", "SRG_DATA_STREAM",)
    RETURN_NAMES = ("data", "sampler_output",)
    FUNCTION = "sample"

    CATEGORY = UI.CATEGORY_SAMPLING

    def sample(self, data=None, sampler_input=None):
        if data is None:
            data = {}

        if sampler_input is None:
            sampler_input = retrieve_parameter(Names.SAMPLER_INPUT, data)

        if sampler_input is None:
            return (data, None,)

        base_model = retrieve_parameter(Names.BASE_MODEL, sampler_input)
        base_positive = retrieve_parameter(Names.BASE_POSITIVE, sampler_input)
        base_negative = retrieve_parameter(Names.BASE_NEGATIVE, sampler_input)
        refiner_model = retrieve_parameter(Names.REFINER_MODEL, sampler_input)
        refiner_positive = retrieve_parameter(Names.REFINER_POSITIVE, sampler_input)
        refiner_negative = retrieve_parameter(Names.REFINER_NEGATIVE, sampler_input)
        latent_image = retrieve_parameter(Names.LATENT_IMAGE, sampler_input)
        noise_seed = retrieve_parameter(Names.NOISE_SEED, sampler_input, 4815162342)
        steps = retrieve_parameter(Names.STEPS, sampler_input, 25)
        cfg = retrieve_parameter(Names.CFG, sampler_input, 7.0)
        sampler_name = retrieve_parameter(Names.SAMPLER_NAME, sampler_input, "dpmpp_2m")
        scheduler = retrieve_parameter(Names.SCHEDULER, sampler_input, "karras")
        base_ratio = retrieve_parameter(Names.BASE_RATIO, sampler_input, 0.8)
        denoise = retrieve_parameter(Names.DENOISE, sampler_input, 1.0)
        cfg_method = retrieve_parameter(Names.DYNAMIC_CFG_METHOD, sampler_input)
        dynamic_base_cfg = retrieve_parameter(Names.DYNAMIC_BASE_CFG, sampler_input, 0.0)
        dynamic_refiner_cfg = retrieve_parameter(Names.DYNAMIC_REFINER_CFG, sampler_input, 0.0)
        refiner_detail_boost = retrieve_parameter(Names.REFINER_DETAIL_BOOST, sampler_input, 0.0)

        has_refiner_model = refiner_model is not None

        base_steps = int(steps * (base_ratio + 0.0001)) if has_refiner_model else steps
        refiner_steps = max(0, steps - base_steps)

        if cfg_method == UI.NONE:
            cfg_method = None

        if denoise < 0.005:
            return (data,)

        if refiner_steps == 0 or not has_refiner_model:
            result = sdxl_ksampler(base_model, None, noise_seed, base_steps, 0, cfg, sampler_name,
                                   scheduler, base_positive, base_negative, None, None,
                                   latent_image, denoise=denoise, disable_noise=False, start_step=0, last_step=steps,
                                   force_full_denoise=True, dynamic_base_cfg=dynamic_base_cfg, cfg_method=cfg_method)
        else:
            result = sdxl_ksampler(base_model, refiner_model, noise_seed, base_steps, refiner_steps, cfg, sampler_name,
                                   scheduler, base_positive, base_negative, refiner_positive, refiner_negative,
                                   latent_image, denoise=denoise, disable_noise=False,
                                   start_step=0, last_step=steps, force_full_denoise=True,
                                   dynamic_base_cfg=dynamic_base_cfg, dynamic_refiner_cfg=dynamic_refiner_cfg,
                                   cfg_method=cfg_method, refiner_detail_boost=refiner_detail_boost)

        data[Names.SAMPLER_OUTPUT] = result[0]

        return (data, data[Names.SAMPLER_OUTPUT],)
