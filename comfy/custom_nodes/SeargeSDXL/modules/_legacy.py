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

import datetime

import comfy.samplers
import comfy_extras.nodes_upscale_model
import comfy_extras.nodes_post_processing
import folder_paths
import json
import nodes
import numpy as np
import os

from comfy.cli_args import args
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# from ._experimental import SeargeSamplerAdvanced
from .custom_sdxl_ksampler import sdxl_ksampler


# ====================================================================================================
# SDXL Sampler with base and refiner support
# ====================================================================================================

class SeargeSDXLSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model": ("MODEL",),
            "base_positive": ("CONDITIONING",),
            "base_negative": ("CONDITIONING",),
            "refiner_model": ("MODEL",),
            "refiner_positive": ("CONDITIONING",),
            "refiner_negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
            "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("",)
    FUNCTION = "sample"

    CATEGORY = "Searge/_deprecated_/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
               latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise):

        base_steps = int(steps * base_ratio)

        if denoise < 0.01:
            return (latent_image,)

        if base_steps >= steps:
            return nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive,
                                         base_negative, latent_image, denoise=denoise, disable_noise=False,
                                         start_step=0, last_step=steps, force_full_denoise=True)

        base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive,
                                            base_negative, latent_image, denoise=denoise, disable_noise=False,
                                            start_step=0, last_step=base_steps, force_full_denoise=False)

        return nodes.common_ksampler(refiner_model, noise_seed, steps, cfg, sampler_name, scheduler, refiner_positive,
                                     refiner_negative, base_result[0], denoise=1.0, disable_noise=True,
                                     start_step=base_steps, last_step=steps, force_full_denoise=True)


# ====================================================================================================
# SDXL Sampler with base and refiner support
# ====================================================================================================

class SeargeSDXLSamplerV3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model": ("MODEL",),
            "base_positive": ("CONDITIONING",),
            "base_negative": ("CONDITIONING",),
            "refiner_model": ("MODEL",),
            "refiner_positive": ("CONDITIONING",),
            "refiner_negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            "sampler_name": ("SAMPLER_NAME", {"default": "ddim"}),
            "scheduler": ("SCHEDULER_NAME", {"default": "ddim_uniform"}),
            "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        },
            "optional": {
                "refiner_prep_steps": ("INT", {"default": 0, "min": 0, "max": 10}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Searge/_deprecated_/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
               latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise,
               refiner_prep_steps=None):

        base_steps = int(steps * (base_ratio + 0.0001))
        refiner_steps = max(0, steps - base_steps)

        if denoise < 0.01:
            return (latent_image,)

        start_at_step = 0
        input_latent = latent_image

        if refiner_prep_steps is not None:
            if refiner_prep_steps >= base_steps:
                refiner_prep_steps = base_steps - 1

            if refiner_prep_steps > 0:
                start_at_step = refiner_prep_steps
                precondition_result = nodes.common_ksampler(refiner_model, noise_seed + 2, steps, cfg, sampler_name,
                                                            scheduler, refiner_positive, refiner_negative,
                                                            latent_image, denoise=denoise, disable_noise=False,
                                                            start_step=steps - refiner_prep_steps, last_step=steps,
                                                            force_full_denoise=False)

                input_latent = precondition_result[0]

        if base_steps >= steps:
            return nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive,
                                         base_negative, input_latent, denoise=denoise, disable_noise=False,
                                         start_step=start_at_step, last_step=steps, force_full_denoise=True)

        return sdxl_ksampler(base_model, refiner_model, noise_seed, base_steps, refiner_steps, cfg, sampler_name,
                             scheduler, base_positive, base_negative, refiner_positive, refiner_negative, input_latent,
                             denoise=denoise, disable_noise=False, start_step=start_at_step, last_step=steps,
                             force_full_denoise=True)


# ====================================================================================================
# SDXL Image2Image Sampler (incl. HiRes Fix)
# ====================================================================================================

class SeargeSDXLImage2ImageSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model": ("MODEL",),
            "base_positive": ("CONDITIONING",),
            "base_negative": ("CONDITIONING",),
            "refiner_model": ("MODEL",),
            "refiner_positive": ("CONDITIONING",),
            "refiner_negative": ("CONDITIONING",),
            "image": ("IMAGE",),
            "vae": ("VAE",),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0}),
            "steps": ("INT", {"default": 20, "min": 0, "max": 200}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            "sampler_name": ("SAMPLER_NAME", {"default": "ddim"}),
            "scheduler": ("SCHEDULER_NAME", {"default": "ddim_uniform"}),
            "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
        },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "scaled_width": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "scaled_height": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
                "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"

    CATEGORY = "Searge/_deprecated_/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
               image, vae, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, softness,
               upscale_model=None, scaled_width=None, scaled_height=None, noise_offset=None, refiner_strength=None):

        base_steps = int(steps * (base_ratio + 0.0001))

        if noise_offset is None:
            noise_offset = 1

        if refiner_strength is None:
            refiner_strength = 1.0

        if refiner_strength < 0.01:
            refiner_strength = 0.01

        if steps < 1:
            return (image,)

        scaled_image = image

        use_upscale_model = upscale_model is not None and softness < 0.9999
        if use_upscale_model:
            upscale_result = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(upscale_model, image)
            scaled_image = upscale_result[0]

        if scaled_width is not None and scaled_height is not None:
            upscale_result = nodes.ImageScale().upscale(scaled_image, "bicubic", scaled_width,
                                                        scaled_height, "center")

            scaled_image = upscale_result[0]

        if use_upscale_model and softness > 0.0001:
            upscale_result = nodes.ImageScale().upscale(image, "bicubic", scaled_width, scaled_height,
                                                        "center")

            scaled_original = upscale_result[0]

            blend_result = comfy_extras.nodes_post_processing.Blend().blend_images(scaled_image, scaled_original,
                                                                                   softness, "normal")

            scaled_image = blend_result[0]

        if denoise < 0.01:
            return (scaled_image,)

        vae_encode_result = nodes.VAEEncode().encode(vae, scaled_image)
        input_latent = vae_encode_result[0]

        if base_steps >= steps:
            result_latent = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler,
                                                  base_positive, base_negative, input_latent, denoise=denoise,
                                                  disable_noise=False, start_step=0, last_step=steps,
                                                  force_full_denoise=True)
        else:
            base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler,
                                                base_positive, base_negative, input_latent, denoise=denoise,
                                                disable_noise=False, start_step=0, last_step=base_steps,
                                                force_full_denoise=True)

            result_latent = nodes.common_ksampler(refiner_model, noise_seed + noise_offset, steps, cfg, sampler_name,
                                                  scheduler, refiner_positive, refiner_negative, base_result[0],
                                                  denoise=denoise * refiner_strength, disable_noise=False,
                                                  start_step=base_steps, last_step=steps, force_full_denoise=True)

        vae_decode_result = nodes.VAEDecode().decode(vae, result_latent[0])
        output_image = vae_decode_result[0]

        return (output_image,)


# ====================================================================================================
# SDXL Sampler with base and refiner support
# ====================================================================================================

class SeargeSDXLSampler2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model": ("MODEL",),
            "base_positive": ("CONDITIONING",),
            "base_negative": ("CONDITIONING",),
            "refiner_model": ("MODEL",),
            "refiner_positive": ("CONDITIONING",),
            "refiner_negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            "sampler_name": ("SAMPLER_NAME", {"default": "ddim"}),
            "scheduler": ("SCHEDULER_NAME", {"default": "ddim_uniform"}),
            "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        },
            "optional": {
                "refiner_prep_steps": ("INT", {"default": 0, "min": 0, "max": 10}),
                "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
                "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Searge/_deprecated_/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
               latent_image, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise,
               refiner_prep_steps=None, noise_offset=None, refiner_strength=None):

        base_steps = int(steps * (base_ratio + 0.0001))

        if noise_offset is None:
            noise_offset = 1

        if refiner_strength is None:
            refiner_strength = 1.0

        if refiner_strength < 0.01:
            refiner_strength = 0.01

        if denoise < 0.01:
            return (latent_image,)

        start_at_step = 0
        input_latent = latent_image

        if refiner_prep_steps is not None:
            if refiner_prep_steps >= base_steps:
                refiner_prep_steps = base_steps - 1

            if refiner_prep_steps > 0:
                start_at_step = refiner_prep_steps
                precondition_result = nodes.common_ksampler(refiner_model, noise_seed + 2, steps, cfg, sampler_name,
                                                            scheduler, refiner_positive, refiner_negative,
                                                            latent_image, denoise=denoise, disable_noise=False,
                                                            start_step=steps - refiner_prep_steps, last_step=steps,
                                                            force_full_denoise=False)

                input_latent = precondition_result[0]

        if base_steps >= steps:
            return nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive,
                                         base_negative, input_latent, denoise=denoise, disable_noise=False,
                                         start_step=start_at_step, last_step=steps, force_full_denoise=True)

        base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler, base_positive,
                                            base_negative, input_latent, denoise=denoise, disable_noise=False,
                                            start_step=start_at_step, last_step=base_steps, force_full_denoise=True)

        return nodes.common_ksampler(refiner_model, noise_seed + noise_offset, steps, cfg, sampler_name, scheduler,
                                     refiner_positive, refiner_negative, base_result[0],
                                     denoise=denoise * refiner_strength, disable_noise=False, start_step=base_steps,
                                     last_step=steps, force_full_denoise=True)


# ====================================================================================================
# SDXL Image2Image Sampler (incl. HiRes Fix)
# ====================================================================================================

class SeargeSDXLImage2ImageSampler2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model": ("MODEL",),
            "base_positive": ("CONDITIONING",),
            "base_negative": ("CONDITIONING",),
            "refiner_model": ("MODEL",),
            "refiner_positive": ("CONDITIONING",),
            "refiner_negative": ("CONDITIONING",),
            "image": ("IMAGE",),
            "vae": ("VAE",),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0}),
            "steps": ("INT", {"default": 20, "min": 0, "max": 200}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            "sampler_name": ("SAMPLER_NAME", {"default": "ddim"}),
            "scheduler": ("SCHEDULER_NAME", {"default": "ddim_uniform"}),
            "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
        },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "scaled_width": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "scaled_height": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
                "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"

    CATEGORY = "Searge/_deprecated_/Sampling"

    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
               image, vae, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, softness,
               upscale_model=None, scaled_width=None, scaled_height=None, noise_offset=None, refiner_strength=None):

        base_steps = int(steps * (base_ratio + 0.0001))

        if noise_offset is None:
            noise_offset = 1

        if refiner_strength is None:
            refiner_strength = 1.0

        if refiner_strength < 0.01:
            refiner_strength = 0.01

        if steps < 1:
            return (image,)

        scaled_image = image

        use_upscale_model = upscale_model is not None and softness < 0.9999
        if use_upscale_model:
            upscale_result = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(upscale_model, image)
            scaled_image = upscale_result[0]

        if scaled_width is not None and scaled_height is not None:
            upscale_result = nodes.ImageScale().upscale(scaled_image, "bicubic", scaled_width,
                                                        scaled_height, "center")

            scaled_image = upscale_result[0]

        if use_upscale_model and softness > 0.0001:
            upscale_result = nodes.ImageScale().upscale(image, "bicubic", scaled_width, scaled_height,
                                                        "center")

            scaled_original = upscale_result[0]

            blend_result = comfy_extras.nodes_post_processing.Blend().blend_images(scaled_image, scaled_original,
                                                                                   softness, "normal")

            scaled_image = blend_result[0]

        if denoise < 0.01:
            return (scaled_image,)

        vae_encode_result = nodes.VAEEncode().encode(vae, scaled_image)
        input_latent = vae_encode_result[0]

        if base_steps >= steps:
            result_latent = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler,
                                                  base_positive, base_negative, input_latent, denoise=denoise,
                                                  disable_noise=False, start_step=0, last_step=steps,
                                                  force_full_denoise=True)
        else:
            base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler,
                                                base_positive, base_negative, input_latent, denoise=denoise,
                                                disable_noise=False, start_step=0, last_step=base_steps,
                                                force_full_denoise=True)

            result_latent = nodes.common_ksampler(refiner_model, noise_seed + noise_offset, steps, cfg, sampler_name,
                                                  scheduler, refiner_positive, refiner_negative, base_result[0],
                                                  denoise=denoise * refiner_strength, disable_noise=False,
                                                  start_step=base_steps, last_step=steps, force_full_denoise=True)

        vae_decode_result = nodes.VAEDecode().decode(vae, result_latent[0])
        output_image = vae_decode_result[0]

        return (output_image,)


# ====================================================================================================
# UI: Parameter Processor
# ====================================================================================================

class SeargeParameterProcessor:
    # hard - refiner uses same seed as base | soft - refiner uses different seed from base
    REFINER_INTENSITY = ["hard", "soft"]
    # same - HRF uses same seed as image generation | distinct - HRF uses different seed than image generation
    HRF_SEED_OFFSET = ["same", "distinct"]
    # simple "boolean"-like type
    STATES = ["disabled", "enabled"]
    # operating modes, determine if the latent source is empty or from a source image, inpainting also uses mask
    OPERATION_MODE = ["text to image", "image to image", "inpainting"]
    # sorted from easy-to-use to harder-to-use
    PROMPT_STYLE = ["simple", "3 prompts G+L-N", "subject focus", "style focus", "weighted", "overlay",
                    "subject - style", "style - subject", "style only", "weighted - overlay", "overlay - weighted"]
    # (work in progress)
    STYLE_TEMPLATE = ["none", "from preprocessor", "test"]
    # save folder for generated images
    SAVE_TO = ["output folder", "input folder"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "inputs": ("PARAMETER_INPUTS",),
        },
        }

    RETURN_TYPES = ("PARAMETERS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"

    CATEGORY = "Searge/_deprecated_/UI"

    def process(self, inputs):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        if parameters["denoise"] is None:
            parameters["denoise"] = 1.0

        saturation = parameters["refiner_intensity"]
        if saturation is not None:
            # "soft"
            if saturation == SeargeParameterProcessor.REFINER_INTENSITY[1]:
                parameters["noise_offset"] = 1
            # incl. SeargeParameterProcessor.REFINER_INTENSITY[1] -> "hard"
            else:
                parameters["noise_offset"] = 0

        hires_fix = parameters["hires_fix"]
        # "disabled"
        if hires_fix is not None and hires_fix == SeargeParameterProcessor.STATES[0]:
            parameters["hrf_steps"] = 0

        hrf_saturation = parameters["hrf_intensity"]
        if hrf_saturation is not None:
            # "soft"
            if hrf_saturation == SeargeParameterProcessor.REFINER_INTENSITY[1]:
                parameters["hrf_noise_offset"] = 1
            # incl. SeargeParameterProcessor.REFINER_INTENSITY[0] -> "hard"
            else:
                parameters["hrf_noise_offset"] = 0

        seed_offset = parameters["hrf_seed_offset"]
        if seed_offset is not None:
            seed = parameters["seed"] if parameters["seed"] is not None else 0
            # "distinct"
            if seed_offset == SeargeParameterProcessor.HRF_SEED_OFFSET[1]:
                parameters["hrf_seed"] = seed + 3
            # incl. SeargeParameterProcessor.HRF_SEED_OFFSET[0] -> "same"
            else:
                parameters["hrf_seed"] = seed

        style_template = parameters["style_template"]
        if style_template is not None:
            # "from preprocessor"
            if style_template == SeargeParameterProcessor.STYLE_TEMPLATE[1]:
                # this does nothing here, but will be used in the preprocessor
                pass
            # "test"
            if style_template == SeargeParameterProcessor.STYLE_TEMPLATE[2]:
                if parameters["noise_offset"] is not None:
                    parameters["noise_offset"] = 1 - parameters["hrf_noise_offset"]
                if parameters["hrf_noise_offset"] is not None:
                    parameters["hrf_noise_offset"] = 1 - parameters["hrf_noise_offset"]
            # incl. SeargeParameterProcessor.STYLE_TEMPLATE[0] -> "none"
            else:
                # TODO: apply style based on its name here...
                pass

        operation_mode = parameters["operation_mode"]
        if operation_mode is not None:
            # "image to image":
            if operation_mode == SeargeParameterProcessor.OPERATION_MODE[1]:
                parameters["operation_selector"] = 1
            # "inpainting":
            elif operation_mode == SeargeParameterProcessor.OPERATION_MODE[2]:
                parameters["operation_selector"] = 2
            # incl. SeargeParameterProcessor.OPERATION_MODE[0] -> "text to image":
            else:
                parameters["operation_selector"] = 0
                # always fully denoise in img2img mode
                parameters["denoise"] = 1.0

        prompt_style = parameters["prompt_style"]
        if prompt_style is not None:
            # "simple"
            if prompt_style == SeargeParameterProcessor.PROMPT_STYLE[0]:
                parameters["prompt_style_selector"] = 0
                parameters["prompt_style_group"] = 0
                main_prompt = parameters["main_prompt"]
                parameters["secondary_prompt"] = main_prompt
                parameters["style_prompt"] = ""
                parameters["negative_style"] = ""
            # "subject focus"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[2]:
                parameters["prompt_style_selector"] = 1
                parameters["prompt_style_group"] = 0
            # "style focus"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[3]:
                parameters["prompt_style_selector"] = 2
                parameters["prompt_style_group"] = 0
            # "weighted"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[4]:
                parameters["prompt_style_selector"] = 3
                parameters["prompt_style_group"] = 0
            # "overlay"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[5]:
                parameters["prompt_style_selector"] = 4
                parameters["prompt_style_group"] = 0
            # "subject - style"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[6]:
                parameters["prompt_style_selector"] = 0
                parameters["prompt_style_group"] = 1
            # "style - subject"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[7]:
                parameters["prompt_style_selector"] = 1
                parameters["prompt_style_group"] = 1
            # "style only"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[8]:
                parameters["prompt_style_selector"] = 2
                parameters["prompt_style_group"] = 1
            # "weighted - overlay"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[9]:
                parameters["prompt_style_selector"] = 3
                parameters["prompt_style_group"] = 1
            # "overlay - weighted"
            elif prompt_style == SeargeParameterProcessor.PROMPT_STYLE[10]:
                parameters["prompt_style_selector"] = 4
                parameters["prompt_style_group"] = 1
            # incl. SeargeParameterProcessor.PROMPT_STYLE[1] -> "3 prompts G+L-N"
            else:
                parameters["prompt_style_selector"] = 0
                parameters["prompt_style_group"] = 0
                parameters["style_prompt"] = ""
                parameters["negative_style"] = ""

        # TODO: replace this special logic and the dirty hacks by creating new generated parameters for saving
        save_image = parameters["save_image"]
        if save_image is not None:
            # "disabled"
            if save_image == SeargeParameterProcessor.STATES[0]:
                # when image saving is disabled, we also don't want to save the upscaled image, even if that's enabled
                parameters["save_upscaled_image"] = SeargeParameterProcessor.STATES[0]
                # HACK: this is a bit dirty, but the variable hires_fix determines if the image should be saved
                #       but when image saving is disabled, we don't want that to happen
                parameters["hires_fix"] = SeargeParameterProcessor.STATES[0]
            # "enabled"
            else:
                # in case we are saving to the input folder, we need to enable saving after the hires fix, even
                # if that's disabled in the settings
                if parameters["save_directory"] == SeargeParameterProcessor.SAVE_TO[1]:
                    parameters["hires_fix"] = SeargeParameterProcessor.STATES[1]

        return (parameters,)


# ====================================================================================================
# UI: Style Processor
# ====================================================================================================

class SeargeStylePreprocessor:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "inputs": ("PARAMETER_INPUTS",),
            "active_style_name": ("STRING", {"multiline": False, "default": ""}),
            "style_definitions": ("STRING", {"multiline": True, "default": "[unfinished work in progress]"}),
        },
        }

    RETURN_TYPES = ("PARAMETER_INPUTS",)
    RETURN_NAMES = ("inputs",)
    FUNCTION = "process"

    CATEGORY = "Searge/_deprecated_/UI"

    def process(self, inputs, active_style_name, style_definitions):
        if inputs is None:
            inputs = {}

        style_template = inputs["style_template"]
        # not "from preprocessor"
        if style_template is None or style_template != SeargeParameterProcessor.STYLE_TEMPLATE[1]:
            return (inputs,)

        # TODO: do what needs to be done to apply the selected style

        return (inputs,)


# ====================================================================================================
# UI: Prompt Inputs
# ====================================================================================================

class SeargeInput1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "main_prompt": ("STRING", {"multiline": True, "default": ""}),
            "secondary_prompt": ("STRING", {"multiline": True, "default": ""}),
            "style_prompt": ("STRING", {"multiline": True, "default": ""}),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "negative_style": ("STRING", {"multiline": True, "default": ""}),
        },
            "optional": {
                "inputs": ("PARAMETER_INPUTS",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("PARAMETER_INPUTS",)
    RETURN_NAMES = ("inputs",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/UI/Inputs"

    def mux(self, main_prompt, secondary_prompt, style_prompt, negative_prompt, negative_style, inputs=None,
            image=None, mask=None):

        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["main_prompt"] = main_prompt
        parameters["secondary_prompt"] = secondary_prompt
        parameters["style_prompt"] = style_prompt
        parameters["negative_prompt"] = negative_prompt
        parameters["negative_style"] = negative_style
        parameters["image"] = image
        parameters["mask"] = mask

        return (parameters,)


# ====================================================================================================
# UI: Generation Parameters Input
# ====================================================================================================

class SeargeInput2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "image_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "image_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "steps": ("INT", {"default": 20, "min": 0, "max": 200}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "ddim"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "ddim_uniform"}),
            "save_image": (SeargeParameterProcessor.STATES, {"default": SeargeParameterProcessor.STATES[1]}),
            "save_directory": (SeargeParameterProcessor.SAVE_TO,
                               {"default": SeargeParameterProcessor.SAVE_TO[0]}),
        },
            "optional": {
                "inputs": ("PARAMETER_INPUTS",),
            },
        }

    RETURN_TYPES = ("PARAMETER_INPUTS",)
    RETURN_NAMES = ("inputs",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/UI/Inputs"

    def mux(self, seed, image_width, image_height, steps, cfg, sampler_name, scheduler, save_image, save_directory,
            inputs=None):

        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["seed"] = seed
        parameters["image_width"] = image_width
        parameters["image_height"] = image_height
        parameters["steps"] = steps
        parameters["cfg"] = cfg
        parameters["sampler_name"] = sampler_name
        parameters["scheduler"] = scheduler
        parameters["save_image"] = save_image
        parameters["save_directory"] = save_directory

        return (parameters,)


# ====================================================================================================
# UI: Advanced Parameters Input
# ====================================================================================================

class SeargeInput3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
            "refiner_intensity": (SeargeParameterProcessor.REFINER_INTENSITY,
                                  {"default": SeargeParameterProcessor.REFINER_INTENSITY[1]}),
            "precondition_steps": ("INT", {"default": 0, "min": 0, "max": 10}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
            "upscale_resolution_factor": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 4.0, "step": 0.25}),
            "save_upscaled_image": (SeargeParameterProcessor.STATES, {"default": SeargeParameterProcessor.STATES[1]}),
        },
            "optional": {
                "inputs": ("PARAMETER_INPUTS",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("PARAMETER_INPUTS",)
    RETURN_NAMES = ("inputs",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/UI/Inputs"

    def mux(self, base_ratio, refiner_strength, refiner_intensity, precondition_steps, batch_size,
            upscale_resolution_factor, save_upscaled_image, inputs=None, denoise=None):

        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["denoise"] = denoise
        parameters["base_ratio"] = base_ratio
        parameters["refiner_strength"] = refiner_strength
        parameters["refiner_intensity"] = refiner_intensity
        parameters["precondition_steps"] = precondition_steps
        parameters["batch_size"] = batch_size
        parameters["upscale_resolution_factor"] = upscale_resolution_factor
        parameters["save_upscaled_image"] = save_upscaled_image

        return (parameters,)


# ====================================================================================================
# UI: Model Selector Input
# ====================================================================================================

class SeargeInput4:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model": (folder_paths.get_filename_list("checkpoints"),),
            "refiner_model": (folder_paths.get_filename_list("checkpoints"),),
            "vae_model": (folder_paths.get_filename_list("vae"),),
            "main_upscale_model": (folder_paths.get_filename_list("upscale_models"),),
            "support_upscale_model": (folder_paths.get_filename_list("upscale_models"),),
            "lora_model": (folder_paths.get_filename_list("loras"),),
        },
            "optional": {
                "model_settings": ("MODEL_SETTINGS",),
            },
        }

    RETURN_TYPES = ("MODEL_NAMES",)
    RETURN_NAMES = ("model_names",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/UI/Inputs"

    def mux(self, base_model, refiner_model, vae_model, main_upscale_model, support_upscale_model, lora_model,
            model_settings=None):

        if model_settings is None:
            model_names = {}
        else:
            model_names = model_settings

        model_names["base_model"] = base_model
        model_names["refiner_model"] = refiner_model
        model_names["vae_model"] = vae_model
        model_names["main_upscale_model"] = main_upscale_model
        model_names["support_upscale_model"] = support_upscale_model
        model_names["lora_model"] = lora_model

        return (model_names,)


# ====================================================================================================
# UI: Prompt Processing Input
# ====================================================================================================

class SeargeInput5:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_conditioning_scale": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 4.0, "step": 0.25}),
            "refiner_conditioning_scale": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 4.0, "step": 0.25}),
            "style_prompt_power": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
            "negative_style_power": ("FLOAT", {"default": 0.67, "min": 0.0, "max": 1.0, "step": 0.01}),
            "style_template": (SeargeParameterProcessor.STYLE_TEMPLATE,
                               {"default": SeargeParameterProcessor.STYLE_TEMPLATE[0]}),
        },
            "optional": {
                "inputs": ("PARAMETER_INPUTS",),
            },
        }

    RETURN_TYPES = ("PARAMETER_INPUTS",)
    RETURN_NAMES = ("inputs",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/UI/Inputs"

    def mux(self, base_conditioning_scale, refiner_conditioning_scale, style_prompt_power, negative_style_power,
            style_template, inputs=None):

        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["base_conditioning_scale"] = base_conditioning_scale
        parameters["refiner_conditioning_scale"] = refiner_conditioning_scale
        parameters["style_prompt_power"] = style_prompt_power
        parameters["negative_style_power"] = negative_style_power
        parameters["style_template"] = style_template

        return (parameters,)


# ====================================================================================================
# UI: HiResFix Parameters Input
# ====================================================================================================

class SeargeInput6:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "hires_fix": (SeargeParameterProcessor.STATES, {"default": SeargeParameterProcessor.STATES[1]}),
            "hrf_steps": ("INT", {"default": 0, "min": 0, "max": 100}),
            "hrf_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "hrf_upscale_factor": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 4.0, "step": 0.25}),
            "hrf_intensity": (SeargeParameterProcessor.REFINER_INTENSITY,
                              {"default": SeargeParameterProcessor.REFINER_INTENSITY[1]}),
            "hrf_seed_offset": (SeargeParameterProcessor.HRF_SEED_OFFSET,
                                {"default": SeargeParameterProcessor.HRF_SEED_OFFSET[1]}),
            "hrf_smoothness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        },
            "optional": {
                "inputs": ("PARAMETER_INPUTS",),
            },
        }

    RETURN_TYPES = ("PARAMETER_INPUTS",)
    RETURN_NAMES = ("inputs",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/UI/Inputs"

    def mux(self, hires_fix, hrf_steps, hrf_denoise, hrf_upscale_factor, hrf_intensity, hrf_seed_offset,
            hrf_smoothness, inputs=None):

        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["hires_fix"] = hires_fix
        parameters["hrf_steps"] = hrf_steps
        parameters["hrf_denoise"] = hrf_denoise
        parameters["hrf_upscale_factor"] = hrf_upscale_factor
        parameters["hrf_intensity"] = hrf_intensity
        parameters["hrf_seed_offset"] = hrf_seed_offset
        parameters["hrf_smoothness"] = hrf_smoothness

        return (parameters,)


# ====================================================================================================
# UI: Misc Inputs
# ====================================================================================================

class SeargeInput7:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "lora_strength": ("FLOAT", {"default": 0.2, "min": -10.0, "max": 10.0, "step": 0.05}),
            "operation_mode": (SeargeParameterProcessor.OPERATION_MODE,
                               {"default": SeargeParameterProcessor.OPERATION_MODE[0]}),
            "prompt_style": (SeargeParameterProcessor.PROMPT_STYLE,
                             {"default": SeargeParameterProcessor.PROMPT_STYLE[0]}),
        },
            "optional": {
                "inputs": ("PARAMETER_INPUTS",),
            },
        }

    RETURN_TYPES = ("PARAMETER_INPUTS",)
    RETURN_NAMES = ("inputs",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/UI/Inputs"

    def mux(self, lora_strength, operation_mode, prompt_style, inputs=None):
        if inputs is None:
            parameters = {}
        else:
            parameters = inputs

        parameters["lora_strength"] = lora_strength
        parameters["operation_mode"] = operation_mode
        parameters["prompt_style"] = prompt_style

        return (parameters,)


# ====================================================================================================
# UI: Prompt Outputs
# ====================================================================================================

class SeargeOutput1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "parameters": ("PARAMETERS",),
        },
        }

    RETURN_TYPES = ("PARAMETERS", "STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE", "MASK",)
    RETURN_NAMES = ("parameters", "main_prompt", "secondary_prompt", "style_prompt", "negative_prompt",
                    "negative_style", "image", "mask",)
    FUNCTION = "demux"

    CATEGORY = "Searge/_deprecated_/UI/Outputs"

    def demux(self, parameters):
        main_prompt = parameters["main_prompt"]
        secondary_prompt = parameters["secondary_prompt"]
        style_prompt = parameters["style_prompt"]
        negative_prompt = parameters["negative_prompt"]
        negative_style = parameters["negative_style"]
        image = parameters["image"]
        mask = parameters["mask"]

        return (parameters, main_prompt, secondary_prompt, style_prompt, negative_prompt, negative_style, image, mask,)


# ====================================================================================================
# UI: Generation Parameters Output
# ====================================================================================================

class SeargeOutput2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "parameters": ("PARAMETERS",),
        },
        }

    RETURN_TYPES = ("PARAMETERS", "INT", "INT", "INT", "INT", "FLOAT", "SAMPLER_NAME", "SCHEDULER_NAME",
                    "ENABLE_STATE", "SAVE_FOLDER",)
    RETURN_NAMES = ("parameters", "seed", "image_width", "image_height", "steps", "cfg", "sampler_name", "scheduler",
                    "save_image", "save_directory",)
    FUNCTION = "demux"

    CATEGORY = "Searge/_deprecated_/UI/Outputs"

    def demux(self, parameters):
        seed = parameters["seed"]
        image_width = parameters["image_width"]
        image_height = parameters["image_height"]
        steps = parameters["steps"]
        cfg = parameters["cfg"]
        sampler_name = parameters["sampler_name"]
        scheduler = parameters["scheduler"]
        save_image = parameters["save_image"]
        save_directory = parameters["save_directory"]

        return (parameters, seed, image_width, image_height, steps, cfg, sampler_name, scheduler, save_image,
                save_directory,)


# ====================================================================================================
# UI: Advanced Parameters Output
# ====================================================================================================

class SeargeOutput3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "parameters": ("PARAMETERS",),
        },
        }

    RETURN_TYPES = ("PARAMETERS", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "FLOAT", "ENABLE_STATE",)
    RETURN_NAMES = ("parameters", "denoise", "base_ratio", "refiner_strength", "noise_offset", "precondition_steps",
                    "batch_size", "upscale_resolution_factor", "save_upscaled_image",)
    FUNCTION = "demux"

    CATEGORY = "Searge/_deprecated_/UI/Outputs"

    def demux(self, parameters):
        denoise = parameters["denoise"]
        base_ratio = parameters["base_ratio"]
        refiner_strength = parameters["refiner_strength"]
        noise_offset = parameters["noise_offset"]
        precondition_steps = parameters["precondition_steps"]
        batch_size = parameters["batch_size"]
        upscale_resolution_factor = parameters["upscale_resolution_factor"]
        save_upscaled_image = parameters["save_upscaled_image"]

        return (parameters, denoise, base_ratio, refiner_strength, noise_offset, precondition_steps, batch_size,
                upscale_resolution_factor, save_upscaled_image,)


# ====================================================================================================
# UI: Model Selector
# ====================================================================================================

class SeargeOutput4:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_names": ("MODEL_NAMES",),
        },
        }

    RETURN_TYPES = ("MODEL_NAMES", "CHECKPOINT_NAME", "CHECKPOINT_NAME", "VAE_NAME", "UPSCALER_NAME", "UPSCALER_NAME",
                    "LORA_NAME",)
    RETURN_NAMES = ("model_names", "base_model", "refiner_model", "vae_model", "main_upscale_model",
                    "support_upscale_model", "lora_model",)
    FUNCTION = "demux"

    CATEGORY = "Searge/_deprecated_/UI/Outputs"

    def demux(self, model_names):
        base_model = model_names["base_model"]
        refiner_model = model_names["refiner_model"]
        vae_model = model_names["vae_model"]
        main_upscale_model = model_names["main_upscale_model"]
        support_upscale_model = model_names["support_upscale_model"]
        lora_model = model_names["lora_model"]

        return (model_names, base_model, refiner_model, vae_model, main_upscale_model, support_upscale_model,
                lora_model,)


# ====================================================================================================
# UI: Prompt Processing Output
# ====================================================================================================

class SeargeOutput5:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "parameters": ("PARAMETERS",),
        },
        }

    RETURN_TYPES = ("PARAMETERS", "FLOAT", "FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("parameters", "base_conditioning_scale", "refiner_conditioning_scale", "style_prompt_power",
                    "negative_style_power",)
    FUNCTION = "demux"

    CATEGORY = "Searge/_deprecated_/UI/Outputs"

    def demux(self, parameters):
        base_conditioning_scale = parameters["base_conditioning_scale"]
        refiner_conditioning_scale = parameters["refiner_conditioning_scale"]
        style_prompt_power = parameters["style_prompt_power"]
        negative_style_power = parameters["negative_style_power"]

        return (parameters, base_conditioning_scale, refiner_conditioning_scale, style_prompt_power,
                negative_style_power,)


# ====================================================================================================
# UI: HiResFix Parameters Output
# ====================================================================================================

class SeargeOutput6:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "parameters": ("PARAMETERS",),
        },
        }

    RETURN_TYPES = ("PARAMETERS", "INT", "FLOAT", "FLOAT", "INT", "INT", "ENABLE_STATE", "FLOAT",)
    RETURN_NAMES = ("parameters", "hrf_steps", "hrf_denoise", "hrf_upscale_factor", "hrf_noise_offset",
                    "hrf_seed", "hires_fix", "hrf_smoothness",)
    FUNCTION = "demux"

    CATEGORY = "Searge/_deprecated_/UI/Outputs"

    def demux(self, parameters):
        hrf_steps = parameters["hrf_steps"]
        hrf_denoise = parameters["hrf_denoise"]
        hrf_upscale_factor = parameters["hrf_upscale_factor"]
        hrf_noise_offset = parameters["hrf_noise_offset"]
        hrf_seed = parameters["hrf_seed"]
        hires_fix = parameters["hires_fix"]
        hrf_smoothness = parameters["hrf_smoothness"]

        return (parameters, hrf_steps, hrf_denoise, hrf_upscale_factor, hrf_noise_offset, hrf_seed, hires_fix,
                hrf_smoothness,)


# ====================================================================================================
# UI: Misc Outputs
# ====================================================================================================

class SeargeOutput7:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "parameters": ("PARAMETERS",),
        },
        }

    RETURN_TYPES = ("PARAMETERS", "FLOAT",)
    RETURN_NAMES = ("parameters", "lora_strength",)
    FUNCTION = "demux"

    CATEGORY = "Searge/_deprecated_/UI/Outputs"

    def demux(self, parameters):
        lora_strength = parameters["lora_strength"]

        return (parameters, lora_strength,)


# ====================================================================================================
# UI: Generated outputs for flow control
# ====================================================================================================

class SeargeGenerated1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "parameters": ("PARAMETERS",),
        },
        }

    RETURN_TYPES = ("PARAMETERS", "INT", "INT", "INT",)
    RETURN_NAMES = ("parameters", "operation_selector", "prompt_style_selector", "prompt_style_group",)
    FUNCTION = "demux"

    CATEGORY = "Searge/_deprecated_/UI/Generated"

    def demux(self, parameters):
        operation_selector = parameters["operation_selector"]
        prompt_style_selector = parameters["prompt_style_selector"]
        prompt_style_group = parameters["prompt_style_group"]

        return (parameters, operation_selector, prompt_style_selector, prompt_style_group,)


# ====================================================================================================
# SDXL CLIP Text Encoder for prompts with base and refiner support
# ====================================================================================================

class SeargeSDXLPromptEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_clip": ("CLIP",),
            "refiner_clip": ("CLIP",),
            "pos_g": ("STRING", {"multiline": True, "default": "POS_G"}),
            "pos_l": ("STRING", {"multiline": True, "default": "POS_L"}),
            "pos_r": ("STRING", {"multiline": True, "default": "POS_R"}),
            "neg_g": ("STRING", {"multiline": True, "default": "NEG_G"}),
            "neg_l": ("STRING", {"multiline": True, "default": "NEG_L"}),
            "neg_r": ("STRING", {"multiline": True, "default": "NEG_R"}),
            "base_width": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "base_height": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "target_width": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "target_height": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "pos_ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "neg_ascore": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "refiner_width": ("INT", {"default": 2048, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "refiner_height": ("INT", {"default": 2048, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
        },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("base_positive", "base_negative", "refiner_positive", "refiner_negative",)
    FUNCTION = "encode"

    CATEGORY = "Searge/_deprecated_/ClipEncoding"

    def encode(self, base_clip, refiner_clip, pos_g, pos_l, pos_r, neg_g, neg_l, neg_r, base_width, base_height,
               crop_w, crop_h, target_width, target_height, pos_ascore, neg_ascore, refiner_width, refiner_height):

        empty = base_clip.tokenize("")

        # positive base prompt
        tokens1 = base_clip.tokenize(pos_g)
        tokens1["l"] = base_clip.tokenize(pos_l)["l"]

        if len(tokens1["l"]) != len(tokens1["g"]):
            while len(tokens1["l"]) < len(tokens1["g"]):
                tokens1["l"] += empty["l"]
            while len(tokens1["l"]) > len(tokens1["g"]):
                tokens1["g"] += empty["g"]

        cond1, pooled1 = base_clip.encode_from_tokens(tokens1, return_pooled=True)
        res1 = [[cond1, {"pooled_output": pooled1, "width": base_width, "height": base_height, "crop_w": crop_w,
                         "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]

        # negative base prompt
        tokens2 = base_clip.tokenize(neg_g)
        tokens2["l"] = base_clip.tokenize(neg_l)["l"]

        if len(tokens2["l"]) != len(tokens2["g"]):
            while len(tokens2["l"]) < len(tokens2["g"]):
                tokens2["l"] += empty["l"]
            while len(tokens2["l"]) > len(tokens2["g"]):
                tokens2["g"] += empty["g"]

        cond2, pooled2 = base_clip.encode_from_tokens(tokens2, return_pooled=True)
        res2 = [[cond2, {"pooled_output": pooled2, "width": base_width, "height": base_height, "crop_w": crop_w,
                         "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]

        # positive refiner prompt
        tokens3 = refiner_clip.tokenize(pos_r)
        cond3, pooled3 = refiner_clip.encode_from_tokens(tokens3, return_pooled=True)
        res3 = [[cond3, {"pooled_output": pooled3, "aesthetic_score": pos_ascore, "width": refiner_width,
                         "height": refiner_height}]]

        # negative refiner prompt
        tokens4 = refiner_clip.tokenize(neg_r)
        cond4, pooled4 = refiner_clip.encode_from_tokens(tokens4, return_pooled=True)
        res4 = [[cond4, {"pooled_output": pooled4, "aesthetic_score": neg_ascore, "width": refiner_width,
                         "height": refiner_height}]]

        return (res1, res2, res3, res4,)


# ====================================================================================================
# SDXL CLIP Text Encoder for base prompts
# ====================================================================================================

class SeargeSDXLBasePromptEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_clip": ("CLIP",),
            "pos_g": ("STRING", {"multiline": True, "default": "POS_G"}),
            "pos_l": ("STRING", {"multiline": True, "default": "POS_L"}),
            "neg_g": ("STRING", {"multiline": True, "default": "NEG_G"}),
            "neg_l": ("STRING", {"multiline": True, "default": "NEG_L"}),
            "base_width": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "base_height": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "target_width": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "target_height": ("INT", {"default": 4096, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
        },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("base_positive", "base_negative",)
    FUNCTION = "encode"

    CATEGORY = "Searge/_deprecated_/ClipEncoding"

    def encode(self, base_clip, pos_g, pos_l, neg_g, neg_l, base_width, base_height, crop_w, crop_h, target_width,
               target_height):

        empty = base_clip.tokenize("")

        # positive base prompt
        tokens1 = base_clip.tokenize(pos_g)
        tokens1["l"] = base_clip.tokenize(pos_l)["l"]

        if len(tokens1["l"]) != len(tokens1["g"]):
            while len(tokens1["l"]) < len(tokens1["g"]):
                tokens1["l"] += empty["l"]
            while len(tokens1["l"]) > len(tokens1["g"]):
                tokens1["g"] += empty["g"]

        cond1, pooled1 = base_clip.encode_from_tokens(tokens1, return_pooled=True)
        res1 = [[cond1, {"pooled_output": pooled1, "width": base_width, "height": base_height, "crop_w": crop_w,
                         "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]

        # negative base prompt
        tokens2 = base_clip.tokenize(neg_g)
        tokens2["l"] = base_clip.tokenize(neg_l)["l"]

        if len(tokens2["l"]) != len(tokens2["g"]):
            while len(tokens2["l"]) < len(tokens2["g"]):
                tokens2["l"] += empty["l"]
            while len(tokens2["l"]) > len(tokens2["g"]):
                tokens2["g"] += empty["g"]

        cond2, pooled2 = base_clip.encode_from_tokens(tokens2, return_pooled=True)
        res2 = [[cond2, {"pooled_output": pooled2, "width": base_width, "height": base_height, "crop_w": crop_w,
                         "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]

        return (res1, res2,)


# ====================================================================================================
# SDXL CLIP Text Encoder for prompts with base and refiner support
# ====================================================================================================

class SeargeSDXLRefinerPromptEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "refiner_clip": ("CLIP",),
            "pos_r": ("STRING", {"multiline": True, "default": "POS_R"}),
            "neg_r": ("STRING", {"multiline": True, "default": "NEG_R"}),
            "pos_ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "neg_ascore": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "refiner_width": ("INT", {"default": 2048, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "refiner_height": ("INT", {"default": 2048, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
        },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("refiner_positive", "refiner_negative",)
    FUNCTION = "encode"

    CATEGORY = "Searge/_deprecated_/ClipEncoding"

    def encode(self, refiner_clip, pos_r, neg_r, pos_ascore, neg_ascore, refiner_width, refiner_height):
        # positive refiner prompt
        tokens1 = refiner_clip.tokenize(pos_r)
        cond1, pooled1 = refiner_clip.encode_from_tokens(tokens1, return_pooled=True)
        res1 = [[cond1, {"pooled_output": pooled1, "aesthetic_score": pos_ascore, "width": refiner_width,
                         "height": refiner_height}]]

        # negative refiner prompt
        tokens2 = refiner_clip.tokenize(neg_r)
        cond2, pooled2 = refiner_clip.encode_from_tokens(tokens2, return_pooled=True)
        res2 = [[cond2, {"pooled_output": pooled2, "aesthetic_score": neg_ascore, "width": refiner_width,
                         "height": refiner_height}]]

        return (res1, res2,)


# ====================================================================================================
# Tool: text input node for prompt text
# ====================================================================================================

class SeargePromptText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True}),
        },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Prompting"

    def get_value(self, prompt):
        return (prompt,)


# Tool: text input node for prompt text

class SeargePromptCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt1": ("STRING", {"default": "", "multiline": True}),
            "separator": ("STRING", {"default": ", ", "multiline": False}),
            "prompt2": ("STRING", {"default": "", "multiline": True}),
        },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined prompt",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Prompting"

    def get_value(self, prompt1, separator, prompt2):
        len1 = len(prompt1)
        len2 = len(prompt2)
        prompt = ""
        if len1 > 0 and len2 > 0:
            prompt = prompt1 + separator + prompt2
        elif len1 > 0:
            prompt = prompt1
        elif len2 > 0:
            prompt = prompt2
        return (prompt,)


# ====================================================================================================
# Input: sampler inputs
# ====================================================================================================

class SeargeSamplerInputs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "ddim"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "ddim_uniform"}),
        },
        }

    RETURN_TYPES = ("SAMPLER_NAME", "SCHEDULER_NAME",)
    RETURN_NAMES = ("sampler_name", "scheduler",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Inputs"

    def get_value(self, sampler_name, scheduler):
        return (sampler_name, scheduler,)


# ====================================================================================================
# Input: enabler inputs
# ====================================================================================================

class SeargeEnablerInputs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "state": (SeargeParameterProcessor.STATES, {"default": SeargeParameterProcessor.STATES[1]}),
        },
        }

    RETURN_TYPES = ("ENABLE_STATE",)
    RETURN_NAMES = ("state",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Inputs"

    def get_value(self, state):
        return (state,)


# ====================================================================================================
# Input: save folder inputs
# ====================================================================================================

class SeargeSaveFolderInputs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "save_to": (SeargeParameterProcessor.SAVE_TO, {"default": SeargeParameterProcessor.SAVE_TO[0]}),
        },
        }

    RETURN_TYPES = ("SAVE_FOLDER",)
    RETURN_NAMES = ("save_to",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Inputs"

    def get_value(self, save_to):
        return (save_to,)


# ====================================================================================================
# Tool: integer constant
# ====================================================================================================

class SeargeIntegerConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Integers"

    def get_value(self, value):
        return (value,)


# ====================================================================================================
# Tool: integer pair
# ====================================================================================================

class SeargeIntegerPair:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value1": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "value2": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("value 1", "value 2",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Integers"

    def get_value(self, value1, value2):
        return (value1, value2,)


# ====================================================================================================
# Tool: integer math
# ====================================================================================================

class SeargeIntegerMath:
    OPERATIONS = ["a * b + c", "a + c", "a - c", "a * b", "a / b"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "op": (SeargeIntegerMath.OPERATIONS, {"default": "a * b + c"}),
            "a": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "b": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
            "c": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Integers"

    def get_value(self, op, a, b, c):
        res = 0
        if op == "a * b + c":
            res = a * b + c
        elif op == "a + c":
            res = a + c
        elif op == "a - c":
            res = a - c
        elif op == "a * b":
            res = a * b
        elif op == "a / b":
            res = a // b
        return (int(res),)


# ====================================================================================================
# Tool: integer scaler
# ====================================================================================================

class SeargeIntegerScaler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "factor": ("FLOAT", {"default": 1.0, "step": 0.01}),
            "multiple_of": ("INT", {"default": 1, "min": 0, "max": 65536}),
        },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Integers"

    def get_value(self, value, factor, multiple_of):
        return (int(value * factor // multiple_of) * multiple_of,)


# ====================================================================================================
# Tool: float constant
# ====================================================================================================

class SeargeFloatConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("FLOAT", {"default": 0.0, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Floats"

    def get_value(self, value):
        return (value,)


# ====================================================================================================
# Tool: float pair
# ====================================================================================================

class SeargeFloatPair:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value1": ("FLOAT", {"default": 0.0, "step": 0.01}),
            "value2": ("FLOAT", {"default": 0.0, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT",)
    RETURN_NAMES = ("value 1", "value 2",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Floats"

    def get_value(self, value1, value2):
        return (value1, value2,)


# ====================================================================================================
# Tool: float math
# ====================================================================================================

class SeargeFloatMath:
    OPERATIONS = ["a * b + c", "a + c", "a - c", "a * b", "a / b"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "op": (SeargeFloatMath.OPERATIONS, {"default": "a * b + c"}),
            "a": ("FLOAT", {"default": 0.0, "step": 0.01}),
            "b": ("FLOAT", {"default": 1.0, "step": 0.01}),
            "c": ("FLOAT", {"default": 0.0, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"

    CATEGORY = "Searge/_deprecated_/Floats"

    def get_value(self, op, a, b, c):
        res = 0.0
        if op == "a * b + c":
            res = a * b + c
        elif op == "a + c":
            res = a + c
        elif op == "a - c":
            res = a - c
        elif op == "a * b":
            res = a * b
        elif op == "a / b":
            res = a / b
        return (res,)


# ====================================================================================================
# Util: custom save node (without preview)
# ====================================================================================================

class SeargeImageSave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "SeargeSDXL-%date%/Image"}),
                "state": ("ENABLE_STATE", {"default": SeargeParameterProcessor.STATES[1]}),
                "save_to": ("SAVE_FOLDER", {"default": SeargeParameterProcessor.SAVE_TO[0]}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Searge/_deprecated_/Files"

    def save_images(self, images, filename_prefix, state, save_to, prompt=None, extra_pnginfo=None):
        # "disabled"
        if state == SeargeParameterProcessor.STATES[0]:
            return {}

        # "input folder"
        if save_to == SeargeParameterProcessor.SAVE_TO[1]:
            output_dir = folder_paths.get_input_directory()
            filename_prefix = "output-%date%"
        # incl. SeargeParameterProcessor.SAVE_TO[0] -> "output folder"
        else:
            output_dir = folder_paths.get_output_directory()

        filename_prefix = filename_prefix.replace("%date%", datetime.now().strftime("%Y-%m-%d"))

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if args.disable_metadata is None or not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)

            counter += 1

        return {}


# ====================================================================================================
# Util: custom checkpoint loader
# ====================================================================================================

class SeargeCheckpointLoader:
    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": ("CHECKPOINT_NAME",),
        },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "Searge/_deprecated_/Files"

    def load_checkpoint(self, ckpt_name):
        return self.chkp_loader.load_checkpoint(ckpt_name)


# ====================================================================================================
# Util: custom vae loader
# ====================================================================================================

class SeargeVAELoader:
    def __init__(self):
        self.vae_loader = nodes.VAELoader()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae_name": ("VAE_NAME",),
        },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"

    CATEGORY = "Searge/_deprecated_/Files"

    def load_vae(self, vae_name):
        return self.vae_loader.load_vae(vae_name)


# ====================================================================================================
# Util: custom upscale model loader
# ====================================================================================================

class SeargeUpscaleModelLoader:
    def __init__(self):
        self.upscale_model_loader = comfy_extras.nodes_upscale_model.UpscaleModelLoader()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "upscaler_name": ("UPSCALER_NAME",),
        },
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_upscaler"

    CATEGORY = "Searge/_deprecated_/Files"

    def load_upscaler(self, upscaler_name):
        return self.upscale_model_loader.load_model(upscaler_name)


# ====================================================================================================
# Util: custom upscale model loader
# ====================================================================================================

class SeargeLoraLoader:
    def __init__(self):
        self.lora_loader = nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_name": ("LORA_NAME",),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("MODEL", "CLIP",)
    FUNCTION = "load_lora"

    CATEGORY = "Searge/_deprecated_/Files"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        return self.lora_loader.load_lora(model, clip, lora_name, strength_model, strength_clip)


# ====================================================================================================
# Tool: Muxer for selecting between 3 latent inputs
# ====================================================================================================

class SeargeLatentMuxer3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input0": ("LATENT",),
            "input1": ("LATENT",),
            "input2": ("LATENT",),
            "input_selector": ("INT", {"default": 0, "min": 0, "max": 2}),
        },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("output",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/FlowControl"

    def mux(self, input0, input1, input2, input_selector):
        if input_selector == 1:
            return (input1,)
        elif input_selector == 2:
            return (input2,)
        else:
            return (input0,)


# ====================================================================================================
# Tool: Muxer for selecting between 5 conditioning inputs
# ====================================================================================================

class SeargeConditioningMuxer2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input0": ("CONDITIONING",),
            "input1": ("CONDITIONING",),
            "input_selector": ("INT", {"default": 0, "min": 0, "max": 1}),
        },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/FlowControl"

    def mux(self, input0, input1, input_selector):
        if input_selector == 1:
            return (input1,)
        else:
            return (input0,)


# ====================================================================================================
# Tool: Muxer for selecting between 5 conditioning inputs
# ====================================================================================================

class SeargeConditioningMuxer5:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input0": ("CONDITIONING",),
            "input1": ("CONDITIONING",),
            "input2": ("CONDITIONING",),
            "input3": ("CONDITIONING",),
            "input4": ("CONDITIONING",),
            "input_selector": ("INT", {"default": 0, "min": 0, "max": 4}),
        },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "mux"

    CATEGORY = "Searge/_deprecated_/FlowControl"

    def mux(self, input0, input1, input2, input3, input4, input_selector):
        if input_selector == 1:
            return (input1,)
        elif input_selector == 2:
            return (input2,)
        elif input_selector == 3:
            return (input3,)
        elif input_selector == 4:
            return (input4,)
        else:
            return (input0,)


# ====================================================================================================
# Register nodes in ComfyUI
# ====================================================================================================

LEGACY_CLASS_MAPPINGS = {
    "SeargeSDXLSampler": SeargeSDXLSampler,
    "SeargeSDXLSampler2": SeargeSDXLSampler2,
    "SeargeSDXLSamplerV3": SeargeSDXLSamplerV3,
    "SeargeSDXLImage2ImageSampler": SeargeSDXLImage2ImageSampler,
    "SeargeSDXLImage2ImageSampler2": SeargeSDXLImage2ImageSampler2,

    "SeargeSamplerInputs": SeargeSamplerInputs,
    "SeargeEnablerInputs": SeargeEnablerInputs,
    "SeargeSaveFolderInputs": SeargeSaveFolderInputs,

    "SeargeSDXLPromptEncoder": SeargeSDXLPromptEncoder,
    "SeargeSDXLBasePromptEncoder": SeargeSDXLBasePromptEncoder,
    "SeargeSDXLRefinerPromptEncoder": SeargeSDXLRefinerPromptEncoder,

    "SeargePromptText": SeargePromptText,
    "SeargePromptCombiner": SeargePromptCombiner,

    "SeargeIntegerConstant": SeargeIntegerConstant,
    "SeargeIntegerPair": SeargeIntegerPair,
    "SeargeIntegerMath": SeargeIntegerMath,
    "SeargeIntegerScaler": SeargeIntegerScaler,

    "SeargeFloatConstant": SeargeFloatConstant,
    "SeargeFloatPair": SeargeFloatPair,
    "SeargeFloatMath": SeargeFloatMath,

    "SeargeImageSave": SeargeImageSave,
    "SeargeCheckpointLoader": SeargeCheckpointLoader,
    "SeargeVAELoader": SeargeVAELoader,
    "SeargeUpscaleModelLoader": SeargeUpscaleModelLoader,
    "SeargeLoraLoader": SeargeLoraLoader,

    "SeargeLatentMuxer3": SeargeLatentMuxer3,
    "SeargeConditioningMuxer2": SeargeConditioningMuxer2,
    "SeargeConditioningMuxer5": SeargeConditioningMuxer5,

    "SeargeParameterProcessor": SeargeParameterProcessor,
    "SeargeStylePreprocessor": SeargeStylePreprocessor,

    "SeargeInput1": SeargeInput1,
    "SeargeInput2": SeargeInput2,
    "SeargeInput3": SeargeInput3,
    "SeargeInput4": SeargeInput4,
    "SeargeInput5": SeargeInput5,
    "SeargeInput6": SeargeInput6,
    "SeargeInput7": SeargeInput7,

    "SeargeOutput1": SeargeOutput1,
    "SeargeOutput2": SeargeOutput2,
    "SeargeOutput3": SeargeOutput3,
    "SeargeOutput4": SeargeOutput4,
    "SeargeOutput5": SeargeOutput5,
    "SeargeOutput6": SeargeOutput6,
    "SeargeOutput7": SeargeOutput7,

    "SeargeGenerated1": SeargeGenerated1,

    # "SeargeSamplerAdvanced": SeargeSamplerAdvanced,
}

# ====================================================================================================
# Human readable names for the nodes
# ====================================================================================================


LEGACY_DISPLAY_NAME_MAPPINGS = {
    "SeargeSDXLSampler": "SDXL Sampler v1 (Searge)",
    "SeargeSDXLSampler2": "SDXL Sampler v2 (Searge)",
    "SeargeSDXLSamplerV3": "SDXL Sampler v3 (Searge)",
    "SeargeSDXLImage2ImageSampler": "Image2Image Sampler v1 (Searge)",
    "SeargeSDXLImage2ImageSampler2": "Image2Image Sampler v2 (Searge)",

    "SeargeSamplerInputs": "Sampler Settings",
    "SeargeEnablerInputs": "Enable / Disable",
    "SeargeSaveFolderInputs": "Save Folder",

    "SeargeSDXLPromptEncoder": "SDXL Prompt Encoder (Searge)",
    "SeargeSDXLBasePromptEncoder": "SDXL Base Prompt Encoder (Searge)",
    "SeargeSDXLRefinerPromptEncoder": "SDXL Refiner Prompt Encoder (Searge)",

    "SeargePromptText": "Prompt text input",
    "SeargePromptCombiner": "Prompt combiner",

    "SeargeIntegerConstant": "Integer Constant",
    "SeargeIntegerPair": "Integer Pair",
    "SeargeIntegerMath": "Integer Math",
    "SeargeIntegerScaler": "Integer Scaler",

    "SeargeFloatConstant": "Float Constant",
    "SeargeFloatPair": "Float Pair",
    "SeargeFloatMath": "Float Math",

    "SeargeImageSave": "Save Image (Searge)",
    "SeargeCheckpointLoader": "Checkpoint Loader (Searge)",
    "SeargeVAELoader": "VAE Loader (Searge)",
    "SeargeUpscaleModelLoader": "Upscale Model Loader (Searge)",
    "SeargeLoraLoader": "Lora Loader (Searge)",

    "SeargeLatentMuxer3": "3-Way Muxer for Latents",
    "SeargeConditioningMuxer2": "2-Way Muxer for Conditioning",
    "SeargeConditioningMuxer5": "5-Way Muxer for Conditioning",

    "SeargeParameterProcessor": "Parameter Processor",
    "SeargeStylePreprocessor": "Style Preprocessor (wip)",

    "SeargeInput1": "Prompts",
    "SeargeInput2": "Generation Parameters",
    "SeargeInput3": "Advanced Parameters",
    "SeargeInput4": "Model Names",
    "SeargeInput5": "Prompt Processing",
    "SeargeInput6": "HiResFix Parameters",
    "SeargeInput7": "Misc Parameters",

    "SeargeOutput1": "Prompts",
    "SeargeOutput2": "Generation Parameters",
    "SeargeOutput3": "Advanced Parameters",
    "SeargeOutput4": "Model Names",
    "SeargeOutput5": "Prompt Processing",
    "SeargeOutput6": "HiResFix Parameters",
    "SeargeOutput7": "Misc Parameters",

    "SeargeGenerated1": "Flow Control Parameters",

    # "SeargeSamplerAdvanced": "Experimental Sampler"
}

"""

# ====================================================================================================
# SDXL Image2Image Sampler (incl. HiRes Fix)
# ====================================================================================================

class SeargeSDXLImage2ImageSamplerV3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_model": ("MODEL",),
                    "base_positive": ("CONDITIONING", ),
                    "base_negative": ("CONDITIONING", ),
                    "refiner_model": ("MODEL",),
                    "refiner_positive": ("CONDITIONING",),
                    "refiner_negative": ("CONDITIONING",),
                    "image": ("IMAGE", ),
                    "vae": ("VAE",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff0}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 200}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                    "sampler_name": ("SAMPLER_NAME", {"default": "ddim"}),
                    "scheduler": ("SCHEDULER_NAME", {"default": "ddim_uniform"}),
                    "base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional": {
                    "upscale_model": ("UPSCALE_MODEL",),
                    "scaled_width": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "scaled_height": ("INT", {"default": 1536, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                    "parameters": ("PARAMETERS",),
#                    "noise_offset": ("INT", {"default": 1, "min": 0, "max": 1}),
#                    "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                    },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "sample"

    CATEGORY = "Searge/_deprecated_/Sampling"

#    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
                image, vae, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, softness,
                upscale_model=None, scaled_width=None, scaled_height=None, noise_offset=None, refiner_strength=None):
    def sample(self, base_model, base_positive, base_negative, refiner_model, refiner_positive, refiner_negative,
               image, vae, noise_seed, steps, cfg, sampler_name, scheduler, base_ratio, denoise, upscale_model=None,
                scaled_width=None, scaled_height=None, softness=0.0, parameters=None):
                
        base_steps = int(steps * (base_ratio + 0.0001))
        refiner_steps = max(0, steps - base_steps)

#        if noise_offset is None:
#            noise_offset = 1

#        if refiner_strength is None:
#            refiner_strength = 1.0

#        if refiner_strength < 0.01:
#            refiner_strength = 0.01

        if steps < 1:
            return (image, )

        scaled_image = image

        use_upscale_model = upscale_model is not None and softness < 0.9999
        if use_upscale_model:
            upscale_result = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(upscale_model, image)
            scaled_image = upscale_result[0]

        if scaled_width is not None and scaled_height is not None:
            upscale_result = nodes.ImageScale().upscale(scaled_image, "bicubic", scaled_width, scaled_height, "center")
            scaled_image = upscale_result[0]

        if use_upscale_model and softness > 0.0001:
            upscale_result = nodes.ImageScale().upscale(image, "bicubic", scaled_width, scaled_height, "center")
            scaled_original = upscale_result[0]

            blend_result = comfy_extras.nodes_post_processing.Blend().blend_images(scaled_image, scaled_original,
                                                                                   softness, "normal")
                                                                                   
            scaled_image = blend_result[0]

        if denoise < 0.01:
            return (scaled_image, )

        vae_encode_result = nodes.VAEEncode().encode(vae, scaled_image)
        input_latent = vae_encode_result[0]

        if base_steps >= steps:
            result_latent = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler,
                                                  base_positive, base_negative, input_latent, denoise=denoise,
                                                  disable_noise=False, start_step=0, last_step=steps,
                                                  force_full_denoise=True)
        else:
            result_latent = sdxl_ksampler(base_model, refiner_model, noise_seed, base_steps, refiner_steps, cfg,
                                          sampler_name, scheduler, base_positive, base_negative, refiner_positive,
                                          refiner_negative, input_latent, denoise=denoise, disable_noise=False,
                                          start_step=0, last_step=steps, force_full_denoise=True)

#            base_result = nodes.common_ksampler(base_model, noise_seed, steps, cfg, sampler_name, scheduler,
#                                                base_positive, base_negative, input_latent, denoise=denoise,
#                                                disable_noise=False, start_step=0, last_step=base_steps,
#                                                force_full_denoise=True)

#            result_latent = nodes.common_ksampler(refiner_model, noise_seed + noise_offset, steps, cfg, sampler_name,
#                                                  scheduler, refiner_positive, refiner_negative, base_result[0],
#                                                  denoise=denoise * refiner_strength, disable_noise=False,
#                                                  start_step=base_steps, last_step=steps, force_full_denoise=True)

        vae_decode_result = nodes.VAEDecode().decode(vae, result_latent[0])
        output_image = vae_decode_result[0]

        return (output_image, )

"""
