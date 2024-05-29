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

import nodes
import comfy_extras.nodes_mask
import comfy_extras.nodes_post_processing
import comfy_extras.nodes_clip_sdxl
import comfy_extras.nodes_upscale_model
import comfy_extras.nodes_freelunch

from .custom_sdxl_ksampler import sdxl_ksampler
from .ui import UI


# ====================================================================================================
# Wrapper for other ComfyUI nodes
# ====================================================================================================

class NodeWrapper:
    checkpoint_loader = nodes.CheckpointLoaderSimple()
    clipvision_encoder = nodes.CLIPVisionEncode()
    clipvision_loader = nodes.CLIPVisionLoader()
    controlnet_advanced = nodes.ControlNetApplyAdvanced()
    controlnet_loader = nodes.ControlNetLoader()
    empty_latent = nodes.EmptyLatentImage()
    freeu = comfy_extras.nodes_freelunch.FreeU()
    freeu_v2 = comfy_extras.nodes_freelunch.FreeU_V2()
    image_blend = comfy_extras.nodes_post_processing.Blend()
    image_blur = comfy_extras.nodes_post_processing.Blur()
    image_composite = comfy_extras.nodes_mask.ImageCompositeMasked()
    image_scale = nodes.ImageScale()
    image_to_mask = comfy_extras.nodes_mask.ImageToMask()
    latent_repeater = nodes.RepeatLatentBatch()
    latent_selector = nodes.LatentFromBatch()
    latent_upscale_by = nodes.LatentUpscaleBy()
    lora_loader = nodes.LoraLoader()
    mask_to_image = comfy_extras.nodes_mask.MaskToImage()
    scale_with_model = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel()
    sdxl_clip_base_encoder = comfy_extras.nodes_clip_sdxl.CLIPTextEncodeSDXL()
    sdxl_clip_refiner_encoder = comfy_extras.nodes_clip_sdxl.CLIPTextEncodeSDXLRefiner()
    set_latent_mask = nodes.SetLatentNoiseMask()
    unclip_conditioning = nodes.unCLIPConditioning()
    upscale_loader = comfy_extras.nodes_upscale_model.UpscaleModelLoader()
    vae_decoder = nodes.VAEDecode()
    vae_encoder = nodes.VAEEncode()
    vae_loader = nodes.VAELoader()
    zero_out_cond = nodes.ConditioningZeroOut()

    @staticmethod
    def sdxl_sampler(base_model, base_positive, base_negative, latent_image, noise_seed, steps, cfg,
                     sampler_name, scheduler, refiner_model=None, refiner_positive=None, refiner_negative=None,
                     base_ratio=0.8, denoise=1.0, cfg_method=None, dynamic_base_cfg=0.0, dynamic_refiner_cfg=0.0,
                     refiner_detail_boost=0.0):
        if base_model is None:
            return None

        has_refiner_model = refiner_model is not None

        base_steps = int(steps * (base_ratio + 0.0001)) if has_refiner_model else steps
        refiner_steps = max(0, steps - base_steps)

        if cfg_method == UI.NONE:
            cfg_method = None

        if denoise < 0.005:
            return latent_image

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

        return result[0]

    @staticmethod
    def common_sampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                       denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        result = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                       latent, denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                       last_step=last_step, force_full_denoise=force_full_denoise)

        return result[0]
