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

import torch
import warnings

import comfy.sample
import comfy.samplers
import comfy.sampler_helpers
import comfy.utils
import latent_preview

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.model_management import get_torch_device, load_models_gpu

from .utils import slerp_latents
from .utils import bilateral_blur


# --------------------------------------------------------------------------------

class CfgMethods:
    INTERPOLATE = "interpolate"
    RESCALE = "rescale"
    TONEMAP = "tonemap"


# --------------------------------------------------------------------------------

def unet_function(func, params):
    cond_or_uncond = params["cond_or_uncond"]

    input_x = params["input"]
    timestep = params["timestep"]
    c = params["c"]

    transformer_options = c["transformer_options"]
    transformer_options["uc_mask"] = torch.Tensor(cond_or_uncond).to(input_x).float()[:, None, None, None]

    return func(input_x, timestep, **c)


# --------------------------------------------------------------------------------

def new_unet_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    x0 = old_unet_forward(self, x, timesteps, context, y, control, transformer_options, **kwargs)

    # do filtering here
    if "uc_mask" in transformer_options:
        uc_mask = transformer_options["uc_mask"]

        count = timesteps.shape[0] // uc_mask.shape[0]
        uc_mask = uc_mask.repeat(count, 1, 1, 1)

        sharpness = 2.0
        alpha = 1.0 - (timesteps / 999.0)[:, None, None, None].clone()
        alpha *= 0.001 * sharpness
        degraded_x0 = bilateral_blur(x0, (13, 13), 3.0, 3.0) * alpha + x0 * (1.0 - alpha)
        x0 = x0 * uc_mask + degraded_x0 * (1.0 - uc_mask)

    return x0


old_unet_forward = UNetModel.forward
UNetModel.forward = new_unet_forward


# --------------------------------------------------------------------------------

def sdxl_sample(base_model, refiner_model, noise, base_steps, refiner_steps, cfg, sampler_name, scheduler,
                base_positive, base_negative, refiner_positive, refiner_negative, latent_image, batch_inds,
                denoise=1.0, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None,
                base_callback=None, refiner_callback=None, disable_pbar=False, seed=None, cfg_method=None,
                dynamic_base_cfg=0.0, dynamic_refiner_cfg=0.0, refiner_detail_boost=0.0):
    device = get_torch_device()

    if noise_mask is not None:
        noise_mask = comfy.sampler_helpers.prepare_mask(noise_mask, noise.shape, device)

    steps = base_steps + refiner_steps

    def base_cfg_callback(args):
        (cond, uncond, cond_scale, timestep) = (args["cond"], args["uncond"], args["cond_scale"], args["timestep"])

        dyn_cfg = dynamic_base_cfg

        if dyn_cfg < 0.0:
            dyn_cfg = -dyn_cfg
            ts = 1.0 - float(timestep) / 999.0
        else:
            ts = float(timestep) / 999.0

        if dyn_cfg > 0.0999:
            cond_scale = cond_scale * ts + (cond_scale * (1.0 - dyn_cfg) + dyn_cfg) * (1.0 - ts)

        return uncond + (cond - uncond) * cond_scale

    def base_rescale_cfg(args):
        multiplier = dynamic_base_cfg if dynamic_base_cfg >= 0.0 else -dynamic_base_cfg

        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]

        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)

        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

        return x_final

    def base_tonemap_reinhard(args):
        multiplier = dynamic_base_cfg if dynamic_base_cfg >= 0.0 else -dynamic_base_cfg

        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]

        noise_pred = (cond - uncond)
        noise_pred_vector_magnitude = (torch.linalg.vector_norm(noise_pred, dim=(1)) + 0.0000000001)[:, None]
        noise_pred /= noise_pred_vector_magnitude

        mean = torch.mean(noise_pred_vector_magnitude, dim=(1, 2, 3), keepdim=True)
        std = torch.std(noise_pred_vector_magnitude, dim=(1, 2, 3), keepdim=True)

        top = (std * 3 + mean) * multiplier

        noise_pred_vector_magnitude *= (1.0 / top)
        new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
        new_magnitude *= top

        return uncond + noise_pred * new_magnitude * cond_scale

    base_model = base_model.clone()
    base_model.set_model_unet_function_wrapper(unet_function)

    if cfg_method is not None:
        if cfg_method == CfgMethods.INTERPOLATE:
            base_model.set_model_sampler_cfg_function(base_cfg_callback)
        elif cfg_method == CfgMethods.RESCALE and dynamic_base_cfg > 0.0:
            base_model.set_model_sampler_cfg_function(base_rescale_cfg)
        elif cfg_method == CfgMethods.TONEMAP and dynamic_base_cfg > 0.0:
            base_model.set_model_sampler_cfg_function(base_tonemap_reinhard)

    temp_base_conds = {"positive": base_positive, "negative": base_negative}
    base_conds = {}
    for k in temp_base_conds:
        base_conds[k] = comfy.sampler_helpers.convert_cond(temp_base_conds[k])

    base_models, inference_memory = comfy.sampler_helpers.get_additional_models(base_conds,
                                                                                base_model.model_dtype())

    real_base_model = base_model.model

    original_latent = latent_image

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    base_sampler = comfy.samplers.KSampler(base_model, steps=steps, device=device, sampler=sampler_name,
                                           scheduler=scheduler, denoise=denoise, model_options=base_model.model_options)

    base_samples = base_sampler.sample(noise, base_positive, base_negative, cfg=cfg, latent_image=latent_image,
                                       start_step=start_step, last_step=base_steps, force_full_denoise=False,
                                       denoise_mask=noise_mask, sigmas=sigmas, callback=base_callback,
                                       disable_pbar=disable_pbar, seed=seed)

    comfy.sampler_helpers.cleanup_additional_models(base_models)

    noise = torch.zeros(base_samples.size(), dtype=base_samples.dtype, layout=base_samples.layout, device=device)

    if refiner_steps < 1:
        return base_samples.cpu()

    if refiner_detail_boost > 0.0:
        new_noise = comfy.sample.prepare_noise(original_latent, seed + 1, batch_inds).to(device)
        new_noise /= real_base_model.latent_format.scale_factor

        factor = base_sampler.sigmas[-refiner_steps - 1]
        new_noise = new_noise * factor

        noised_samples = base_samples + new_noise

        base_samples = slerp_latents(base_samples, noised_samples, refiner_detail_boost)

    if noise_mask is not None:
        latent_from_base = base_samples * noise_mask + latent_image * (1.0 - noise_mask)
    else:
        latent_from_base = base_samples

    def refiner_cfg_callback(args):
        (cond, uncond, cond_scale, timestep) = (args["cond"], args["uncond"], args["cond_scale"], args["timestep"])

        dyn_cfg = dynamic_refiner_cfg

        if dyn_cfg < 0.0:
            dyn_cfg = -dyn_cfg
            ts = 1.0 - float(timestep) / 999.0
        else:
            ts = float(timestep) / 999.0

        if dyn_cfg > 0.0999:
            cond_scale = cond_scale * ts + (cond_scale * (1.0 - dyn_cfg) + dyn_cfg) * (1.0 - ts)

        return uncond + (cond - uncond) * cond_scale

    def refiner_rescale_cfg(args):
        multiplier = dynamic_refiner_cfg if dynamic_refiner_cfg >= 0.0 else -dynamic_refiner_cfg

        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]

        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)

        x_rescaled = x_cfg * (ro_pos / ro_cfg)

        return multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

    def refiner_tonemap_reinhard(args):
        multiplier = dynamic_refiner_cfg if dynamic_refiner_cfg >= 0.0 else -dynamic_refiner_cfg

        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]

        noise_pred = (cond - uncond)
        noise_pred_vector_magnitude = (torch.linalg.vector_norm(noise_pred, dim=(1)) + 0.0000000001)[:, None]
        noise_pred /= noise_pred_vector_magnitude

        mean = torch.mean(noise_pred_vector_magnitude, dim=(1, 2, 3), keepdim=True)
        std = torch.std(noise_pred_vector_magnitude, dim=(1, 2, 3), keepdim=True)

        top = (std * 3 + mean) * multiplier

        noise_pred_vector_magnitude *= (1.0 / top)
        new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
        new_magnitude *= top

        return uncond + noise_pred * new_magnitude * cond_scale

    refiner_model = refiner_model.clone()
    refiner_model.set_model_unet_function_wrapper(unet_function)

    if cfg_method is not None:
        if cfg_method == CfgMethods.INTERPOLATE:
            refiner_model.set_model_sampler_cfg_function(refiner_cfg_callback)
        elif cfg_method == CfgMethods.RESCALE and dynamic_refiner_cfg > 0.0:
            refiner_model.set_model_sampler_cfg_function(refiner_rescale_cfg)
        elif cfg_method == CfgMethods.TONEMAP and dynamic_refiner_cfg > 0.0:
            refiner_model.set_model_sampler_cfg_function(refiner_tonemap_reinhard)

    temp_refiner_conds = {"positive": refiner_positive, "negative": refiner_negative}
    refiner_conds = {}
    for k in temp_refiner_conds:
        refiner_conds[k] = comfy.sampler_helpers.convert_cond(temp_refiner_conds[k])

    refiner_models, inference_memory = comfy.sampler_helpers.get_additional_models(refiner_conds,
                                                                                   refiner_model.model_dtype())

    refiner_sampler = comfy.samplers.KSampler(refiner_model, steps=steps, device=device, sampler=sampler_name,
                                              scheduler=scheduler, denoise=denoise,
                                              model_options=refiner_model.model_options)

    refiner_samples = refiner_sampler.sample(noise, refiner_positive, refiner_negative, cfg=cfg,
                                             latent_image=latent_from_base, start_step=base_steps, last_step=last_step,
                                             force_full_denoise=force_full_denoise,
                                             denoise_mask=noise_mask, sigmas=sigmas, callback=refiner_callback,
                                             disable_pbar=disable_pbar, seed=seed)

    refiner_samples = refiner_samples.cpu()

    comfy.sampler_helpers.cleanup_additional_models(refiner_models)

    return refiner_samples


# --------------------------------------------------------------------------------

def sdxl_ksampler(base_model, refiner_model, seed, base_steps, refiner_steps, cfg, sampler_name, scheduler,
                  base_positive, base_negative, refiner_positive, refiner_negative, latent, denoise=1.0,
                  disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, cfg_method=None,
                  dynamic_base_cfg=0.0, dynamic_refiner_cfg=0.0, refiner_detail_boost=0.0):
    device = get_torch_device()
    latent_image = latent["samples"]

    batch_inds = None
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    base_previewer = latent_preview.get_previewer(device, base_model.model.latent_format)
    refiner_previewer = None
    if refiner_model is not None:
        refiner_previewer = latent_preview.get_previewer(device, refiner_model.model.latent_format)

    steps = base_steps + refiner_steps
    pbar = comfy.utils.ProgressBar(steps)

    def base_callback(step, x0, x, total_steps):
        preview_bytes = None
        if base_previewer:
            preview_bytes = base_previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    def refiner_callback(step, x0, x, total_steps):
        preview_bytes = None
        if refiner_previewer:
            preview_bytes = refiner_previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sdxl_sample(base_model, refiner_model, noise, base_steps, refiner_steps, cfg, sampler_name, scheduler,
                              base_positive, base_negative, refiner_positive, refiner_negative, latent_image,
                              batch_inds, denoise=denoise, start_step=start_step, last_step=last_step,
                              force_full_denoise=force_full_denoise, noise_mask=noise_mask,
                              base_callback=base_callback, refiner_callback=refiner_callback, seed=seed,
                              dynamic_base_cfg=dynamic_base_cfg, dynamic_refiner_cfg=dynamic_refiner_cfg,
                              cfg_method=cfg_method, refiner_detail_boost=refiner_detail_boost)

    out = latent.copy()
    out["samples"] = samples
    return (out,)
