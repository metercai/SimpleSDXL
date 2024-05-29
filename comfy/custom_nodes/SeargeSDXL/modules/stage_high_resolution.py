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

from .data_utils import retrieve_parameter
from .mb_pipeline import PipelineAccess
from .names import Names
from .node_wrapper import NodeWrapper
from .ui import UI
from .utils import get_image_size
from .utils import next_multiple_of


# --------------------------------------------------------------------------------
# Stage: High Resolution
# --------------------------------------------------------------------------------

class SeargeStageHighResolution:
    SIZE_MULTIPLE_OF = 8

    def __init__(self):
        pass

    def get_input(self, data, stage_data):
        # if we still don't have stage data,
        if stage_data is None and data is not None:
            stage_data = {
                PipelineAccess.NAME: retrieve_parameter(PipelineAccess.NAME, data),
            }

        return stage_data

    def process(self, data, stage_input):
        access = PipelineAccess(stage_input)

        base_changed = access.changed_in_pipeline(Names.P_BASE_MODEL)
        refiner_changed = access.changed_in_pipeline(Names.P_REFINER_MODEL)

        base_model = access.get_from_pipeline(Names.P_BASE_MODEL)
        refiner_model = access.get_from_pipeline(Names.P_REFINER_MODEL)
        has_refiner = refiner_model is not None

        vae_changed = access.changed_in_pipeline(Names.P_VAE_MODEL)
        vae_model = access.get_from_pipeline(Names.P_VAE_MODEL)

        upscaler_changed = access.changed_in_pipeline(Names.P_HIRES_UPSCALER)
        upscale_model = access.get_from_pipeline(Names.P_HIRES_UPSCALER)

        detail_processor_changed = access.changed_in_pipeline(Names.P_DETAIL_PROCESSOR)
        detail_processor = access.get_from_pipeline(Names.P_DETAIL_PROCESSOR)

        base_cond_changed = access.changed_in_pipeline(Names.P_BASE_CONDITIONING)
        refiner_cond_changed = access.changed_in_pipeline(Names.P_REFINER_CONDITIONING)

        base_cond = access.get_from_pipeline(Names.P_BASE_CONDITIONING)
        refiner_cond = access.get_from_pipeline(Names.P_REFINER_CONDITIONING)

        base_positive = retrieve_parameter(Names.F_BASE_POSITIVE, base_cond)
        base_negative = retrieve_parameter(Names.F_BASE_NEGATIVE, base_cond)
        refiner_positive = retrieve_parameter(Names.F_REFINER_POSITIVE, refiner_cond)
        refiner_negative = retrieve_parameter(Names.F_REFINER_NEGATIVE, refiner_cond)

        # for now these are here to prepare for the future addition of latent upscaling
        latent_changed = access.changed_in_pipeline(Names.P_LATENT)
        latent = access.get_from_pipeline(Names.P_LATENT)

        image_changed = access.changed_in_pipeline(Names.P_IMAGE)
        image = access.get_from_pipeline(Names.P_IMAGE)

        seed = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SEED, 4815162342)
        steps = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_STEPS, 25)
        cfg = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_CFG, 7.0)
        sampler_name = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SAMPLER_NAME, "dpmpp_2m")
        scheduler = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SCHEDULER, "karras")
        base_ratio = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_BASE_VS_REFINER_RATIO, 0.8)

        hires_mode = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_MODE, UI.NONE)
        hires_mode_changed = access.setting_changed(UI.S_HIGH_RESOLUTION, UI.F_HIRES_MODE)
        hires_mode_enabled = hires_mode != UI.NONE

        hires_scale = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_SCALE, 1.5)
        hires_denoise = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_DENOISE, 0.2)
        hires_softness = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_SOFTNESS, 0.5)
        hires_detail_boost = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_DETAIL_BOOST, 0.0)
        hires_detailer = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_LATENT_DETAILER, UI.NONE)

        hires_detailer_enabled = hires_detailer != UI.NONE

        if not has_refiner:
            refiner_model = None
            refiner_positive = None
            refiner_negative = None
            base_ratio = 1.0
            hires_detail_boost = 0.0

        def run_sampler(latent, refiner_model, steps, denoise, cfg_method, dynamic_cfg, detail_boost):
            sampler = NodeWrapper.sdxl_sampler
            latent = sampler(base_model, base_positive, base_negative, latent, seed, steps, cfg,
                             sampler_name, scheduler, refiner_model=refiner_model,
                             refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                             base_ratio=base_ratio, denoise=denoise, cfg_method=cfg_method,
                             dynamic_base_cfg=dynamic_cfg, dynamic_refiner_cfg=dynamic_cfg,
                             refiner_detail_boost=detail_boost)
            return latent

        upscale_factor = 1.0
        if hires_scale == UI.HIRES_SCALE_1_25:
            upscale_factor = 1.25
        if hires_scale == UI.HIRES_SCALE_1_5:
            upscale_factor = 1.5
        if hires_scale == UI.HIRES_SCALE_2_0:
            upscale_factor = 2.0

        (image_width, image_height) = get_image_size(image)
        new_width = next_multiple_of(image_width * upscale_factor, self.SIZE_MULTIPLE_OF)
        new_height = next_multiple_of(image_height * upscale_factor, self.SIZE_MULTIPLE_OF)

        # use this to make sure old cached latents are not kept when the high resolution mode changes
        def cleanup_cache():
            # do this for all types of upscaled latents that we cache before the sampler at the end
            access.remove_from_cache(Names.C_HIRES_LATENT_SIMPLE)
            access.remove_from_cache(Names.C_HIRES_LATENT_NORMAL)

        larger = upscale_factor > 1.0
        if larger and hires_mode == UI.HIRES_MODE_SIMPLE:
            parameters = [
                hires_mode,
                image_width,
                image_height,
                hires_scale,
                upscale_factor,
                new_width,
                new_height,
                hires_softness,
            ]

            any_changes = (
                    vae_changed or
                    image_changed)

            hires_latent_changed = access.changed_in_cache(Names.C_HIRES_LATENT_SIMPLE, parameters)
            if any_changes or hires_latent_changed:
                need_nearest = hires_softness < 0.999
                need_bicubic = hires_softness > 0.001

                if need_nearest:
                    nearest = NodeWrapper.image_scale.upscale(image, "nearest-exact", new_width, new_height,
                                                              "center")[0]
                else:
                    nearest = None

                if need_bicubic:
                    bicubic = NodeWrapper.image_scale.upscale(image, "bicubic", new_width, new_height, "center")[0]
                else:
                    bicubic = None

                if need_nearest and need_bicubic and nearest is not None and bicubic is not None:
                    softened = NodeWrapper.image_blend.blend_images(nearest, bicubic, hires_softness, "normal")[0]
                elif need_nearest and nearest is not None:
                    softened = nearest
                elif need_bicubic and bicubic is not None:
                    softened = bicubic
                else:
                    softened = None

                if softened is not None:
                    latent = NodeWrapper.vae_encoder.encode(vae_model, softened)[0]

                cleanup_cache()
                access.update_in_cache(Names.C_HIRES_LATENT_SIMPLE, parameters, latent)
                access.update_in_pipeline(Names.P_LATENT, latent)
            else:
                latent = access.get_from_cache(Names.C_HIRES_LATENT_SIMPLE)
                access.restore_in_pipeline(Names.P_LATENT, latent)

        elif larger and hires_mode == UI.HIRES_MODE_NORMAL:
            parameters = [
                hires_mode,
                image_width,
                image_height,
                hires_scale,
                upscale_factor,
                new_width,
                new_height,
                hires_softness,
                hires_detailer_enabled,
            ]

            any_changes = (
                    vae_changed or
                    upscaler_changed or
                    detail_processor_changed or
                    image_changed)

            hires_latent_changed = access.changed_in_cache(Names.C_HIRES_LATENT_NORMAL, parameters)
            if any_changes or hires_latent_changed:
                need_upscaled = hires_softness < 0.999
                need_bicubic = hires_softness > 0.001

                if upscale_model is not None or detail_processor is not None:
                    if need_upscaled:
                        upscaled = image
                        (scaled_width, scaled_height) = (image_width, image_height)

                        if upscale_model is not None:
                            upscaled = NodeWrapper.scale_with_model.upscale(upscale_model, upscaled)[0]
                            (scaled_width, scaled_height) = get_image_size(upscaled)
                            if scaled_width != 4 * image_width or scaled_height != 4 * image_height:
                                print("Warning: high res upscaler should be a 4x ESRGAN model")

                        if detail_processor is not None and hires_detailer_enabled:
                            upscaled = NodeWrapper.scale_with_model.upscale(detail_processor, upscaled)[0]
                            (detailed_width, detailed_height) = get_image_size(upscaled)
                            if detailed_width != scaled_width or detailed_height != scaled_height:
                                print("Warning: detail processor should be a 1x ESRGAN model")

                        if scaled_width != new_width or scaled_height != new_height:
                            width_factor = float(scaled_width) / float(new_width)
                            height_factor = float(scaled_height) / float(new_height)

                            if width_factor >= 3.0 or height_factor >= 3.0:
                                step_width = next_multiple_of(new_width * 2.66666, self.SIZE_MULTIPLE_OF)
                                step_height = next_multiple_of(new_height * 2.66666, self.SIZE_MULTIPLE_OF)
                                image = NodeWrapper.image_scale.upscale(image, "bilinear", step_width, step_height,
                                                                        "center")[0]

                            if width_factor >= 2.5 or height_factor >= 2.5:
                                step_width = next_multiple_of(new_width * 2.0, self.SIZE_MULTIPLE_OF)
                                step_height = next_multiple_of(new_height * 2.0, self.SIZE_MULTIPLE_OF)
                                image = NodeWrapper.image_scale.upscale(image, "bilinear", step_width, step_height,
                                                                        "center")[0]

                            if width_factor >= 2.0 or height_factor >= 2.0:
                                step_width = next_multiple_of(new_width * 1.5, self.SIZE_MULTIPLE_OF)
                                step_height = next_multiple_of(new_height * 1.5, self.SIZE_MULTIPLE_OF)
                                image = NodeWrapper.image_scale.upscale(image, "bilinear", step_width, step_height,
                                                                        "center")[0]

                            if width_factor >= 1.5 or height_factor >= 1.5:
                                step_width = next_multiple_of(new_width * 1.33333, self.SIZE_MULTIPLE_OF)
                                step_height = next_multiple_of(new_height * 1.33333, self.SIZE_MULTIPLE_OF)
                                image = NodeWrapper.image_scale.upscale(image, "bilinear", step_width, step_height,
                                                                        "center")[0]

                        upscaled = NodeWrapper.image_scale.upscale(upscaled, "bicubic", new_width, new_height,
                                                                   "center")[0]
                    else:
                        upscaled = None

                    if need_bicubic:
                        bicubic = NodeWrapper.image_scale.upscale(image, "bicubic", new_width, new_height, "center")[0]
                    else:
                        bicubic = None

                    if need_upscaled and need_bicubic and upscaled is not None and bicubic is not None:
                        softened = NodeWrapper.image_blend.blend_images(upscaled, bicubic, hires_softness, "normal")[0]
                    elif need_upscaled and upscaled is not None:
                        softened = upscaled
                    elif need_bicubic and bicubic is not None:
                        softened = bicubic
                    else:
                        softened = None

                    if softened is not None:
                        latent = NodeWrapper.vae_encoder.encode(vae_model, softened)[0]
                else:
                    latent = None

                cleanup_cache()
                access.update_in_cache(Names.C_HIRES_LATENT_NORMAL, parameters, latent)
                access.update_in_pipeline(Names.P_LATENT, latent)
            else:
                latent = access.get_from_cache(Names.C_HIRES_LATENT_NORMAL)
                access.restore_in_pipeline(Names.P_LATENT, latent)

        else:
            latent = None
            if hires_mode_changed:
                cleanup_cache()
                access.update_in_pipeline(Names.P_LATENT, latent)
            else:
                access.restore_in_pipeline(Names.P_LATENT, latent)

        parameters = [
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            base_ratio,
            hires_softness,
            hires_denoise,
            hires_detail_boost,
        ]

        latent_changed = access.changed_in_pipeline(Names.P_LATENT)

        # DON'T DO THIS HERE (keep using the current latent variable): latent = access.get_from_pipeline(Names.P_LATENT)

        any_changes = (
                base_changed or
                refiner_changed or
                base_cond_changed or
                refiner_cond_changed or
                latent_changed)

        hires_latent_changed = access.changed_in_cache(Names.C_HIRES_LATENT, parameters)
        if any_changes or hires_latent_changed:
            if hires_mode_enabled and latent is not None:
                if "noise_mask" in latent:
                    latent = latent.clone()
                    latent.pop("noise_mask")

                hires_steps = int((steps * 2 + 2) // 3)
                latent = run_sampler(latent, refiner_model, hires_steps, hires_denoise, cfg_method=None,
                                     dynamic_cfg=0.0, detail_boost=hires_detail_boost)
            else:
                latent = None

            # NOTE: it's important NOT to call the cleanup cache function here, because it's unrelated to this cache
            access.update_in_cache(Names.C_HIRES_LATENT, parameters, latent)
            access.update_in_pipeline(Names.P_LATENT, latent)
        else:
            latent = access.get_from_cache(Names.C_HIRES_LATENT)
            access.restore_in_pipeline(Names.P_LATENT, latent)

        high_res_output = {
            Names.F_LATENT_HIRES: latent,
        }

        if data is not None:
            data[Names.S_HIRES_OUTPUT] = high_res_output

        stage_output = {
            Names.S_HIRES_OUTPUT: high_res_output,
        }

        return (data, stage_output,)
