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

from comfy.sampler_helpers import prepare_mask

from .data_utils import retrieve_parameter
from .mb_pipeline import PipelineAccess
from .names import Names
from .node_wrapper import NodeWrapper
from .ui import UI


# --------------------------------------------------------------------------------
# Stage: Latent Detailer
# --------------------------------------------------------------------------------

class SeargeStageLatentDetailer:
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
        base_model = access.get_from_pipeline(Names.P_BASE_MODEL)

        base_cond_changed = access.changed_in_pipeline(Names.P_BASE_CONDITIONING)
        base_cond = access.get_from_pipeline(Names.P_BASE_CONDITIONING)

        base_positive = retrieve_parameter(Names.F_BASE_POSITIVE, base_cond)
        base_negative = retrieve_parameter(Names.F_BASE_NEGATIVE, base_cond)

        latent_changed = access.changed_in_pipeline(Names.P_LATENT)
        latent = access.get_from_pipeline(Names.P_LATENT)

        seed = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SEED, 4815162342)
        cfg = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_CFG, 7.0)

        latent_detailer = access.get_active_setting(UI.S_ADVANCED_PARAMETERS, UI.F_LATENT_DETAILER, UI.NONE)

        parameters = [
            seed,
            cfg,
            latent_detailer,
        ]

        any_changes = (
                base_changed or
                base_cond_changed or
                latent_changed)

        sampled_changed = access.changed_in_cache(Names.C_SAMPLED_DETAILER, parameters)
        if any_changes or sampled_changed:
            latent_original = latent
            if latent_detailer == UI.DETAILER_NORMAL:
                latent = self.detailer(latent, 5, "nearest-exact", base_model, base_positive, base_negative, seed, cfg)
            elif latent_detailer == UI.DETAILER_SOFT:
                latent = self.detailer(latent, 5, "bicubic", base_model, base_positive, base_negative, seed, cfg)
            elif latent_detailer == UI.DETAILER_BLURRY:
                latent = self.detailer(latent, 10, "nearest-exact", base_model, base_positive, base_negative, seed, cfg)
            elif latent_detailer == UI.DETAILER_SOFT_BLURRY:
                latent = self.detailer(latent, 10, "bicubic", base_model, base_positive, base_negative, seed, cfg)

            if "noise_mask" in latent_original and "samples" in latent_original and "samples" in latent:
                old_samples = latent_original["samples"]
                new_samples = latent["samples"]

                noise_mask = latent_original["noise_mask"]
                noise_mask = prepare_mask(noise_mask, old_samples.shape, "cpu")

                latent["samples"] = new_samples * noise_mask + old_samples * (1.0 - noise_mask)

            access.update_in_cache(Names.C_SAMPLED_DETAILER, parameters, latent)
            access.update_in_pipeline(Names.P_LATENT, latent)
        else:
            latent = access.get_from_cache(Names.C_SAMPLED_DETAILER)
            access.restore_in_pipeline(Names.P_LATENT, latent)

        detailed_output = {
            Names.F_DETAILED_SAMPLED: latent,
        }

        if data is not None:
            data[Names.S_LATENT_DETAILED] = detailed_output

        stage_output = {
            Names.S_LATENT_DETAILED: detailed_output,
        }

        return (data, stage_output,)

    def detailer(self, latent, percent, method, base_model, base_positive, base_negative, seed, cfg):
        sampler = NodeWrapper.common_sampler
        scaler = NodeWrapper.latent_upscale_by

        sampler_name = "dpmpp_2m"
        scheduler = "karras"

        latent = scaler.upscale(latent, method, 2.0)[0]

        latent = sampler(base_model, seed, 100, cfg, sampler_name, scheduler,
                         base_positive, base_negative, latent, denoise=1.0, disable_noise=False,
                         start_step=int(100 - percent * 2), last_step=int(100 - percent),
                         force_full_denoise=False)

        latent = scaler.upscale(latent, method, 0.5)[0]

        latent = sampler(base_model, seed, 100, cfg, sampler_name, scheduler,
                         base_positive, base_negative, latent, denoise=1.0, disable_noise=True,
                         start_step=int(100 - percent * 2), last_step=100,
                         force_full_denoise=True)

        return latent
