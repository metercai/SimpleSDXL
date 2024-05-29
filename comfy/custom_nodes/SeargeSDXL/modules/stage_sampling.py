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


# --------------------------------------------------------------------------------
# Stage: Sampling
# --------------------------------------------------------------------------------

class SeargeStageSampling:
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

        base_cond_changed = access.changed_in_pipeline(Names.P_BASE_CONDITIONING)
        refiner_cond_changed = access.changed_in_pipeline(Names.P_REFINER_CONDITIONING)

        base_cond = access.get_from_pipeline(Names.P_BASE_CONDITIONING)
        refiner_cond = access.get_from_pipeline(Names.P_REFINER_CONDITIONING)

        base_positive = retrieve_parameter(Names.F_BASE_POSITIVE, base_cond)
        base_negative = retrieve_parameter(Names.F_BASE_NEGATIVE, base_cond)
        refiner_positive = retrieve_parameter(Names.F_REFINER_POSITIVE, refiner_cond)
        refiner_negative = retrieve_parameter(Names.F_REFINER_NEGATIVE, refiner_cond)

        latent_changed = access.changed_in_pipeline(Names.P_LATENT)
        latent = access.get_from_pipeline(Names.P_LATENT)

        seed = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SEED, 4815162342)
        steps = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_STEPS, 25)
        cfg = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_CFG, 7.0)
        sampler_name = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SAMPLER_NAME, "dpmpp_2m")
        scheduler = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SCHEDULER, "karras")
        base_ratio = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_BASE_VS_REFINER_RATIO, 0.8)

        denoise = access.get_active_setting(UI.S_IMG2IMG_INPAINTING, UI.F_DENOISE, 0.5)

        cfg_method = access.get_active_setting(UI.S_ADVANCED_PARAMETERS, UI.F_DYNAMIC_CFG_METHOD)
        dynamic_cfg = access.get_active_setting(UI.S_ADVANCED_PARAMETERS, UI.F_DYNAMIC_CFG_FACTOR, 0.0)
        refiner_detail_boost = access.get_active_setting(UI.S_ADVANCED_PARAMETERS, UI.F_REFINER_DETAIL_BOOST)

        dynamic_base_cfg = dynamic_cfg
        dynamic_refiner_cfg = dynamic_cfg

        if not has_refiner:
            refiner_model = None
            refiner_positive = None
            refiner_negative = None
            base_ratio = 1.0
            dynamic_refiner_cfg = 0.0
            refiner_detail_boost = None

        parameters = [
            has_refiner,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            base_ratio,
            denoise,
            cfg_method,
            dynamic_base_cfg,
            dynamic_refiner_cfg,
            refiner_detail_boost,
        ]

        any_changes = (
                base_changed or
                refiner_changed or
                base_cond_changed or
                refiner_cond_changed or
                latent_changed)

        sampled_changed = access.changed_in_cache(Names.C_SAMPLED, parameters)
        if any_changes or sampled_changed:
            sampler = NodeWrapper.sdxl_sampler
            latent = sampler(base_model, base_positive, base_negative, latent, seed, steps, cfg,
                             sampler_name, scheduler, refiner_model=refiner_model,
                             refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                             base_ratio=base_ratio, denoise=denoise, cfg_method=cfg_method,
                             dynamic_base_cfg=dynamic_base_cfg, dynamic_refiner_cfg=dynamic_refiner_cfg,
                             refiner_detail_boost=refiner_detail_boost)

            access.update_in_cache(Names.C_SAMPLED, parameters, latent)
            access.update_in_pipeline(Names.P_LATENT, latent)
        else:
            latent = access.get_from_cache(Names.C_SAMPLED)
            access.restore_in_pipeline(Names.P_LATENT, latent)

        sampled_image = {
            Names.F_LATENT_SAMPLED: latent,
        }

        if data is not None:
            data[Names.S_SAMPLED_IMAGE] = sampled_image

        stage_output = {
            Names.S_SAMPLED_IMAGE: sampled_image,
        }

        return (data, stage_output,)
