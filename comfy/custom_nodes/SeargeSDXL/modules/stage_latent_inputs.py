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

from ._experimental import gaussian_latent_noise
from .data_utils import retrieve_parameter
from .mb_pipeline import PipelineAccess
from .names import Names
from .node_wrapper import NodeWrapper
from .ui import UI
from .utils import get_image_size
from .utils import get_mask_size


# --------------------------------------------------------------------------------
# Stage: Latent Inputs
# --------------------------------------------------------------------------------

class SeargeStageLatentInputs:
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

        vae_changed = access.changed_in_pipeline(Names.P_VAE_MODEL)
        vae_model = access.get_from_pipeline(Names.P_VAE_MODEL)

        image_width = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_WIDTH, 1024)
        image_height = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_HEIGHT, 1024)

        seed = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SEED, 4815162342)

        workflow_mode = access.get_active_setting(UI.S_OPERATING_MODE, UI.F_WORKFLOW_MODE, UI.WF_MODE_TEXT_TO_IMAGE)

        image_changed = access.changed_in_pipeline(Names.P_IMAGE)
        mask_changed = access.changed_in_pipeline(Names.P_MASK)

        image = access.get_from_pipeline(Names.P_IMAGE)
        mask = access.get_from_pipeline(Names.P_MASK)

        batch_size = access.get_active_setting(UI.S_OPERATING_MODE, UI.F_BATCH_SIZE, 1)

        precondition_mode = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_PRECONDITION_MODE, UI.NONE)
        precondition_strength = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_PRECONDITION_STRENGTH, 0.1)

        latent = None
        if workflow_mode == UI.WF_MODE_IMAGE_TO_IMAGE or workflow_mode == UI.WF_MODE_IN_PAINTING:
            parameters = [
                image_width,
                image_height,
                batch_size,
                workflow_mode,
            ]

            latent_changed = access.changed_in_cache(Names.C_LATENT_FROM_IMAGE, parameters)
            if latent_changed or image_changed or vae_changed:
                (width, height) = get_image_size(image)

                if width != image_width or height != image_height:
                    image = NodeWrapper.image_scale.upscale(image, "bicubic", image_width, image_height, "center")[0]
                    access.update_in_pipeline(Names.P_IMAGE, image)

                latent = NodeWrapper.vae_encoder.encode(vae_model, image)[0]

                if batch_size > 1:
                    # NOTE: only repeat with batch size here if we are not in inpainting mode (optimization)
                    if workflow_mode == UI.WF_MODE_IMAGE_TO_IMAGE:
                        latent = NodeWrapper.latent_repeater.repeat(latent, batch_size)[0]

                    image = image.repeat(batch_size, 1, 1, 1)
                    access.update_in_pipeline(Names.P_IMAGE, image)

                access.remove_from_cache(Names.C_EMPTY_LATENT)
                access.remove_from_cache(Names.C_IMAGE_MASK)
                access.remove_from_cache(Names.C_LATENT_WITH_MASK)
                access.update_in_cache(Names.C_LATENT_FROM_IMAGE, parameters, (latent, image))
                access.update_in_pipeline(Names.P_LATENT, latent)
            else:
                (latent, image) = access.get_from_cache(Names.C_LATENT_FROM_IMAGE)
                access.restore_in_pipeline(Names.P_LATENT, latent)
                access.restore_in_pipeline(Names.P_IMAGE, image)

        if workflow_mode == UI.WF_MODE_IN_PAINTING:
            parameters = [
                image_width,
                image_height,
                batch_size,
                workflow_mode,
            ]

            image_mask_changed = access.changed_in_cache(Names.C_IMAGE_MASK, parameters)
            if mask_changed or image_mask_changed:
                (width, height) = get_mask_size(mask)

                if width != image_width or height != image_height:
                    image_scale = NodeWrapper.image_scale

                    mask_image = NodeWrapper.mask_to_image.mask_to_image(mask)[0]
                    mask_image = image_scale.upscale(mask_image, "bicubic", image_width, image_height, "center")[0]

                    mask = NodeWrapper.image_to_mask.image_to_mask(mask_image, "green")[0]

                access.remove_from_cache(Names.C_EMPTY_LATENT)
                access.remove_from_cache(Names.C_LATENT_WITH_MASK)
                access.update_in_cache(Names.C_IMAGE_MASK, parameters, mask)
                access.update_in_pipeline(Names.P_MASK, mask)
            else:
                mask = access.get_from_cache(Names.C_IMAGE_MASK)
                access.restore_in_pipeline(Names.P_MASK, mask)

            latent_changed = access.changed_in_pipeline(Names.P_LATENT)
            mask_changed = access.changed_in_pipeline(Names.P_MASK)
            if latent_changed or mask_changed:
                latent = access.get_from_pipeline(Names.P_LATENT)

                # in case we are using an older cached latent that was already repeated with batch size, take only first
                latent = NodeWrapper.latent_selector.frombatch(latent, 0, 1)[0]

                latent = NodeWrapper.set_latent_mask.set_mask(latent, mask)[0]

                # repeat with batch size, will also repeat the mask in addition to the latent
                if batch_size > 1:
                    latent = NodeWrapper.latent_repeater.repeat(latent, batch_size)[0]

                access.remove_from_cache(Names.C_EMPTY_LATENT)
                access.update_in_cache(Names.C_LATENT_WITH_MASK, parameters, latent)
                access.update_in_pipeline(Names.P_LATENT, latent)
            else:
                latent = access.get_from_cache(Names.C_LATENT_WITH_MASK)
                access.restore_in_pipeline(Names.P_LATENT, latent)

        if workflow_mode == UI.WF_MODE_TEXT_TO_IMAGE:
            parameters = [
                seed,
                image_width,
                image_height,
                batch_size,
                precondition_mode,
                precondition_strength,
            ]

            empty_latent_changed = access.changed_in_cache(Names.C_EMPTY_LATENT, parameters)
            if empty_latent_changed:
                if precondition_mode == UI.NONE or precondition_strength < 0.001:
                    latent = NodeWrapper.empty_latent.generate(image_width, image_height, batch_size)[0]
                elif precondition_mode == UI.PRECONDITION_MODE_GAUSSIAN:
                    latent = gaussian_latent_noise(image_width // 8, image_height // 8, seed, precondition_strength,
                                                   batch_size)
                else:
                    latent = NodeWrapper.empty_latent.generate(image_width, image_height, batch_size)[0]

                access.remove_from_cache(Names.C_LATENT_FROM_IMAGE)
                access.remove_from_cache(Names.C_IMAGE_MASK)
                access.remove_from_cache(Names.C_LATENT_WITH_MASK)
                access.update_in_cache(Names.C_EMPTY_LATENT, parameters, latent)
                access.update_in_pipeline(Names.P_LATENT, latent)
            else:
                latent = access.get_from_cache(Names.C_EMPTY_LATENT)
                access.restore_in_pipeline(Names.P_LATENT, latent)

        latent_input = {
            Names.F_LATENT_IMAGE: latent,
        }

        if data is not None:
            data[Names.S_LATENT_INPUTS] = latent_input

        stage_output = {
            Names.S_LATENT_INPUTS: latent_input,
        }

        return (data, stage_output,)
