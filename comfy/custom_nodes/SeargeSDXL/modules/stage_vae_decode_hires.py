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
# Stage: VAE Decode Hires
# --------------------------------------------------------------------------------

class SeargeStageVAEDecodeHires:
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

        latent_changed = access.changed_in_pipeline(Names.P_LATENT)
        latent = access.get_from_pipeline(Names.P_LATENT)

        hires_contrast_factor = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_CONTRAST_FACTOR, 0.0)
        hires_saturation_factor = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_SATURATION_FACTOR, 0.0)

        hires_mode = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_HIRES_MODE, UI.NONE)
        hires_mode_changed = access.setting_changed(UI.S_HIGH_RESOLUTION, UI.F_HIRES_MODE)

        hires_mode_enabled = hires_mode != UI.NONE

        # vae decoding

        parameters = [
            hires_mode,
        ]

        any_changes = (
                vae_changed or
                latent_changed
        )

        vae_decoded_changed = access.changed_in_cache(Names.C_VAE_DECODED_HIRES, parameters)
        if vae_decoded_changed or any_changes:
            if hires_mode_enabled:
                image = NodeWrapper.vae_decoder.decode(vae_model, latent)[0]
            else:
                image = None

            access.update_in_cache(Names.C_VAE_DECODED_HIRES, parameters, image)
            access.update_in_pipeline(Names.P_IMAGE, image)
        else:
            image = access.get_from_cache(Names.C_VAE_DECODED_HIRES)
            access.restore_in_pipeline(Names.P_IMAGE, image)

        # post processing

        image_changed = access.changed_in_pipeline(Names.P_IMAGE)
        image = access.get_from_pipeline(Names.P_IMAGE)

        parameters = [
            hires_mode_enabled,
            hires_contrast_factor,
            hires_saturation_factor,
        ]

        any_changes = (
                hires_mode_changed or
                image_changed
        )

        post_processed = None

        post_processed_changed = access.changed_in_cache(Names.C_POST_PROCESSED_HIRES, parameters)
        if post_processed_changed or any_changes:
            if hires_contrast_factor > 0.0:
                if post_processed is None:
                    post_processed = image

                if hires_mode_enabled and post_processed is not None:
                    post_processed = NodeWrapper.image_blend.blend_images(post_processed, post_processed,
                                                                          hires_contrast_factor * 0.5, "multiply")[0]
                else:
                    post_processed = None

            if hires_saturation_factor > 0.0:
                if post_processed is None:
                    post_processed = image

                if hires_mode_enabled and post_processed is not None:
                    post_processed = NodeWrapper.image_blend.blend_images(post_processed, post_processed,
                                                                          hires_saturation_factor * 0.5, "overlay")[0]
                else:
                    post_processed = None

            access.update_in_cache(Names.C_POST_PROCESSED_HIRES, parameters, post_processed)
            access.update_in_pipeline(Names.P_IMAGE, post_processed)
        else:
            post_processed = access.get_from_cache(Names.C_POST_PROCESSED_HIRES)
            access.restore_in_pipeline(Names.P_IMAGE, post_processed)

        if not hires_mode_enabled:
            image = None
            post_processed = None

        vae_decoded = {
            Names.F_DECODED_HIRES_IMAGE: image,
            Names.F_HIRES_POST_PROCESSED: post_processed,
        }

        if data is not None:
            data[Names.S_VAE_DECODED_HIRES] = vae_decoded

        # special treatment for the stage output here to match the structure of other vae stage outputs

        stage_output = {
            Names.S_VAE_DECODED: {
                Names.F_DECODED_IMAGE: image,
                Names.F_POST_PROCESSED: post_processed,
            },
        }

        return (data, stage_output,)
