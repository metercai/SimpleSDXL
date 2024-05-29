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
# Stage: VAE Decode Sampled
# --------------------------------------------------------------------------------

class SeargeStageVAEDecodeSampled:
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

        contrast_factor = access.get_active_setting(UI.S_ADVANCED_PARAMETERS, UI.F_CONTRAST_FACTOR, 0.0)
        saturation_factor = access.get_active_setting(UI.S_ADVANCED_PARAMETERS, UI.F_SATURATION_FACTOR, 0.0)

        source_image_changed = access.changed_in_pipeline(Names.P_IMAGE)
        source_image = access.get_from_pipeline(Names.P_IMAGE)

        batch_size = access.get_active_setting(UI.S_OPERATING_MODE, UI.F_BATCH_SIZE, 1)

        any_changes = (
                vae_changed or
                latent_changed
        )

        vae_decoded_changed = access.changed_in_cache(Names.C_VAE_DECODED, [])
        if vae_decoded_changed or any_changes:
            image = NodeWrapper.vae_decoder.decode(vae_model, latent)[0]

            access.update_in_cache(Names.C_VAE_DECODED, [], image)
            access.update_in_pipeline(Names.P_IMAGE, image)
        else:
            image = access.get_from_cache(Names.C_VAE_DECODED)
            access.restore_in_pipeline(Names.P_IMAGE, image)

        image_changed = access.changed_in_pipeline(Names.P_IMAGE)
        mask_changed = access.changed_in_pipeline(Names.P_MASK)

        image = access.get_from_pipeline(Names.P_IMAGE)
        mask = access.get_from_pipeline(Names.P_MASK)

        parameters = [
            contrast_factor,
            saturation_factor,
            batch_size,
        ]

        any_changes = (
                source_image_changed or
                image_changed or
                mask_changed
        )

        post_processed = None

        post_processed_changed = access.changed_in_cache(Names.C_POST_PROCESSED, parameters)
        if post_processed_changed or any_changes:
            if contrast_factor > 0.0:
                if post_processed is None:
                    post_processed = image

                post_processed = NodeWrapper.image_blend.blend_images(post_processed, post_processed,
                                                                      contrast_factor * 0.5, "multiply")[0]

            if saturation_factor > 0.0:
                if post_processed is None:
                    post_processed = image

                post_processed = NodeWrapper.image_blend.blend_images(post_processed, post_processed,
                                                                      saturation_factor * 0.5, "overlay")[0]

            if "noise_mask" in latent:
                image = NodeWrapper.image_composite.composite(source_image, image, 0, 0, False, mask)[0]
                post_processed = NodeWrapper.image_composite.composite(source_image, post_processed, 0, 0,
                                                                       False, mask)[0]

            access.update_in_cache(Names.C_POST_PROCESSED, parameters, post_processed)
            access.update_in_pipeline(Names.P_IMAGE, post_processed)
        else:
            post_processed = access.get_from_cache(Names.C_POST_PROCESSED)
            access.restore_in_pipeline(Names.P_IMAGE, post_processed)

        vae_decoded = {
            Names.F_DECODED_SAMPLED_IMAGE: image,
            Names.F_SAMPLED_POST_PROCESSED: post_processed,
        }

        if data is not None:
            data[Names.S_VAE_DECODED_SAMPLED] = vae_decoded

        stage_output = {
            Names.S_VAE_DECODED: {
                Names.F_DECODED_IMAGE: image,
                Names.F_POST_PROCESSED: post_processed,
            },
        }

        return (data, stage_output,)
