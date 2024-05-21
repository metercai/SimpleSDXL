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
# Stage: Upscaling
# --------------------------------------------------------------------------------

class SeargeStageUpscaling:
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

        primary_upscaler = access.get_from_pipeline(Names.P_PRIMARY_UPSCALER)
        secondary_upscaler = access.get_from_pipeline(Names.P_SECONDARY_UPSCALER)

        primary_upscaler_changed = access.changed_in_pipeline(Names.P_PRIMARY_UPSCALER)
        secondary_upscaler_changed = access.changed_in_pipeline(Names.P_SECONDARY_UPSCALER)

        upscale_size = access.get_active_setting(UI.S_HIGH_RESOLUTION, UI.F_FINAL_UPSCALE_SIZE, UI.NONE)

        upscale_factor = 1.0
        if upscale_size == UI.UPSCALE_FACTOR_1_2:
            upscale_factor = 1.2
        if upscale_size == UI.UPSCALE_FACTOR_1_25:
            upscale_factor = 1.25
        if upscale_size == UI.UPSCALE_FACTOR_1_333:
            upscale_factor = 1.33333
        elif upscale_size == UI.UPSCALE_FACTOR_1_5:
            upscale_factor = 1.5
        elif upscale_size == UI.UPSCALE_FACTOR_2_0:
            upscale_factor = 2.0
        elif upscale_size == UI.UPSCALE_FACTOR_3_0:
            upscale_factor = 3.0
        elif upscale_size == UI.UPSCALE_FACTOR_4_0:
            upscale_factor = 4.0

        image = access.get_from_pipeline(Names.P_IMAGE)
        image_changed = access.changed_in_pipeline(Names.P_IMAGE)

        parameters = [
            upscale_size,
            upscale_factor,
        ]

        any_changes = (
                primary_upscaler_changed or
                secondary_upscaler_changed or
                image_changed
        )

        upscaled_image_changed = access.changed_in_cache(Names.C_UPSCALED_IMAGE, parameters)
        if upscaled_image_changed or any_changes:
            if image is not None:
                (image_width, image_height) = get_image_size(image)
            else:
                (image_width, image_height) = (1024, 1024)

            new_width = next_multiple_of(image_width * upscale_factor, self.SIZE_MULTIPLE_OF)
            new_height = next_multiple_of(image_height * upscale_factor, self.SIZE_MULTIPLE_OF)

            if upscale_factor > 1.0 and image is not None:
                if primary_upscaler is not None and secondary_upscaler is not None:
                    image1 = NodeWrapper.scale_with_model.upscale(primary_upscaler, image)[0]
                    (scaled_width1, scaled_height1) = get_image_size(image1)
                    if scaled_width1 != 4 * image_width or scaled_height1 != 4 * image_height:
                        print("Warning: primary upscaler should be a 4x ESRGAN model")

                    image2 = NodeWrapper.scale_with_model.upscale(secondary_upscaler, image)[0]
                    (scaled_width2, scaled_height2) = get_image_size(image2)
                    if scaled_width2 != 4 * image_width or scaled_height2 != 4 * image_height:
                        print("Warning: secondary upscaler should be a 4x ESRGAN model")

                    image = NodeWrapper.image_blend.blend_images(image1, image2, 0.2, "normal")[0]

                elif primary_upscaler is not None:
                    image = NodeWrapper.scale_with_model.upscale(primary_upscaler, image)[0]
                    (scaled_width, scaled_height) = get_image_size(image)
                    if scaled_width != 4 * image_width or scaled_height != 4 * image_height:
                        print("Warning: primary upscaler should be a 4x ESRGAN model")

                elif secondary_upscaler is not None:
                    image = NodeWrapper.scale_with_model.upscale(secondary_upscaler, image)[0]
                    (scaled_width, scaled_height) = get_image_size(image)
                    if scaled_width != 4 * image_width or scaled_height != 4 * image_height:
                        print("Warning: secondary upscaler should be a 4x ESRGAN model")

                else:
                    image = None

            else:
                image = None

            if image is not None:
                (scaled_width, scaled_height) = get_image_size(image)

                if scaled_width != new_width or scaled_height != new_height:
                    width_factor = float(scaled_width) / float(new_width)
                    height_factor = float(scaled_height) / float(new_height)

                    if width_factor >= 3.0 or height_factor >= 3.0:
                        step_width = next_multiple_of(new_width * 2.66666, self.SIZE_MULTIPLE_OF)
                        step_height = next_multiple_of(new_height * 2.66666, self.SIZE_MULTIPLE_OF)
                        image = NodeWrapper.image_scale.upscale(image, "bilinear", step_width, step_height, "center")[0]

                    if width_factor >= 2.5 or height_factor >= 2.5:
                        step_width = next_multiple_of(new_width * 2.0, self.SIZE_MULTIPLE_OF)
                        step_height = next_multiple_of(new_height * 2.0, self.SIZE_MULTIPLE_OF)
                        image = NodeWrapper.image_scale.upscale(image, "bilinear", step_width, step_height, "center")[0]

                    if width_factor >= 2.0 or height_factor >= 2.0:
                        step_width = next_multiple_of(new_width * 1.5, self.SIZE_MULTIPLE_OF)
                        step_height = next_multiple_of(new_height * 1.5, self.SIZE_MULTIPLE_OF)
                        image = NodeWrapper.image_scale.upscale(image, "bilinear", step_width, step_height, "center")[0]

                    if width_factor >= 1.5 or height_factor >= 1.5:
                        step_width = next_multiple_of(new_width * 1.33333, self.SIZE_MULTIPLE_OF)
                        step_height = next_multiple_of(new_height * 1.33333, self.SIZE_MULTIPLE_OF)
                        image = NodeWrapper.image_scale.upscale(image, "bilinear", step_width, step_height, "center")[0]

                    image = NodeWrapper.image_scale.upscale(image, "bicubic", new_width, new_height, "center")[0]

            access.update_in_cache(Names.C_UPSCALED_IMAGE, parameters, image)
            access.update_in_pipeline(Names.P_IMAGE, image)
        else:
            image = access.get_from_cache(Names.C_UPSCALED_IMAGE)
            access.restore_in_pipeline(Names.P_IMAGE, image)

        upscaled = {
            Names.F_UPSCALED_IMAGE: image,
        }

        if data is not None:
            data[Names.S_UPSCALED] = upscaled

        stage_output = {
            Names.S_UPSCALED: upscaled,
        }

        return (data, stage_output,)
