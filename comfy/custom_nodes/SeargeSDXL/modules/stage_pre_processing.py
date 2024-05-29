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

from .data_utils import retrieve_input
from .data_utils import retrieve_parameter
from .mb_pipeline import PipelineAccess
from .names import Names
from .node_wrapper import NodeWrapper
from .ui import UI
from .utils import get_image_size


# --------------------------------------------------------------------------------
# Stage: Pre Process Data
# --------------------------------------------------------------------------------

class SeargePreProcessData:
    def __init__(self):
        self.UI_OUTPUT_KEYS = None
        self.STAGE_OUTPUT_KEYS = None

    def get_input(self, data, stage_data):
        # if we still don't have stage data,
        if stage_data is None and data is not None:
            stage_data = {
                PipelineAccess.NAME: retrieve_parameter(PipelineAccess.NAME, data),
                UI.S_IMAGE_INPUTS: retrieve_parameter(UI.S_IMAGE_INPUTS, data),
            }

        return stage_data

    def process(self, data, stage_input):
        access = PipelineAccess(stage_input)

        denoise = access.get_active_setting(UI.S_IMG2IMG_INPAINTING, UI.F_DENOISE, 0.5)
        workflow_mode = access.get_active_setting(UI.S_OPERATING_MODE, UI.F_WORKFLOW_MODE, UI.WF_MODE_TEXT_TO_IMAGE)

        if denoise is not None and workflow_mode == UI.WF_MODE_TEXT_TO_IMAGE:
            denoise = 1.0
            access.override_setting(UI.S_IMG2IMG_INPAINTING, UI.F_DENOISE, denoise)

        size_preset = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_SIZE_PRESET, UI.USE_SETTINGS)

        need_image = (
                workflow_mode == UI.WF_MODE_IMAGE_TO_IMAGE or
                workflow_mode == UI.WF_MODE_IN_PAINTING or
                size_preset == UI.RESOLUTION_FROM_IMAGE
        )

        if need_image:
            image_inputs = retrieve_input(UI.S_IMAGE_INPUTS, data, stage_input)
            image_changed = retrieve_parameter(UI.F_SOURCE_IMAGE_CHANGED, image_inputs, False)
            image_inputs[UI.F_SOURCE_IMAGE_CHANGED] = False

            if image_changed:
                image = retrieve_parameter(UI.F_SOURCE_IMAGE, image_inputs)

                access.update_in_cache(Names.C_SOURCE_IMAGE, [], image)
                access.update_in_pipeline(Names.P_IMAGE, image)
            else:
                image = access.get_from_cache(Names.C_SOURCE_IMAGE)
                access.restore_in_pipeline(Names.P_IMAGE, image)

        if size_preset == UI.RESOLUTION_1024x1024:
            (image_width, image_height) = (1024, 1024)
        elif size_preset == UI.RESOLUTION_1152x896:
            (image_width, image_height) = (1152, 896)
        elif size_preset == UI.RESOLUTION_1216x832:
            (image_width, image_height) = (1216, 832)
        elif size_preset == UI.RESOLUTION_1344x768:
            (image_width, image_height) = (1344, 768)
        elif size_preset == UI.RESOLUTION_1536x640:
            (image_width, image_height) = (1536, 640)
        elif size_preset == UI.RESOLUTION_896x1152:
            (image_width, image_height) = (896, 1152)
        elif size_preset == UI.RESOLUTION_832x1216:
            (image_width, image_height) = (832, 1216)
        elif size_preset == UI.RESOLUTION_768x1344:
            (image_width, image_height) = (768, 1344)
        elif size_preset == UI.RESOLUTION_640x1536:
            (image_width, image_height) = (640, 1536)
        else:
            (image_width, image_height) = (
                access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_WIDTH, 1024),
                access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_HEIGHT, 1024),
            )

        image_changed = access.changed_in_pipeline(Names.P_IMAGE)
        changed_in_cache = access.changed_in_cache(Names.C_IMAGE_SIZE, [size_preset])

        any_changes = (
                image_changed or
                changed_in_cache
        )

        if size_preset == UI.RESOLUTION_FROM_IMAGE:
            if any_changes:
                image = access.get_from_pipeline(Names.P_IMAGE)

                if image is not None:
                    (image_width, image_height) = get_image_size(image)
                    access.update_in_cache(Names.C_IMAGE_SIZE, [size_preset], (image_width, image_height))

            elif access.has_in_cache(Names.C_IMAGE_SIZE):
                (image_width, image_height) = access.get_from_cache(Names.C_IMAGE_SIZE)

        access.override_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_WIDTH, image_width)
        access.override_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_HEIGHT, image_height)

        mask_mode = access.get_active_setting(UI.S_IMG2IMG_INPAINTING, UI.F_INPAINT_MASK_MODE)
        mask_mode_changed = access.setting_changed(UI.S_IMG2IMG_INPAINTING, UI.F_INPAINT_MASK_MODE)

        if workflow_mode == UI.WF_MODE_IN_PAINTING:
            image_inputs = retrieve_input(UI.S_IMAGE_INPUTS, data, stage_input)

            if mask_mode == UI.MASK_MODE_UPLOADED_FULL:
                mask_changed = retrieve_parameter(UI.F_UPLOADED_MASK_CHANGED, image_inputs, False)
                image_inputs[UI.F_UPLOADED_MASK_CHANGED] = False
            else:
                mask_changed = retrieve_parameter(UI.F_IMAGE_MASK_CHANGED, image_inputs, False)
                image_inputs[UI.F_IMAGE_MASK_CHANGED] = False

            if mask_changed or mask_mode_changed:
                if mask_mode == UI.MASK_MODE_UPLOADED_FULL:
                    mask = retrieve_parameter(UI.F_UPLOADED_MASK, image_inputs)
                else:
                    mask = retrieve_parameter(UI.F_IMAGE_MASK, image_inputs)

                access.update_in_cache(Names.C_SOURCE_MASK, [], mask)
                access.update_in_pipeline(Names.P_MASK, mask)

            else:
                mask = access.get_from_cache(Names.C_SOURCE_MASK)
                access.restore_in_pipeline(Names.P_MASK, mask)

            mask_blur = access.get_active_setting(UI.S_IMG2IMG_INPAINTING, UI.F_INPAINT_MASK_BLUR, 8)

            parameters = [
                mask_blur,
                mask_mode,
            ]

            mask_changed = access.changed_in_pipeline(Names.P_MASK)
            changed_in_cache = access.changed_in_cache(Names.C_BLURRY_MASK, parameters)

            any_changes = (
                    mask_changed or
                    changed_in_cache
            )

            if any_changes:
                mask = access.get_from_pipeline(Names.P_MASK)
                if mask is not None and mask_blur > 0 and changed_in_cache:
                    mask = NodeWrapper.mask_to_image.mask_to_image(mask)[0]
                    mask = NodeWrapper.image_blur.blur(mask, mask_blur, 3.0)[0]
                    mask = NodeWrapper.image_to_mask.image_to_mask(mask, "green")[0]

                access.update_in_cache(Names.C_BLURRY_MASK, parameters, mask)
                access.update_in_pipeline(Names.P_MASK, mask)
            elif access.has_in_cache(Names.C_BLURRY_MASK):
                mask = access.get_from_cache(Names.C_BLURRY_MASK)
                access.restore_in_pipeline(Names.P_MASK, mask)

        prompting_mode = access.get_active_setting(UI.S_OPERATING_MODE, UI.F_PROMPTING_MODE)

        main_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_MAIN_PROMPT)
        secondary_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_SECONDARY_PROMPT)
        style_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_STYLE_PROMPT)
        negative_main_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_NEGATIVE_MAIN_PROMPT)
        negative_secondary_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_NEGATIVE_SECONDARY_PROMPT)
        negative_style_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_NEGATIVE_STYLE_PROMPT)

        ignore_positive = False
        ignore_negative = False

        if prompting_mode == UI.PROMPTING_IGNORE_ALL:
            ignore_positive = True
            ignore_negative = True

        elif prompting_mode == UI.PROMPTING_IGNORE_EMPTY:
            ignore_positive = (
                    (main_prompt is None or main_prompt == "") and
                    (secondary_prompt is None or secondary_prompt == "") and
                    (style_prompt is None or style_prompt == "")
            )

            ignore_negative = (
                    (negative_main_prompt is None or negative_main_prompt == "") and
                    (negative_secondary_prompt is None or negative_secondary_prompt == "") and
                    (negative_style_prompt is None or negative_style_prompt == "")
            )

        access.override_setting(Names.S_CONDITION_ZEROING, Names.F_ZERO_POSITIVES, ignore_positive)
        access.override_setting(Names.S_CONDITION_ZEROING, Names.F_ZERO_NEGATIVES, ignore_negative)

        stage_results = {
        }

        stage_output = {
            Names.PLACEHOLDER: stage_results,
        }

        return (data, stage_output,)
