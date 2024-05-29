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
# Stage: Apply Controlnet
# --------------------------------------------------------------------------------

class SeargeStageApplyControlnet:
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

        stack = access.get_active_setting(UI.S_CONTROLNET_INPUTS, UI.F_CN_STACK, [])

        base_cond = access.get_from_pipeline(Names.P_BASE_CONDITIONING)
        base_cond_changed = access.changed_in_pipeline(Names.P_BASE_CONDITIONING)

        if base_cond is not None:
            base_cond = base_cond.copy()

        clip_vision_changed = access.changed_in_pipeline(Names.P_CLIP_VISION_MODEL)
        canny_changed = access.changed_in_pipeline(Names.P_CN_CANNY_MODEL)
        depth_changed = access.changed_in_pipeline(Names.P_CN_DEPTH_MODEL)
        recolor_changed = access.changed_in_pipeline(Names.P_CN_RECOLOR_MODEL)
        sketch_changed = access.changed_in_pipeline(Names.P_CN_SKETCH_MODEL)
        custom_changed = access.changed_in_pipeline(Names.P_CN_CUSTOM_MODEL)

        ignore_positive = access.get_active_setting(Names.S_CONDITION_ZEROING, Names.F_ZERO_POSITIVES, False)
        ignore_negative = access.get_active_setting(Names.S_CONDITION_ZEROING, Names.F_ZERO_NEGATIVES, False)

        (cn_stack, images_changed) = self.comparable_stack(stack)

        parameters = [
            cn_stack,
            ignore_positive,
            ignore_negative,
        ]

        any_changes = (
                images_changed or
                base_cond_changed or
                clip_vision_changed or
                canny_changed or
                depth_changed or
                recolor_changed or
                sketch_changed or
                custom_changed
        )

        applied_controlnet_changed = access.changed_in_cache(Names.C_APPLIED_CONTROLNET, parameters)
        if any_changes or applied_controlnet_changed:

            base_positive = retrieve_parameter(Names.F_BASE_POSITIVE, base_cond)
            base_negative = retrieve_parameter(Names.F_BASE_NEGATIVE, base_cond)

            if ignore_positive:
                base_positive = NodeWrapper.zero_out_cond.zero_out(base_positive)[0]

            if ignore_negative:
                base_negative = NodeWrapper.zero_out_cond.zero_out(base_negative)[0]

            base_cond[Names.F_BASE_POSITIVE] = base_positive
            base_cond[Names.F_BASE_NEGATIVE] = base_negative

            (base_positive, base_negative, changed_cond) = self.apply_controlnet(access, stack, base_cond)

            access.update_in_cache(Names.C_APPLIED_CONTROLNET, parameters, (base_positive, base_negative,
                                                                            base_cond, changed_cond))

            access.update_in_pipeline(Names.P_BASE_CONDITIONING, base_cond)

        else:
            (base_positive, base_negative, base_cond, changed_cond) = access.get_from_cache(Names.C_APPLIED_CONTROLNET)
            if changed_cond:
                access.restore_in_pipeline(Names.P_BASE_CONDITIONING, base_cond)

        controlnet_output = {
            Names.F_CN_BASE_POSITIVE: base_positive,
            Names.F_CN_BASE_NEGATIVE: base_negative,
        }

        if data is not None:
            data[Names.S_CONTROLNET_OUTPUT] = controlnet_output

        stage_output = {
            Names.S_CONTROLNET_OUTPUT: controlnet_output,
        }

        return (data, stage_output,)

    def comparable_stack(self, stack):
        new_stack = []

        images_changed = False

        for controlnet in stack:
            entry = controlnet.copy()

            if UI.F_REV_CN_IMAGE in entry:
                entry.pop(UI.F_REV_CN_IMAGE)

            if UI.F_REV_CN_IMAGE_CHANGED in entry:
                if entry[UI.F_REV_CN_IMAGE_CHANGED]:
                    images_changed = True
                    controlnet[UI.F_REV_CN_IMAGE_CHANGED] = False

                entry.pop(UI.F_REV_CN_IMAGE_CHANGED)

            new_stack.append(entry)

        return (new_stack, images_changed,)

    def apply_controlnet(self, access, stack, base_cond):
        base_positive = retrieve_parameter(Names.F_BASE_POSITIVE, base_cond)
        base_negative = retrieve_parameter(Names.F_BASE_NEGATIVE, base_cond)

        changed_cond = False
        for controlnet in stack:
            mode = retrieve_parameter(UI.F_REV_CN_MODE, controlnet, UI.NONE)
            strength = retrieve_parameter(UI.F_REV_CN_STRENGTH, controlnet, 0.0)
            cn_image = retrieve_parameter(UI.F_REV_CN_IMAGE, controlnet)

            base_positive = retrieve_parameter(Names.F_BASE_POSITIVE, base_cond)
            base_negative = retrieve_parameter(Names.F_BASE_NEGATIVE, base_cond)

            controlnet_model = None
            if mode == UI.NONE:
                continue

            elif mode == UI.CN_MODE_REVISION:
                clipvision_model = access.get_from_pipeline(Names.P_CLIP_VISION_MODEL)
                if clipvision_model is not None and cn_image is not None:
                    clip_vision = NodeWrapper.clipvision_encoder.encode(clipvision_model, cn_image)[0]
                else:
                    clip_vision = None

                if clip_vision is not None and base_positive is not None and strength != 0.0:
                    noise_aug = retrieve_parameter(UI.F_REV_NOISE_AUGMENTATION, controlnet, 0.0)
                    enhancer = retrieve_parameter(UI.F_REV_ENHANCER, controlnet, False)

                    base_positive = NodeWrapper.unclip_conditioning.apply_adm(base_positive, clip_vision,
                                                                              strength, noise_aug)[0]
                    base_cond[Names.F_BASE_POSITIVE] = base_positive

                    if base_negative is not None and strength > 0.0 and enhancer:
                        base_negative = NodeWrapper.unclip_conditioning.apply_adm(base_negative, clip_vision,
                                                                                  -strength, noise_aug)[0]
                        base_cond[Names.F_BASE_NEGATIVE] = base_negative

                    changed_cond = True

            elif mode == UI.CN_MODE_CANNY:
                controlnet_model = access.get_from_pipeline(Names.P_CN_CANNY_MODEL)

            elif mode == UI.CN_MODE_DEPTH:
                controlnet_model = access.get_from_pipeline(Names.P_CN_DEPTH_MODEL)

            elif mode == UI.CN_MODE_RECOLOR:
                controlnet_model = access.get_from_pipeline(Names.P_CN_RECOLOR_MODEL)

            elif mode == UI.CN_MODE_SKETCH:
                controlnet_model = access.get_from_pipeline(Names.P_CN_SKETCH_MODEL)

            elif mode == UI.CUSTOM:
                controlnet_model = access.get_from_pipeline(Names.P_CN_CUSTOM_MODEL)

            if controlnet_model is None:
                continue

            if cn_image is not None and base_positive is not None and base_negative is not None:
                start = retrieve_parameter(UI.F_CN_START, controlnet, 0.0)
                end = retrieve_parameter(UI.F_CN_END, controlnet, 1.0)

                result = NodeWrapper.controlnet_advanced.apply_controlnet(base_positive, base_negative,
                                                                          controlnet_model, cn_image, strength,
                                                                          start, end)
                base_positive = result[0]
                base_negative = result[1]

                base_cond[Names.F_BASE_POSITIVE] = base_positive
                base_cond[Names.F_BASE_NEGATIVE] = base_negative

                changed_cond = True

        return (base_positive, base_negative, changed_cond)
