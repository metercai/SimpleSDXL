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
from .utils import next_multiple_of


# --------------------------------------------------------------------------------
# Stage: Clip Conditioning
# --------------------------------------------------------------------------------

class SeargeStageClipConditioning:
    PROMPT_PLACEHOLDER = "<prompt>"
    CONDITIONING_ROUNDING = 16

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

        base_clip_changed = access.changed_in_pipeline(Names.P_BASE_CLIP)
        refiner_clip_changed = access.changed_in_pipeline(Names.P_REFINER_CLIP)

        base_clip = access.get_from_pipeline(Names.P_BASE_CLIP)
        refiner_clip = access.get_from_pipeline(Names.P_REFINER_CLIP)

        has_base_clip = base_clip is not None
        has_refiner_clip = refiner_clip is not None

        main_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_MAIN_PROMPT, "")
        secondary_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_SECONDARY_PROMPT, "")
        style_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_STYLE_PROMPT, "")
        neg_main_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_NEGATIVE_MAIN_PROMPT, "")
        neg_secondary_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_NEGATIVE_SECONDARY_PROMPT, "")
        neg_style_prompt = access.get_active_setting(UI.S_PROMPTS, UI.F_NEGATIVE_STYLE_PROMPT, "")

        prompting_mode = access.get_active_setting(UI.S_OPERATING_MODE, UI.F_PROMPTING_MODE, UI.PROMPTING_DEFAULT)

        image_width = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_WIDTH, 1024)
        image_height = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_IMAGE_HEIGHT, 1024)

        base_cond_scale = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_BASE_CONDITIONING_SCALE, 1)
        refiner_cond_scale = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_REFINER_CONDITIONING_SCALE, 1)
        target_cond_scale = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_TARGET_CONDITIONING_SCALE, 1)
        pos_cond_scale = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_POSITIVE_CONDITIONING_SCALE, 1.5)
        neg_cond_scale = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_NEGATIVE_CONDITIONING_SCALE, 0.75)
        pos_ascore = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_POSITIVE_AESTHETIC_SCORE, 6.0)
        neg_ascore = access.get_active_setting(UI.S_CONDITIONING_PARAMETERS, UI.F_NEGATIVE_AESTHETIC_SCORE, 2.5)

        prompts = [
            prompting_mode,
            main_prompt,
            secondary_prompt,
            style_prompt,
            neg_main_prompt,
            neg_secondary_prompt,
            neg_style_prompt,
        ]

        def pack_prompts(processed):
            (base_pos_main, base_pos_sec, base_pos_style, base_neg_main, base_neg_sec, base_neg_style,
             ref_pos, ref_pos_style, ref_neg, ref_neg_style) = processed

            return {
                Names.F_BASE_POSITIVE_MAIN_PROMPT: base_pos_main,
                Names.F_BASE_POSITIVE_SECONDARY_PROMPT: base_pos_sec,
                Names.F_BASE_POSITIVE_STYLE_PROMPT: base_pos_style,
                Names.F_BASE_NEGATIVE_MAIN_PROMPT: base_neg_main,
                Names.F_BASE_NEGATIVE_SECONDARY_PROMPT: base_neg_sec,
                Names.F_BASE_NEGATIVE_STYLE_PROMPT: base_neg_style,
                Names.F_REFINER_POSITIVE_PROMPT: ref_pos,
                Names.F_REFINER_POSITIVE_STYLE_PROMPT: ref_pos_style,
                Names.F_REFINER_NEGATIVE_PROMPT: ref_neg,
                Names.F_REFINER_NEGATIVE_STYLE_PROMPT: ref_neg_style,
            }

        prompts_changed = access.changed_in_cache(Names.C_PROCESSED_PROMPTS, prompts)
        if prompts_changed:
            empty = ""
            if prompting_mode == UI.PROMPTING_DEFAULT:
                processed_prompts = self.create_standard_prompts(
                    main_prompt, secondary_prompt, style_prompt,
                    neg_main_prompt, neg_secondary_prompt, neg_style_prompt)

            elif prompting_mode == UI.PROMPTING_MAIN_AND_NEGATIVE_ONLY:
                processed_prompts = self.create_standard_prompts(
                    main_prompt, main_prompt, empty,
                    neg_main_prompt, neg_main_prompt, empty)

            elif prompting_mode == UI.PROMPTING_MAIN_SECONDARY_AND_NEGATIVE:
                processed_prompts = self.create_standard_prompts(
                    main_prompt, secondary_prompt, empty,
                    neg_main_prompt, neg_main_prompt, empty)

            elif prompting_mode == UI.PROMPTING_MAIN_ALL_EXCEPT_SECONDARY:
                processed_prompts = self.create_standard_prompts(
                    main_prompt, main_prompt, style_prompt,
                    neg_main_prompt, neg_secondary_prompt, neg_style_prompt)

            else:
                processed_prompts = self.create_pass_through_prompts(
                    main_prompt, secondary_prompt, style_prompt,
                    neg_main_prompt, neg_secondary_prompt, neg_style_prompt)

            access.update_in_cache(Names.C_PROCESSED_PROMPTS, prompts, processed_prompts)
            access.update_in_pipeline(Names.P_PROCESSED_PROMPTS, pack_prompts(processed_prompts))
        else:
            processed_prompts = access.get_from_cache(Names.C_PROCESSED_PROMPTS)
            access.restore_in_pipeline(Names.P_PROCESSED_PROMPTS, pack_prompts(processed_prompts))

        (base_pos_main, base_pos_sec, base_pos_style, base_neg_main, base_neg_sec, base_neg_style,
         ref_pos, ref_pos_style, ref_neg, ref_neg_style) = processed_prompts

        base_prompts = [
            base_cond_scale,
            target_cond_scale,
            pos_cond_scale,
            neg_cond_scale,
            base_pos_main,
            base_pos_sec,
            base_pos_style,
            base_neg_main,
            base_neg_sec,
            base_neg_style,
        ]

        refiner_prompts = [
            refiner_cond_scale,
            pos_cond_scale,
            neg_cond_scale,
            pos_ascore,
            neg_ascore,
            ref_pos,
            ref_pos_style,
            ref_neg,
            ref_neg_style,
        ]

        def pack_base_cond(encoded):
            (pos, pos_style, neg, neg_style) = encoded

            return {
                Names.F_BASE_POSITIVE: pos,
                Names.F_BASE_POSITIVE_STYLE: pos_style,
                Names.F_BASE_NEGATIVE: neg,
                Names.F_BASE_NEGATIVE_STYLE: neg_style,
            }

        def pack_ref_cond(encoded):
            (pos, pos_style, neg, neg_style) = encoded

            return {
                Names.F_REFINER_POSITIVE: pos,
                Names.F_REFINER_POSITIVE_STYLE: pos_style,
                Names.F_REFINER_NEGATIVE: neg,
                Names.F_REFINER_NEGATIVE_STYLE: neg_style,
            }

        if has_base_clip:
            base_cond_changed = access.changed_in_cache(Names.C_BASE_CONDITIONING, base_prompts)
            if base_cond_changed or base_clip_changed:
                encoded_base = self.encode_base(base_clip, processed_prompts, image_width, image_height,
                                                base_cond_scale, target_cond_scale, pos_cond_scale, neg_cond_scale)

                access.update_in_cache(Names.C_BASE_CONDITIONING, base_prompts, encoded_base)
                access.update_in_pipeline(Names.P_BASE_CONDITIONING, pack_base_cond(encoded_base))
            else:
                encoded_base = access.get_from_cache(Names.C_BASE_CONDITIONING)
                access.restore_in_pipeline(Names.P_BASE_CONDITIONING, pack_base_cond(encoded_base))

        else:
            encoded_base = (None, None, None, None,)
            if base_clip_changed:
                access.update_in_pipeline(Names.P_BASE_CONDITIONING, pack_ref_cond(encoded_base))
            else:
                access.restore_in_pipeline(Names.P_BASE_CONDITIONING, pack_ref_cond(encoded_base))

        if has_refiner_clip:
            ref_cond_changed = access.changed_in_cache(Names.C_REFINER_CONDITIONING, refiner_prompts)
            if ref_cond_changed or refiner_clip_changed:
                encoded_ref = self.encode_ref(refiner_clip, processed_prompts, image_width, image_height,
                                              refiner_cond_scale, pos_cond_scale, neg_cond_scale,
                                              pos_ascore, neg_ascore)

                access.update_in_cache(Names.C_REFINER_CONDITIONING, refiner_prompts, encoded_ref)
                access.update_in_pipeline(Names.P_REFINER_CONDITIONING, pack_ref_cond(encoded_ref))
            else:
                encoded_ref = access.get_from_cache(Names.C_REFINER_CONDITIONING)
                access.restore_in_pipeline(Names.P_REFINER_CONDITIONING, pack_ref_cond(encoded_ref))
        else:
            encoded_ref = (None, None, None, None,)
            if refiner_clip_changed:
                access.update_in_pipeline(Names.P_REFINER_CONDITIONING, pack_ref_cond(encoded_ref))
            else:
                access.restore_in_pipeline(Names.P_REFINER_CONDITIONING, pack_ref_cond(encoded_ref))

        (base_positive, base_positive_style, base_negative, base_negative_style) = encoded_base
        (refiner_positive, refiner_positive_style, refiner_negative, refiner_negative_style) = encoded_ref

        processed_prompts = pack_prompts(processed_prompts)

        conditioning = {
            Names.F_BASE_POSITIVE: base_positive,
            Names.F_BASE_POSITIVE_STYLE: base_positive_style,
            Names.F_BASE_NEGATIVE: base_negative,
            Names.F_BASE_NEGATIVE_STYLE: base_negative_style,
            Names.F_REFINER_POSITIVE: refiner_positive,
            Names.F_REFINER_POSITIVE_STYLE: refiner_positive_style,
            Names.F_REFINER_NEGATIVE: refiner_negative,
            Names.F_REFINER_NEGATIVE_STYLE: refiner_negative_style,
        }

        if data is not None:
            data[Names.S_CONDITIONING] = conditioning
            data[Names.S_PROCESSED_PROMPTS] = processed_prompts

        stage_output = {
            Names.S_CONDITIONING: conditioning,
            Names.S_PROCESSED_PROMPTS: processed_prompts,
        }

        return (data, stage_output,)

    def create_standard_prompts(self, main, secondary, pos_style, neg_main, neg_secondary, neg_style):
        main = "" if main is None else main
        secondary = main if secondary is None else secondary
        neg_main = "" if neg_main is None else neg_main
        neg_secondary = neg_main if neg_secondary is None else neg_secondary

        if pos_style is not None and len(pos_style) > 0:
            if pos_style.find(self.PROMPT_PLACEHOLDER) >= 0:
                base_pos_main = pos_style.replace(self.PROMPT_PLACEHOLDER, main)
                base_pos_sec = pos_style.replace(self.PROMPT_PLACEHOLDER, secondary)
            else:
                if len(main) > 0:
                    base_pos_main = main + ". " + pos_style
                else:
                    base_pos_main = pos_style

                if len(secondary) > 0:
                    base_pos_sec = secondary + ". " + pos_style
                else:
                    base_pos_sec = pos_style
        else:
            base_pos_main = main
            base_pos_sec = secondary

        base_neg_main = neg_main
        base_neg_sec = neg_secondary

        if pos_style is not None and len(pos_style) > 0:
            if pos_style.find(self.PROMPT_PLACEHOLDER) >= 0:
                ref_pos = pos_style.replace(self.PROMPT_PLACEHOLDER, main)
            else:
                if len(main) > 0:
                    ref_pos = main + ". " + pos_style
                else:
                    ref_pos = pos_style
        else:
            ref_pos = main

        if len(neg_main) > 0 and len(neg_secondary) > 0:
            ref_neg = neg_main + ". " + neg_secondary
        elif len(neg_main) > 0:
            ref_neg = neg_main
        elif len(neg_secondary) > 0:
            ref_neg = neg_secondary
        else:
            ref_neg = ""

        base_pos_style = pos_style.replace(self.PROMPT_PLACEHOLDER, "")
        base_neg_style = neg_style.replace(self.PROMPT_PLACEHOLDER, "")
        ref_pos_style = base_pos_style
        ref_neg_style = base_neg_style

        return (base_pos_main, base_pos_sec, base_pos_style, base_neg_main, base_neg_sec, base_neg_style,
                ref_pos, ref_pos_style, ref_neg, ref_neg_style)

    def create_pass_through_prompts(self, main, secondary, style_prompt, neg_main, neg_secondary, neg_style):
        base_pos_main = main
        base_pos_sec = secondary
        base_pos_style = style_prompt

        base_neg_main = neg_main
        base_neg_sec = neg_secondary
        base_neg_style = neg_style

        if len(main) > 0 and len(secondary) > 0:
            ref_pos = main + ". " + secondary
        elif len(main) > 0:
            ref_pos = main
        elif len(secondary) > 0:
            ref_pos = secondary
        else:
            ref_pos = ""

        ref_pos_style = style_prompt

        if len(neg_main) > 0 and len(neg_secondary) > 0:
            ref_neg = neg_main + ". " + neg_secondary
        elif len(neg_main) > 0:
            ref_neg = neg_main
        elif len(neg_secondary) > 0:
            ref_neg = neg_secondary
        else:
            ref_neg = ""

        ref_neg_style = neg_style

        return (base_pos_main, base_pos_sec, base_pos_style, base_neg_main, base_neg_sec, base_neg_style,
                ref_pos, ref_pos_style, ref_neg, ref_neg_style)

    def encode_base(self, base_clip, std_prompts, image_width, image_height, cond_scale=1.0, target_scale=1.0,
                    pos_scale=1.0, neg_scale=1.0):
        encoder = NodeWrapper.sdxl_clip_base_encoder

        (pos_main, pos_sec, pos_style, neg_main, neg_sec, neg_style, _, _, _, _) = std_prompts

        base_width = next_multiple_of(image_width * cond_scale, self.CONDITIONING_ROUNDING)
        base_height = next_multiple_of(image_height * cond_scale, self.CONDITIONING_ROUNDING)
        target_width = next_multiple_of(image_width * target_scale, self.CONDITIONING_ROUNDING)
        target_height = next_multiple_of(image_height * target_scale, self.CONDITIONING_ROUNDING)

        pos_width = next_multiple_of(base_width * pos_scale, self.CONDITIONING_ROUNDING)
        pos_height = next_multiple_of(base_height * pos_scale, self.CONDITIONING_ROUNDING)

        neg_width = next_multiple_of(base_width * neg_scale, self.CONDITIONING_ROUNDING)
        neg_height = next_multiple_of(base_height * neg_scale, self.CONDITIONING_ROUNDING)

        base_positive = encoder.encode(base_clip, pos_width, pos_height, 0, 0, target_width, target_height,
                                       pos_main, pos_sec)[0]
        base_positive_style = encoder.encode(base_clip, pos_width, pos_height, 0, 0, target_width, target_height,
                                             pos_style, pos_style)[0]

        base_negative = encoder.encode(base_clip, neg_width, neg_height, 0, 0, target_width, target_height,
                                       neg_main, neg_sec)[0]
        base_negative_style = encoder.encode(base_clip, neg_width, neg_height, 0, 0, target_width, target_height,
                                             neg_style, neg_style)[0]

        return (base_positive, base_positive_style, base_negative, base_negative_style)

    def encode_ref(self, refiner_clip, std_prompts, image_width, image_height, cond_scale=1.0,
                   pos_scale=1.0, neg_scale=1.0, pos_ascore=6.0, neg_ascore=2.5):
        encoder = NodeWrapper.sdxl_clip_refiner_encoder

        (_, _, _, _, _, _, pos, pos_style, neg, neg_style) = std_prompts

        refiner_width = next_multiple_of(image_width * cond_scale, self.CONDITIONING_ROUNDING)
        refiner_height = next_multiple_of(image_height * cond_scale, self.CONDITIONING_ROUNDING)

        pos_width = next_multiple_of(refiner_width * pos_scale, self.CONDITIONING_ROUNDING)
        pos_height = next_multiple_of(refiner_height * pos_scale, self.CONDITIONING_ROUNDING)

        neg_width = next_multiple_of(refiner_width * neg_scale, self.CONDITIONING_ROUNDING)
        neg_height = next_multiple_of(refiner_height * neg_scale, self.CONDITIONING_ROUNDING)

        refiner_positive = encoder.encode(refiner_clip, pos_ascore, pos_width, pos_height, pos)[0]
        refiner_positive_style = encoder.encode(refiner_clip, pos_ascore, pos_width, pos_height, pos_style)[0]

        refiner_negative = encoder.encode(refiner_clip, neg_ascore, neg_width, neg_height, neg)[0]
        refiner_negative_style = encoder.encode(refiner_clip, neg_ascore, neg_width, neg_height, neg_style)[0]

        return (refiner_positive, refiner_positive_style, refiner_negative, refiner_negative_style)
