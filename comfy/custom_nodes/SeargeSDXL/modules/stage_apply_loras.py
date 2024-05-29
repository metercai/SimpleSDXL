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

import folder_paths

from .data_utils import retrieve_parameter
from .mb_pipeline import PipelineAccess
from .names import Names
from .node_wrapper import NodeWrapper
from .ui import UI


# --------------------------------------------------------------------------------
# Stage: Apply Loras
# --------------------------------------------------------------------------------

class SeargeStageApplyLoras:
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

        base_model = access.get_from_pipeline(Names.P_BASE_MODEL)
        base_clip = access.get_from_pipeline(Names.P_BASE_CLIP)

        base_model_changed = access.changed_in_pipeline(Names.P_BASE_MODEL)
        base_clip_changed = access.changed_in_pipeline(Names.P_BASE_CLIP)

        lora_stack = access.get_active_setting(UI.S_LORAS, UI.F_LORA_STACK, [])

        any_changes = (
                base_model_changed or
                base_clip_changed
        )

        applied_loras = []

        loras_changed = access.changed_in_cache(Names.C_APPLIED_LORAS, lora_stack)
        if loras_changed or any_changes:
            for lora in lora_stack:
                lora_name = retrieve_parameter(UI.F_LORA_NAME, lora)
                lora_strength = retrieve_parameter(UI.F_LORA_STRENGTH, lora, 0.0)

                if folder_paths.get_full_path("loras", lora_name) is None or base_model is None or base_clip is None:
                    lora_name = None

                if lora_name is not None and lora_name != UI.NONE and lora_strength != 0.0:
                    (base_model, base_clip) = NodeWrapper.lora_loader.load_lora(base_model, base_clip, lora_name,
                                                                                lora_strength, lora_strength)
                    applied_loras.append(lora_name)

            access.update_in_cache(Names.C_APPLIED_LORAS, lora_stack, (base_model, base_clip))
            access.update_in_pipeline(Names.P_BASE_MODEL, base_model)
            access.update_in_pipeline(Names.P_BASE_CLIP, base_clip)
        else:
            (base_model, base_clip) = access.get_from_cache(Names.C_APPLIED_LORAS)
            access.restore_in_pipeline(Names.P_BASE_MODEL, base_model)
            access.restore_in_pipeline(Names.P_BASE_CLIP, base_clip)

        loaded_loras = {
            Names.F_LORA_NAMES: applied_loras,
        }

        if data is not None:
            data[Names.S_LOADED_LORAS] = loaded_loras

        stage_output = {
            Names.S_EXAMPLE_STRUCTURE: loaded_loras,
        }

        return (data, stage_output,)
