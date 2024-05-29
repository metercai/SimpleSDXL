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

from .mb_pipeline import Pipeline
from .mb_pipeline import PipelineAccess
from .names import Names
from .ui import Defs
from .ui import UI


# ====================================================================================================
# Magic Box Pipeline Start
# ====================================================================================================

class SeargePipelineStart:
    def __init__(self):
        self.pipeline = Pipeline()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wf_version": (Defs.WORKFLOW_VERSIONS,),
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
                "additional_data": ("SRG_DATA_STREAM",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM",)
    RETURN_NAMES = ("data",)
    FUNCTION = "trigger"

    OUTPUT_NODE = True

    CATEGORY = UI.CATEGORY_MAGIC

    def trigger(self, wf_version, data=None, additional_data=None, prompt=None, extra_pnginfo=None):
        if data is None:
            print("Warning: Pipeline Start - missing data stream")
        else:
            if additional_data is not None:
                data = data | additional_data

            self.pipeline.start(data)

            access = PipelineAccess(data)

            self.pipeline.enable(access.get_active_setting(UI.S_OPERATING_MODE, UI.F_WORKFLOW_MODE) != UI.NONE)

            mb_hidden = {
                Names.F_MAGIC_BOX_PROMPT: prompt,
                Names.F_MAGIC_BOX_EXTRA_PNGINFO: extra_pnginfo,
            }

            mb_version = {
                Names.F_MAGIC_BOX_EXTENSION: Defs.VERSION,
                Names.F_MAGIC_BOX_WORKFLOW: wf_version,
            }

            access.update_in_pipeline(Names.S_MAGIC_BOX_HIDDEN, mb_hidden)
            access.update_in_pipeline(Names.S_MAGIC_BOX_VERSION, mb_version)

            if data is not None:
                data[Names.S_MAGIC_BOX_HIDDEN] = mb_hidden
                data[Names.S_MAGIC_BOX_VERSION] = mb_version

        return (data,)
