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
from .ui import UI


# ====================================================================================================
# UI: Prompt Adapter Output
# ====================================================================================================

class SeargePromptAdapterV2Output:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
                "prompts": ("SRG_DATA_STREAM",),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM", "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("data", UI.F_MAIN_PROMPT, UI.F_SECONDARY_PROMPT, UI.F_STYLE_PROMPT,
                    UI.F_NEGATIVE_MAIN_PROMPT, UI.F_NEGATIVE_SECONDARY_PROMPT, UI.F_NEGATIVE_STYLE_PROMPT,)
    FUNCTION = "output"

    CATEGORY = UI.CATEGORY_UI_PROMPTING

    @staticmethod
    def get_data(data=None, prompts=None):
        if prompts is None:
            prompts = retrieve_parameter("prompts", data)

        if prompts is None:
            return (False, None,)

        return (True, {
            UI.F_MAIN_PROMPT: retrieve_parameter(UI.F_MAIN_PROMPT, prompts),
            UI.F_SECONDARY_PROMPT: retrieve_parameter(UI.F_SECONDARY_PROMPT, prompts),
            UI.F_STYLE_PROMPT: retrieve_parameter(UI.F_STYLE_PROMPT, prompts),
            UI.F_NEGATIVE_MAIN_PROMPT: retrieve_parameter(UI.F_NEGATIVE_MAIN_PROMPT, prompts),
            UI.F_NEGATIVE_SECONDARY_PROMPT: retrieve_parameter(UI.F_NEGATIVE_SECONDARY_PROMPT, prompts),
            UI.F_NEGATIVE_STYLE_PROMPT: retrieve_parameter(UI.F_NEGATIVE_STYLE_PROMPT, prompts),
        })

    def output(self, data=None, prompts=None):
        (has_data, output) = self.get_data(data, prompts)
        if not has_data:
            return (data, None, None, None, None, None, None,)

        return (data, output[UI.F_MAIN_PROMPT], output[UI.F_SECONDARY_PROMPT],
                output[UI.F_STYLE_PROMPT],
                output[UI.F_NEGATIVE_MAIN_PROMPT], output[UI.F_NEGATIVE_SECONDARY_PROMPT],
                output[UI.F_NEGATIVE_STYLE_PROMPT],)
