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

from .ui import UI


# ====================================================================================================
# Adapter for prompt text inputs
# ====================================================================================================

class SeargePromptAdapterV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
                "main_prompt": ("SRG_PROMPT_TEXT",),
                "secondary_prompt": ("SRG_PROMPT_TEXT",),
                "style_prompt": ("SRG_PROMPT_TEXT",),
                "negative_main_prompt": ("SRG_PROMPT_TEXT",),
                "negative_secondary_prompt": ("SRG_PROMPT_TEXT",),
                "negative_style_prompt": ("SRG_PROMPT_TEXT",),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM", "SRG_DATA_STREAM",)
    RETURN_NAMES = ("data", UI.S_PROMPTS,)
    FUNCTION = "get_value"

    CATEGORY = UI.CATEGORY_UI_PROMPTING

    @staticmethod
    def create_dict(main_prompt=None, secondary_prompt=None, style_prompt=None,
                    negative_main_prompt=None, negative_secondary_prompt=None, negative_style_prompt=None):
        return {
            UI.F_MAIN_PROMPT: main_prompt,
            UI.F_SECONDARY_PROMPT: secondary_prompt,
            UI.F_STYLE_PROMPT: style_prompt,
            UI.F_NEGATIVE_MAIN_PROMPT: negative_main_prompt,
            UI.F_NEGATIVE_SECONDARY_PROMPT: negative_secondary_prompt,
            UI.F_NEGATIVE_STYLE_PROMPT: negative_style_prompt,
        }

    def get_value(self, main_prompt=None, secondary_prompt=None, style_prompt=None,
                  negative_main_prompt=None, negative_secondary_prompt=None, negative_style_prompt=None, data=None):
        if data is None:
            data = {}

        data[UI.S_PROMPTS] = self.create_dict(
            main_prompt,
            secondary_prompt,
            style_prompt,
            negative_main_prompt,
            negative_secondary_prompt,
            negative_style_prompt
        )

        return (data, data[UI.S_PROMPTS],)
