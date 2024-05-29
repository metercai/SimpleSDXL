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
# UI: Loras Input
# ====================================================================================================

class SeargeLoras:
    def __init__(self):
        self.expected_lora_stack_size = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_1": (UI.LORAS_WITH_NONE(),),
                "lora_1_strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05},),
                "lora_2": (UI.LORAS_WITH_NONE(),),
                "lora_2_strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05},),
                "lora_3": (UI.LORAS_WITH_NONE(),),
                "lora_3_strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05},),
                "lora_4": (UI.LORAS_WITH_NONE(),),
                "lora_4_strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05},),
                "lora_5": (UI.LORAS_WITH_NONE(),),
                "lora_5_strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05},),
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM",)
    RETURN_NAMES = ("data",)
    FUNCTION = "get"

    CATEGORY = UI.CATEGORY_UI_INPUTS

    @staticmethod
    def create_dict(loras, lora_1, lora_1_strength, lora_2, lora_2_strength, lora_3, lora_3_strength,
                    lora_4, lora_4_strength, lora_5, lora_5_strength):
        loras += [
            {
                UI.F_LORA_NAME: lora_1,
                UI.F_LORA_STRENGTH: round(lora_1_strength, 3),
            },
            {
                UI.F_LORA_NAME: lora_2,
                UI.F_LORA_STRENGTH: round(lora_2_strength, 3),
            },
            {
                UI.F_LORA_NAME: lora_3,
                UI.F_LORA_STRENGTH: round(lora_3_strength, 3),
            },
            {
                UI.F_LORA_NAME: lora_4,
                UI.F_LORA_STRENGTH: round(lora_4_strength, 3),
            },
            {
                UI.F_LORA_NAME: lora_5,
                UI.F_LORA_STRENGTH: round(lora_5_strength, 3),
            },
        ]

        return {
            UI.F_LORA_STACK: loras,
        }

    def get(self, lora_1, lora_1_strength, lora_2, lora_2_strength, lora_3, lora_3_strength, lora_4, lora_4_strength,
            lora_5, lora_5_strength, data=None):
        if data is None:
            data = {}

        loras = retrieve_parameter(UI.F_LORA_STACK, retrieve_parameter(UI.S_LORAS, data), [])

        if self.expected_lora_stack_size is None:
            self.expected_lora_stack_size = len(loras)
        elif self.expected_lora_stack_size == 0:
            loras = []
        elif len(loras) > self.expected_lora_stack_size:
            loras = loras[:self.expected_lora_stack_size]

        data[UI.S_LORAS] = self.create_dict(
            loras,
            lora_1,
            lora_1_strength,
            lora_2,
            lora_2_strength,
            lora_3,
            lora_3_strength,
            lora_4,
            lora_4_strength,
            lora_5,
            lora_5_strength,
        )

        return (data,)
