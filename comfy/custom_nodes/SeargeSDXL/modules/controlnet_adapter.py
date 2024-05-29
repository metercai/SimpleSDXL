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

from folder_paths import get_full_path

from .controlnet import canny
from .controlnet import leres
from .controlnet import hed

from .data_utils import retrieve_parameter
from .ui import UI


# ====================================================================================================
# Adapter for controlnet/revision inputs
# ====================================================================================================

class SeargeControlnetAdapterV2:
    def __init__(self):
        self.expected_size = None

        self.hed_annotator = "ControlNetHED.pth"
        self.leres_annotator = "res101.pth"

        self.hed_annotator_full_path = get_full_path("annotators", self.hed_annotator)
        self.leres_annotator_full_path = get_full_path("annotators", self.leres_annotator)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "controlnet_mode": (UI.CONTROLNET_MODES, {"default": UI.NONE},),
                "controlnet_preprocessor": ("BOOLEAN", {"default": False},),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05},),
                "low_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05},),
                "high_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "noise_augmentation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},),
                "revision_enhancer": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
                "source_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM", "IMAGE",)
    RETURN_NAMES = ("data", "preview",)
    FUNCTION = "get_value"

    CATEGORY = UI.CATEGORY_UI_PROMPTING

    def process_image(self, image, mode, low_threshold, high_threshold):
        if mode == UI.CN_MODE_CANNY:
            image = canny(image, low_threshold, high_threshold)

        elif mode == UI.CN_MODE_DEPTH:
            image = leres(image, low_threshold, high_threshold, self.leres_annotator_full_path)

        elif mode == UI.CN_MODE_SKETCH:
            image = hed(image, self.hed_annotator_full_path)

        else:
            # do nothing for any other mode, just use the provided image unchanged
            pass

        return image

    def create_dict(self, stack, source_image, controlnet_mode, controlnet_preprocessor, strength,
                    low_threshold, high_threshold, start, end, noise_augmentation, revision_enhancer):
        if controlnet_mode is None or controlnet_mode == UI.NONE:
            cn_image = None
        else:
            cn_image = source_image

        low_threshold = round(low_threshold, 3)
        high_threshold = round(high_threshold, 3)

        # NOTE: for the modes "revision" and "custom" no image pre-processing is needed
        if controlnet_mode == UI.CN_MODE_REVISION or controlnet_mode == UI.CUSTOM:
            controlnet_preprocessor = False

        if controlnet_preprocessor and cn_image is not None:
            cn_image = self.process_image(cn_image, controlnet_mode, low_threshold, high_threshold)

        stack += [
            {
                UI.F_REV_CN_IMAGE: cn_image,
                UI.F_REV_CN_IMAGE_CHANGED: True,
                UI.F_REV_CN_MODE: controlnet_mode,
                UI.F_CN_PRE_PROCESSOR: controlnet_preprocessor,
                UI.F_REV_CN_STRENGTH: round(strength, 3),
                UI.F_CN_LOW_THRESHOLD: low_threshold,
                UI.F_CN_HIGH_THRESHOLD: high_threshold,
                UI.F_CN_START: round(start, 3),
                UI.F_CN_END: round(end, 3),
                UI.F_REV_NOISE_AUGMENTATION: round(noise_augmentation, 3),
                UI.F_REV_ENHANCER: revision_enhancer,
            }
        ]

        return (
            {
                UI.F_CN_STACK: stack,
            },
            cn_image,
        )

    def get_value(self, controlnet_mode, controlnet_preprocessor, strength, low_threshold, high_threshold,
                  start_percent, end_percent, noise_augmentation, revision_enhancer, source_image=None, data=None):
        if data is None:
            data = {}

        stack = retrieve_parameter(UI.F_CN_STACK, retrieve_parameter(UI.S_CONTROLNET_INPUTS, data), [])

        if self.expected_size is None:
            self.expected_size = len(stack)
        elif self.expected_size == 0:
            stack = []
        elif len(stack) > self.expected_size:
            stack = stack[:self.expected_size]

        (stack_entry, image) = self.create_dict(
            stack,
            source_image,
            controlnet_mode,
            controlnet_preprocessor,
            strength,
            low_threshold,
            high_threshold,
            start_percent,
            end_percent,
            noise_augmentation,
            revision_enhancer,
        )

        data[UI.S_CONTROLNET_INPUTS] = stack_entry

        return (data, image,)
