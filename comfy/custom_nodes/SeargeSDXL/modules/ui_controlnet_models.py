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
# UI: Upscale Models Input
# ====================================================================================================

class SeargeControlnetModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": (UI.CLIP_VISION_WITH_NONE(),),
                "canny_checkpoint": (UI.CONTROLNETS_WITH_NONE(),),
                "depth_checkpoint": (UI.CONTROLNETS_WITH_NONE(),),
                "recolor_checkpoint": (UI.CONTROLNETS_WITH_NONE(),),
                "sketch_checkpoint": (UI.CONTROLNETS_WITH_NONE(),),
                "custom_checkpoint": (UI.CONTROLNETS_WITH_NONE(),),
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
    def create_dict(clip_vision, canny_checkpoint, depth_checkpoint, recolor_checkpoint, sketch_checkpoint,
                    custom_checkpoint):
        return {
            UI.F_CLIP_VISION_CHECKPOINT: clip_vision,
            UI.F_CANNY_CHECKPOINT: canny_checkpoint,
            UI.F_DEPTH_CHECKPOINT: depth_checkpoint,
            UI.F_RECOLOR_CHECKPOINT: recolor_checkpoint,
            UI.F_SKETCH_CHECKPOINT: sketch_checkpoint,
            UI.F_CUSTOM_CHECKPOINT: custom_checkpoint,
        }

    def get(self, clip_vision, canny_checkpoint, depth_checkpoint, recolor_checkpoint, sketch_checkpoint,
            custom_checkpoint, data=None):
        if data is None:
            data = {}

        data[UI.S_CONTROLNET_MODELS] = self.create_dict(
            clip_vision,
            canny_checkpoint,
            depth_checkpoint,
            recolor_checkpoint,
            sketch_checkpoint,
            custom_checkpoint,
        )

        return (data,)
