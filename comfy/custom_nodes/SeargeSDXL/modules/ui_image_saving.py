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
# UI: Image Saving Input
# ====================================================================================================

class SeargeImageSaving:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_parameters_file": ("BOOLEAN", {"default": True},),
                "save_folder": (UI.SAVE_FOLDERS, {"default": UI.SAVE_TO_OUTPUT_DATE, },),
                "save_generated_image": ("BOOLEAN", {"default": True},),
                "embed_workflow_in_generated": ("BOOLEAN", {"default": True},),
                "generated_image_name": ("STRING", {"multiline": False, "default": "generated", },),
                "save_high_res_image": ("BOOLEAN", {"default": True},),
                "embed_workflow_in_high_res": ("BOOLEAN", {"default": True},),
                "high_res_image_name": ("STRING", {"multiline": False, "default": "high-res", },),
                "save_upscaled_image": ("BOOLEAN", {"default": True},),
                "embed_workflow_in_upscaled": ("BOOLEAN", {"default": True},),
                "upscaled_image_name": ("STRING", {"multiline": False, "default": "upscaled", },),
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
    def create_dict(save_parameters_file, save_folder,
                    save_generated_image, embed_workflow_in_generated, generated_image_name,
                    save_high_res_image, embed_workflow_in_high_res, high_res_image_name,
                    save_upscaled_image, embed_workflow_in_upscaled, upscaled_image_name):
        return {
            UI.F_SAVE_PARAMETERS_FILE: save_parameters_file is not None and save_parameters_file,
            UI.F_SAVE_FOLDER: save_folder,
            UI.F_SAVE_GENERATED_IMAGE: save_generated_image is not None and save_generated_image,
            UI.F_EMBED_WORKFLOW_IN_GENERATED: embed_workflow_in_generated is not None and embed_workflow_in_generated,
            UI.F_GENERATED_IMAGE_NAME: generated_image_name,
            UI.F_SAVE_HIGH_RES_IMAGE: save_high_res_image is not None and save_high_res_image,
            UI.F_EMBED_WORKFLOW_IN_HIGH_RES: embed_workflow_in_high_res is not None and embed_workflow_in_high_res,
            UI.F_HIGH_RES_IMAGE_NAME: high_res_image_name,
            UI.F_SAVE_UPSCALED_IMAGE: save_upscaled_image is not None and save_upscaled_image,
            UI.F_EMBED_WORKFLOW_IN_UPSCALED: embed_workflow_in_upscaled is not None and embed_workflow_in_upscaled,
            UI.F_UPSCALED_IMAGE_NAME: upscaled_image_name,
        }

    def get(self, save_parameters_file, save_folder,
            save_generated_image, embed_workflow_in_generated, generated_image_name,
            save_high_res_image, embed_workflow_in_high_res, high_res_image_name,
            save_upscaled_image, embed_workflow_in_upscaled, upscaled_image_name, data=None):
        if data is None:
            data = {}

        data[UI.S_IMAGE_SAVING] = self.create_dict(
            save_parameters_file,
            save_folder,
            save_generated_image,
            embed_workflow_in_generated,
            generated_image_name,
            save_high_res_image,
            embed_workflow_in_high_res,
            high_res_image_name,
            save_upscaled_image,
            embed_workflow_in_upscaled,
            upscaled_image_name,
        )

        return (data,)
