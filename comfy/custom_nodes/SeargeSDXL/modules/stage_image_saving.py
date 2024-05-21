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

import json
import numpy as np
import os

from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

from .data_utils import retrieve_parameter
from .mb_pipeline import PipelineAccess
from .names import Names
from .ui import UI


# --------------------------------------------------------------------------------
# Stage: Image Saving
# --------------------------------------------------------------------------------

class SeargeStageImageSaving:
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

        save_parameters_file = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_SAVE_PARAMETERS_FILE, False)
        save_folder = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_SAVE_FOLDER, UI.SAVE_TO_OUTPUT_DATE)

        save_generated_image = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_SAVE_GENERATED_IMAGE, True)
        embed_wf_in_generated = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_EMBED_WORKFLOW_IN_GENERATED, True)
        generated_image_name = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_GENERATED_IMAGE_NAME, "generated")

        save_high_res_image = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_SAVE_HIGH_RES_IMAGE, True)
        embed_wf_in_high_res = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_EMBED_WORKFLOW_IN_HIGH_RES, True)
        high_res_image_name = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_HIGH_RES_IMAGE_NAME, "hires")

        save_upscaled_image = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_SAVE_UPSCALED_IMAGE, True)
        embed_wf_in_upscaled = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_EMBED_WORKFLOW_IN_UPSCALED, True)
        upscaled_image_name = access.get_active_setting(UI.S_IMAGE_SAVING, UI.F_UPSCALED_IMAGE_NAME, "upscaled")

        magic_box_hidden = access.get_from_pipeline(Names.S_MAGIC_BOX_HIDDEN)
        hidden_prompt = retrieve_parameter(Names.F_MAGIC_BOX_PROMPT, magic_box_hidden)
        hidden_extra_pnginfo = retrieve_parameter(Names.F_MAGIC_BOX_EXTRA_PNGINFO, magic_box_hidden)

        # get the images from the data stream instead of the pipeline, this is intentional
        vae_decoded_sampled = retrieve_parameter(Names.S_VAE_DECODED_SAMPLED, data)
        generated_images = retrieve_parameter(Names.F_DECODED_SAMPLED_IMAGE, vae_decoded_sampled)
        post_processed_images = retrieve_parameter(Names.F_SAMPLED_POST_PROCESSED, vae_decoded_sampled)

        vae_decoded_hires = retrieve_parameter(Names.S_VAE_DECODED_HIRES, data)
        high_res_images = retrieve_parameter(Names.F_DECODED_HIRES_IMAGE, vae_decoded_hires)
        post_processed_hires = retrieve_parameter(Names.F_HIRES_POST_PROCESSED, vae_decoded_hires)

        upscaling_output = retrieve_parameter(Names.S_UPSCALED, data)
        upscaled_images = retrieve_parameter(Names.F_UPSCALED_IMAGE, upscaling_output)

        seed = access.get_active_setting(UI.S_GENERATION_PARAMETERS, UI.F_SEED)

        save_to_input = save_folder == UI.SAVE_TO_INPUT
        output_folder = folder_paths.get_input_directory() if save_to_input else folder_paths.get_output_directory()

        if save_folder == UI.SAVE_TO_OUTPUT:
            sub_folder = ""
        elif save_folder == UI.SAVE_TO_OUTPUT_DATE:
            sub_folder = "%date%"
        elif save_folder == UI.SAVE_TO_OUTPUT_SEARGE_SDXL_DATE:
            sub_folder = "Searge-SDXL-%date%"
        elif save_folder == UI.SAVE_TO_INPUT:
            sub_folder = ""
        else:
            return (data, None,)

        sub_folder = sub_folder.replace("%date%", datetime.now().strftime("%Y-%m-%d"))
        full_path = os.path.join(output_folder, sub_folder)

        try:
            files = [fn for fn in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, fn))]
        except FileNotFoundError:
            os.makedirs(full_path, exist_ok=True)
            files = []

        num = 0

        for filenum in [fn[0:5] for fn in files if fn[5] == '-' and fn[0:5].isnumeric()]:
            test = int(filenum)
            if test > num:
                num = test

        num = num + 1

        generated_image_path = False
        high_res_image_path = False
        upscaled_image_path = False
        parameter_file_path = False

        anything_saved = False

        if seed is not None:
            if generated_image_name is not None:
                generated_image_name = generated_image_name.replace("%seed%", str(seed))
            if high_res_image_name is not None:
                high_res_image_name = high_res_image_name.replace("%seed%", str(seed))
            if upscaled_image_name is not None:
                upscaled_image_name = upscaled_image_name.replace("%seed%", str(seed))

        if save_generated_image and generated_images is not None:
            generated_image_name = generated_image_name.replace("\\", "_").replace("/", "_").replace(".", "_")
            filename = f"{num:05}-{generated_image_name}"

            generated_image_path = os.path.join(sub_folder, filename)
            images_to_save = generated_images if post_processed_images is None else post_processed_images
            self.save_images(images_to_save, full_path, filename, embed_wf_in_generated,
                             hidden_prompt, hidden_extra_pnginfo)

            anything_saved = True

        if save_high_res_image and high_res_images is not None:
            high_res_image_name = high_res_image_name.replace("\\", "_").replace("/", "_").replace(".", "_")
            filename = f"{num:05}-{high_res_image_name}"

            high_res_image_path = os.path.join(sub_folder, filename)
            images_to_save = high_res_images if post_processed_hires is None else post_processed_hires
            self.save_images(images_to_save, full_path, filename, embed_wf_in_high_res,
                             hidden_prompt, hidden_extra_pnginfo)

            anything_saved = True

        if save_upscaled_image and upscaled_images is not None:
            upscaled_image_name = upscaled_image_name.replace("\\", "_").replace("/", "_").replace(".", "_")
            filename = f"{num:05}-{upscaled_image_name}"

            upscaled_image_path = os.path.join(sub_folder, filename)
            self.save_images(upscaled_images, full_path, filename, embed_wf_in_upscaled,
                             hidden_prompt, hidden_extra_pnginfo)

            anything_saved = True

        if save_parameters_file and anything_saved:
            filename = f"{num:05}-param.txt"
            parameter_file_path = os.path.join(sub_folder, filename)
            full_filename = os.path.join(full_path, filename)

            parameters = {
                Names.S_MAGIC_BOX_VERSION: access.get_from_pipeline(Names.S_MAGIC_BOX_VERSION),

                UI.S_PROMPTS: access.get_effective_structure(UI.S_PROMPTS),
                UI.S_OPERATING_MODE: access.get_effective_structure(UI.S_OPERATING_MODE),
                UI.S_GENERATION_PARAMETERS: access.get_effective_structure(UI.S_GENERATION_PARAMETERS),
                UI.S_CONDITIONING_PARAMETERS: access.get_effective_structure(UI.S_CONDITIONING_PARAMETERS),
                UI.S_ADVANCED_PARAMETERS: access.get_effective_structure(UI.S_ADVANCED_PARAMETERS),
                UI.S_IMG2IMG_INPAINTING: access.get_effective_structure(UI.S_IMG2IMG_INPAINTING),
                UI.S_HIGH_RESOLUTION: access.get_effective_structure(UI.S_HIGH_RESOLUTION),
                UI.S_CHECKPOINTS: access.get_effective_structure(UI.S_CHECKPOINTS),
                UI.S_UPSCALE_MODELS: access.get_effective_structure(UI.S_UPSCALE_MODELS),
                UI.S_LORAS: access.get_effective_structure(UI.S_LORAS),
                UI.S_PROMPT_STYLING: access.get_effective_structure(UI.S_PROMPT_STYLING),  # TODO
                UI.S_CUSTOM_PROMPTING: access.get_effective_structure(UI.S_CUSTOM_PROMPTING),  # TODO
                UI.S_CONDITION_MIXING: access.get_effective_structure(UI.S_CONDITION_MIXING),  # TODO

                "debug_information": {
                    Names.S_PROCESSED_PROMPTS: retrieve_parameter(Names.S_PROCESSED_PROMPTS, data, {"info": "missing"})
                }
            }

            parameters_json = json.dumps(parameters, indent=4)
            with open(full_filename, "w", encoding="utf-8") as f:
                f.write(parameters_json)

        saved_files = {
            Names.F_GENERATED_IMAGE_PATH: generated_image_path,
            Names.F_HIGH_RES_IMAGE_PATH: high_res_image_path,
            Names.F_UPSCALED_IMAGE_PATH: upscaled_image_path,
            Names.F_PARAMETER_FILE_PATH: parameter_file_path,
        }

        if data is not None:
            data[Names.S_SAVED_FILES] = saved_files

        stage_output = {
            Names.S_SAVED_FILES: saved_files,
        }

        return (data, stage_output,)

    @staticmethod
    def save_images(images, full_path, filename, embed_metadata, prompt, extra_pnginfo):
        if images is None:
            print(f"Warning: trying to save {filename}, but no images were provided")
            return

        counter = 1
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if embed_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}-{counter}.png" if counter > 1 else f"{filename}.png"
            counter = counter + 1

            img.save(os.path.join(full_path, file), pnginfo=metadata, compress_level=4)
