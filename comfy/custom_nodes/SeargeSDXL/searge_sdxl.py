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

from .modules.prompt_text_input import SeargeTextInputV2
from .modules.prompt_adapter import SeargePromptAdapterV2
from .modules.image_adapter import SeargeImageAdapterV2
from .modules.controlnet_adapter import SeargeControlnetAdapterV2

from .modules.ui_separator import SeargeSeparator
from .modules.ui_preview_image import SeargePreviewImage

from .modules.ui_model_selector import SeargeModelSelector
from .modules.ui_upscale_models import SeargeUpscaleModels
from .modules.ui_loras import SeargeLoras
from .modules.ui_controlnet_models import SeargeControlnetModels

from .modules.ui_generation_parameters import SeargeGenerationParameters
from .modules.ui_conditioning_parameters import SeargeConditioningParameters
from .modules.ui_advanced_parameters import SeargeAdvancedParameters
from .modules.ui_image_saving import SeargeImageSaving
from .modules.ui_operating_mode import SeargeOperatingMode
from .modules.ui_img2img_inpaint import SeargeImage2ImageAndInpainting
from .modules.ui_custom_prompt_mode import SeargeCustomPromptMode
from .modules.ui_prompt_styles import SeargePromptStyles
from .modules.ui_high_resolution import SeargeHighResolution
from .modules.ui_condition_mixing import SeargeConditionMixing
from .modules.ui_freeu import SeargeFreeU

from .modules.magic_box import SeargeMagicBox
from .modules.mb_pipeline_start import SeargePipelineStart
from .modules.mb_pipeline_terminator import SeargePipelineTerminator

from .modules.after_vae_decode import SeargeCustomAfterVaeDecode
from .modules.after_upscaling import SeargeCustomAfterUpscaling

from .modules.debug_printer import SeargeDebugPrinter

# from .modules.sdxl_sampler import SeargeSDXLSamplerV4
# from .modules.sdxl_sampler_input import SeargeSDXLSamplerV4Inputs
# from .modules.sdxl_sampler_output import SeargeSDXLSamplerV4Outputs

# from .modules.prompt_adapter_output import SeargePromptAdapterV2Output

# ====================================================================================================
# Register nodes in ComfyUI
# ====================================================================================================

SEARGE_CLASS_MAPPINGS = {
    f"SeargeTextInputV2": SeargeTextInputV2,
    f"SeargePromptAdapterV2": SeargePromptAdapterV2,
    f"SeargeImageAdapterV2": SeargeImageAdapterV2,
    f"SeargeControlnetAdapterV2": SeargeControlnetAdapterV2,

    f"SeargeSeparator": SeargeSeparator,
    f"SeargePreviewImage": SeargePreviewImage,

    f"SeargeAdvancedParameters": SeargeAdvancedParameters,
    f"SeargeConditioningParameters": SeargeConditioningParameters,
    f"SeargeConditionMixing": SeargeConditionMixing,
    f"SeargeControlnetModels": SeargeControlnetModels,
    f"SeargeCustomPromptMode": SeargeCustomPromptMode,
    f"SeargeFreeU": SeargeFreeU,
    f"SeargeGenerationParameters": SeargeGenerationParameters,
    f"SeargeHighResolution": SeargeHighResolution,
    f"SeargeImage2ImageAndInpainting": SeargeImage2ImageAndInpainting,
    f"SeargeImageSaving": SeargeImageSaving,
    f"SeargeLoras": SeargeLoras,
    f"SeargeModelSelector": SeargeModelSelector,
    f"SeargeOperatingMode": SeargeOperatingMode,
    f"SeargePromptStyles": SeargePromptStyles,
    f"SeargeUpscaleModels": SeargeUpscaleModels,

    f"SeargeMagicBox": SeargeMagicBox,
    f"SeargePipelineStart": SeargePipelineStart,
    f"SeargePipelineTerminator": SeargePipelineTerminator,

    f"SeargeCustomAfterVaeDecode": SeargeCustomAfterVaeDecode,
    f"SeargeCustomAfterUpscaling": SeargeCustomAfterUpscaling,

    f"SeargeDebugPrinter": SeargeDebugPrinter,
}

# SEARGE_CLASS_MAPPINGS = SEARGE_CLASS_MAPPINGS | {
#     f"SeargeSDXLSamplerV4": SeargeSDXLSamplerV4,
#     f"SeargeSDXLSamplerV4Inputs": SeargeSDXLSamplerV4Inputs,
#     f"SeargeSDXLSamplerV4Outputs": SeargeSDXLSamplerV4Outputs,
#
#     f"SeargePromptAdapterV2Output": SeargePromptAdapterV2Output,
# }

# ====================================================================================================
# Human readable names for the nodes
# ====================================================================================================

SEARGE_DISPLAY_NAME_MAPPINGS = {
    f"SeargeTextInputV2": "Text Input v2",
    f"SeargePromptAdapterV2": "Prompt Adapter v2",
    f"SeargeImageAdapterV2": "Image Adapter v2",
    f"SeargeControlnetAdapterV2": "Controlnet Adapter v2",

    f"SeargeSeparator": "Separator",
    f"SeargePreviewImage": "SeargePreviewImage",

    f"SeargeAdvancedParameters": "Advanced Parameters v2",
    f"SeargeConditioningParameters": "Conditioning Parameters v2",
    f"SeargeConditionMixing": "Condition Mixing v2",
    f"SeargeControlnetModels": "Controlnet Models Selector v2",
    f"SeargeCustomPromptMode": "Custom Prompt Mode v2",
    f"SeargeFreeU": "FreeU v2",
    f"SeargeGenerationParameters": "Generation Parameters v2",
    f"SeargeHighResolution": "High Resolution v2",
    f"SeargeImage2ImageAndInpainting": "Image to Image and Inpainting v2",
    f"SeargeImageSaving": "Image Saving v2",
    f"SeargeLoras": "Lora Selector v2",
    f"SeargeModelSelector": "Model Selector v2",
    f"SeargeOperatingMode": "Operating Mode v2",
    f"SeargePromptStyles": "Prompt Styles v2",
    f"SeargeUpscaleModels": "Upscale Models Selector v2",

    f"SeargeMagicBox": "Searge's Magic Box for SDXL",
    f"SeargePipelineStart": "Magic Box Pipeline Start",
    f"SeargePipelineTerminator": "Magic Box Pipeline Terminator",

    f"SeargeCustomAfterVaeDecode": "After VAE Decode",
    f"SeargeCustomAfterUpscaling": "After Upscaling",

    f"SeargeDebugPrinter": "Debug Printer",
}

# SEARGE_DISPLAY_NAME_MAPPINGS = SEARGE_DISPLAY_NAME_MAPPINGS | {
#     f"SeargeSDXLSamplerV4": "SDXL Sampler v4",
#     f"SeargeSDXLSamplerV4Inputs": "SDXL Sampler v4 Inputs",
#     f"SeargeSDXLSamplerV4Outputs": "SDXL Sampler v4 Outputs",
#
#     f"SeargePromptAdapterV2Output": "Prompt Adapter v2 Output",
# }
