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
import nodes

from inspect import currentframe, getframeinfo
from pathlib import Path

from comfy.samplers import KSampler

from .custom_sdxl_ksampler import CfgMethods


# ====================================================================================================
# UI: Constants, lists, definitions
# ====================================================================================================

class Defs:
    DEV_MODE = False

    VERSION = "3.999" if DEV_MODE else "4.3.1"

    WORKFLOW_VERSIONS = [
        "3.991-dev",
        "3.999-RC",
    ] if DEV_MODE else [
        "4.0",
        "4.1",
        "4.2",
        "4.3",
    ]

    # --- don't touch these ---
    CATEGORY = "Searge"
    EXTENSION_PATH = str(Path(getframeinfo(currentframe()).filename).resolve().parent.parent)


# ====================================================================================================
# UI: Constants, lists, definitions
# ====================================================================================================

class UI:
    CATEGORY_DEBUG = f"{Defs.CATEGORY}/Debug"
    CATEGORY_MAGIC = f"{Defs.CATEGORY}/Magic"
    CATEGORY_MAGIC_CUSTOM_STAGES = f"{Defs.CATEGORY}/Magic/Custom Stages"
    CATEGORY_SAMPLING = f"{Defs.CATEGORY}/Sampling"
    CATEGORY_UI = f"{Defs.CATEGORY}/UI"
    CATEGORY_UI_INPUTS = f"{Defs.CATEGORY}/UI/Inputs"
    CATEGORY_UI_PROMPTING = f"{Defs.CATEGORY}/UI/Prompting"

    MAX_RESOLUTION = nodes.MAX_RESOLUTION

    EXAMPLE = "example"

    # ================================================================================
    # Selections
    # ================================================================================

    NONE = "none"
    CUSTOM = "custom"
    USE_SETTINGS = "none - use settings"

    VAE_FROM_BASE_MODEL = "from base model"
    VAE_FROM_REFINER_MODEL = "from refiner model"

    VAE_SOURCES = [
        VAE_FROM_BASE_MODEL,
        VAE_FROM_REFINER_MODEL,
    ]

    WF_MODE_TEXT_TO_IMAGE = "text-to-image"
    WF_MODE_IMAGE_TO_IMAGE = "image-to-image"
    WF_MODE_IN_PAINTING = "in-painting"

    WORKFLOW_MODES = [
        NONE,
        WF_MODE_TEXT_TO_IMAGE,
        WF_MODE_IMAGE_TO_IMAGE,
        WF_MODE_IN_PAINTING,
    ]

    PROMPTING_DEFAULT = "default - all prompts"
    PROMPTING_MAIN_AND_NEGATIVE_ONLY = "main and neg. only"
    PROMPTING_MAIN_SECONDARY_AND_NEGATIVE = "main, sec., and neg."
    PROMPTING_MAIN_ALL_EXCEPT_SECONDARY = "all except sec."
    PROMPTING_IGNORE_EMPTY = "ignore empty"
    PROMPTING_IGNORE_ALL = "ignore all"

    PROMPTING_MODES = [
        PROMPTING_DEFAULT,
        CUSTOM,
        PROMPTING_MAIN_AND_NEGATIVE_ONLY,
        PROMPTING_MAIN_SECONDARY_AND_NEGATIVE,
        PROMPTING_MAIN_ALL_EXCEPT_SECONDARY,
        PROMPTING_IGNORE_EMPTY,
        PROMPTING_IGNORE_ALL,
    ]

    SAMPLERS = KSampler.SAMPLERS
    SCHEDULERS = KSampler.SCHEDULERS

    SAMPLER_PRESET_DPMPP_2M_KARRAS = "1 - DPM++ 2M Karras"
    SAMPLER_PRESET_EULER_A = "2 - Euler a"
    SAMPLER_PRESET_DPMPP_2M_SDE_KARRAS = "3 - DPM++ 2M SDE Karras"
    SAMPLER_PRESET_DPMPP_3M_SDE_EXPONENTIAL = "4 - DPM++ 3M SDE Exp"
    SAMPLER_PRESET_DDIM_UNIFORM = "5 - DDIM Uniform"

    SAMPLER_PRESETS = [
        USE_SETTINGS,
        SAMPLER_PRESET_DPMPP_2M_KARRAS,
        SAMPLER_PRESET_EULER_A,
        SAMPLER_PRESET_DPMPP_2M_SDE_KARRAS,
        SAMPLER_PRESET_DPMPP_3M_SDE_EXPONENTIAL,
        SAMPLER_PRESET_DDIM_UNIFORM,
    ]

    RESOLUTION_1024x1024 = "1024x1024 (1:1)"
    RESOLUTION_1152x896 = "1152x896 (4:3)"
    RESOLUTION_1216x832 = "1216x832 (3:2)"
    RESOLUTION_1344x768 = "1344x768 (16:9)"
    RESOLUTION_1536x640 = "1536x640 (21:9)"
    RESOLUTION_896x1152 = "896x1152 (3:4)"
    RESOLUTION_832x1216 = "832x1216 (2:3)"
    RESOLUTION_768x1344 = "768x1344 (9:16)"
    RESOLUTION_640x1536 = "640x1536 (9:21)"
    RESOLUTION_FROM_IMAGE = "from source image"

    RESOLUTION_PRESETS = [
        USE_SETTINGS,
        RESOLUTION_1024x1024,
        RESOLUTION_1152x896,
        RESOLUTION_1216x832,
        RESOLUTION_1344x768,
        RESOLUTION_1536x640,
        RESOLUTION_896x1152,
        RESOLUTION_832x1216,
        RESOLUTION_768x1344,
        RESOLUTION_640x1536,
        RESOLUTION_FROM_IMAGE,
    ]

    SAVE_DISABLED = "none - don't save"
    SAVE_TO_OUTPUT = "output"
    SAVE_TO_OUTPUT_DATE = "output/%date%"
    SAVE_TO_OUTPUT_SEARGE_SDXL_DATE = "output/Searge-SDXL-%date%"
    SAVE_TO_INPUT = "input"

    SAVE_FOLDERS = [
        SAVE_DISABLED,
        SAVE_TO_OUTPUT,
        SAVE_TO_OUTPUT_DATE,
        SAVE_TO_OUTPUT_SEARGE_SDXL_DATE,
        SAVE_TO_INPUT,
    ]

    CFG_INTERPOLATE = CfgMethods.INTERPOLATE
    CFG_RESCALE = CfgMethods.RESCALE
    CFG_TONEMAP = CfgMethods.TONEMAP

    DYNAMIC_CFG_METHODS = [
        NONE,
        CFG_INTERPOLATE,
        CFG_RESCALE,
        CFG_TONEMAP,
    ]

    DETAILER_NORMAL = "normal"
    DETAILER_SOFT = "soft"
    DETAILER_BLURRY = "blurry"
    DETAILER_SOFT_BLURRY = "soft blurry"

    LATENT_DETAILERS = [
        NONE,
        DETAILER_NORMAL,
        DETAILER_SOFT,
        DETAILER_BLURRY,
        DETAILER_SOFT_BLURRY,
    ]

    HIRES_MODE_SIMPLE = "simple - fast"
    HIRES_MODE_NORMAL = "normal"

    HIRES_MODES = [
        NONE,
        HIRES_MODE_SIMPLE,
        HIRES_MODE_NORMAL,
    ]

    HIRES_SCALE_1_25 = "1.25x"
    HIRES_SCALE_1_5 = "1.5x"
    HIRES_SCALE_2_0 = "2x"

    HIRES_SCALE_FACTORS = [
        HIRES_SCALE_1_25,
        HIRES_SCALE_1_5,
        HIRES_SCALE_2_0,
    ]

    PRECONDITION_MODE_GAUSSIAN = "gaussian"
    # TODO: refiner preconditioning

    PRECONDITION_MODES = [
        NONE,
        PRECONDITION_MODE_GAUSSIAN,
    ]

    UPSCALE_FACTOR_1_2 = "1.2x"
    UPSCALE_FACTOR_1_25 = "1.25x"
    UPSCALE_FACTOR_1_333 = "1.333x"
    UPSCALE_FACTOR_1_5 = "1.5x"
    UPSCALE_FACTOR_2_0 = "2.0x"
    UPSCALE_FACTOR_3_0 = "3.0x"
    UPSCALE_FACTOR_4_0 = "4.0x"

    UPSCALE_FACTORS = [
        NONE,
        UPSCALE_FACTOR_1_2,
        UPSCALE_FACTOR_1_25,
        UPSCALE_FACTOR_1_333,
        UPSCALE_FACTOR_1_5,
        UPSCALE_FACTOR_2_0,
        UPSCALE_FACTOR_3_0,
        UPSCALE_FACTOR_4_0,
    ]

    MASK_MODE_DRAWN_FULL = "masked - full"
    MASK_MODE_UPLOADED_FULL = "uploaded - full"

    MASK_MODES = [
        MASK_MODE_DRAWN_FULL,
        MASK_MODE_UPLOADED_FULL,
    ]

    CN_MODE_REVISION = "revision"
    CN_MODE_CANNY = "canny"
    CN_MODE_DEPTH = "depth"
    CN_MODE_RECOLOR = "recolor"
    CN_MODE_SKETCH = "sketch"

    CONTROLNET_MODES = [
        NONE,
        CN_MODE_REVISION,
        CN_MODE_CANNY,
        CN_MODE_DEPTH,
        CN_MODE_RECOLOR,
        CN_MODE_SKETCH,
        CUSTOM,
    ]

    FREEU_DEFAULT = "default"
    FREEU_SUBTLE = "subtle"
    FREEU_SD_XL = "sdxl"
    FREEU_SD_XL_V2 = "freeu_v2-sdxl"
    FREEU_DETAILED = "detailed"
    FREEU_SD_1_5 = "sd-1.5"
    FREEU_SD_2_1 = "sd-2.1"

    FREEU_MODES = [
        NONE,
        CUSTOM,
        FREEU_DEFAULT,
        FREEU_SUBTLE,
        FREEU_SD_XL,
        FREEU_SD_XL_V2,
        FREEU_DETAILED,
        FREEU_SD_2_1,
        FREEU_SD_1_5,
    ]

    FREEU_V1 = "freeu_v1"
    FREEU_V2 = "freeu_v2"

    FREEU_VERSION = [
        FREEU_V1,
        FREEU_V2,
    ]

    # ================================================================================
    # Selection Methods
    # ================================================================================

    @staticmethod
    def CHECKPOINTS():
        return folder_paths.get_filename_list("checkpoints")

    @staticmethod
    def CHECKPOINTS_WITH_NONE():
        return [UI.NONE] + folder_paths.get_filename_list("checkpoints")

    @staticmethod
    def VAE_WITH_EMBEDDED():
        return UI.VAE_SOURCES + folder_paths.get_filename_list("vae")

    @staticmethod
    def UPSCALERS_WITH_NONE():
        return [UI.NONE] + folder_paths.get_filename_list("upscale_models")

    @staticmethod
    def UPSCALERS_1x_WITH_NONE():
        return [UI.NONE] + [fn for fn in folder_paths.get_filename_list("upscale_models") if fn.startswith("1x")]

    @staticmethod
    def UPSCALERS_4x_WITH_NONE():
        return [UI.NONE] + [fn for fn in folder_paths.get_filename_list("upscale_models") if fn.startswith("4x")]

    @staticmethod
    def LORAS_WITH_NONE():
        return [UI.NONE] + folder_paths.get_filename_list("loras")

    @staticmethod
    def CONTROLNETS_WITH_NONE():
        return [UI.NONE] + folder_paths.get_filename_list("controlnet")

    @staticmethod
    def CLIP_VISION_WITH_NONE():
        return [UI.NONE] + folder_paths.get_filename_list("clip_vision")

    # ================================================================================
    # PROCESSING
    # ================================================================================

    ALL_UI_INPUTS = []

    S_EXAMPLE_STRUCTURE = "example_structure"
    F_EXAMPLE_FIELD = "example_field"

    # ================================================================================
    # UI DATA OUTPUTS
    # ================================================================================

    # UI: Prompt Adapter
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_PROMPTS = "prompts"
    ALL_UI_INPUTS += [S_PROMPTS]

    F_MAIN_PROMPT = "main_prompt"
    F_SECONDARY_PROMPT = "secondary_prompt"
    F_STYLE_PROMPT = "style_prompt"
    F_NEGATIVE_MAIN_PROMPT = "negative_main_prompt"
    F_NEGATIVE_SECONDARY_PROMPT = "negative_secondary_prompt"
    F_NEGATIVE_STYLE_PROMPT = "negative_style_prompt"

    # UI: Prompt Adapter
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_IMAGE_INPUTS = "image_inputs"
    ALL_UI_INPUTS += [S_IMAGE_INPUTS]

    F_SOURCE_IMAGE = "source_image"
    F_SOURCE_IMAGE_CHANGED = "source_image_changed"
    F_IMAGE_MASK = "image_mask"
    F_IMAGE_MASK_CHANGED = "image_mask_changed"
    F_UPLOADED_MASK = "uploaded_mask"
    F_UPLOADED_MASK_CHANGED = "uploaded_mask_changed"

    # UI: Controlnet Models
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_CONTROLNET_MODELS = "controlnet_models"
    ALL_UI_INPUTS += [S_CONTROLNET_MODELS]

    F_CLIP_VISION_CHECKPOINT = "clip_vision_checkpoint"
    F_CANNY_CHECKPOINT = "canny_checkpoint"
    F_DEPTH_CHECKPOINT = "depth_checkpoint"
    F_RECOLOR_CHECKPOINT = "recolor_checkpoint"
    F_SKETCH_CHECKPOINT = "sketch_checkpoint"
    F_CUSTOM_CHECKPOINT = "custom_checkpoint"

    # UI: Controlnet Adapter
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_CONTROLNET_INPUTS = "controlnet_inputs"
    ALL_UI_INPUTS += [S_CONTROLNET_INPUTS]

    F_CN_STACK = "cn_stack"

    # per controlnet settings
    F_REV_CN_IMAGE = "cn_image"
    F_REV_CN_IMAGE_CHANGED = "cn_image_changed"
    F_REV_CN_MODE = "cn_pre_mode"
    F_CN_PRE_PROCESSOR = "cn_pre_processor"
    F_REV_CN_STRENGTH = "cn_rev_strength"
    F_CN_LOW_THRESHOLD = "cn_low_threshold"
    F_CN_HIGH_THRESHOLD = "cn_high_threshold"
    F_CN_START = "cn_start"
    F_CN_END = "cn_end"
    F_REV_NOISE_AUGMENTATION = "rev_noise_augmentation"
    F_REV_ENHANCER = "rev_enhancer"

    # UI: Model Selector
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_CHECKPOINTS = "checkpoints"
    ALL_UI_INPUTS += [S_CHECKPOINTS]

    F_BASE_CHECKPOINT = "base_checkpoint"
    F_REFINER_CHECKPOINT = "refiner_checkpoint"
    F_VAE_CHECKPOINT = "vae_checkpoint"

    # UI: Generation Parameters
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_GENERATION_PARAMETERS = "generation_parameters"
    ALL_UI_INPUTS += [S_GENERATION_PARAMETERS]

    F_SEED = "seed"
    F_IMAGE_SIZE_PRESET = "image_size_preset"
    F_IMAGE_WIDTH = "image_width"
    F_IMAGE_HEIGHT = "image_height"
    F_STEPS = "steps"
    F_CFG = "cfg"
    F_SAMPLER_PRESET = "sampler_preset"
    F_SAMPLER_NAME = "sampler_name"
    F_SCHEDULER = "scheduler"
    F_BASE_VS_REFINER_RATIO = "base_vs_refiner_ratio"

    # UI: Conditioning Parameters
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_CONDITIONING_PARAMETERS = "conditioning_parameters"
    ALL_UI_INPUTS += [S_CONDITIONING_PARAMETERS]

    F_BASE_CONDITIONING_SCALE = "base_conditioning_scale"
    F_REFINER_CONDITIONING_SCALE = "refiner_conditioning_scale"
    F_TARGET_CONDITIONING_SCALE = "target_conditioning_scale"
    F_POSITIVE_CONDITIONING_SCALE = "positive_conditioning_scale"
    F_NEGATIVE_CONDITIONING_SCALE = "negative_conditioning_scale"
    F_POSITIVE_AESTHETIC_SCORE = "positive_aesthetic_score"
    F_NEGATIVE_AESTHETIC_SCORE = "negative_aesthetic_score"
    F_PRECONDITION_MODE = "precondition_mode"
    F_PRECONDITION_STRENGTH = "precondition_strength"

    # UI: Advanced Parameters
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_ADVANCED_PARAMETERS = "advanced_parameters"
    ALL_UI_INPUTS += [S_ADVANCED_PARAMETERS]

    F_DYNAMIC_CFG_METHOD = "dynamic_cfg_method"
    F_DYNAMIC_CFG_FACTOR = "dynamic_cfg_factor"
    F_REFINER_DETAIL_BOOST = "refiner_detail_boost"
    F_CONTRAST_FACTOR = "contrast_factor"
    F_SATURATION_FACTOR = "saturation_factor"
    F_LATENT_DETAILER = "latent_detailer"

    # UI: Image Saving
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_IMAGE_SAVING = "image_saving"
    ALL_UI_INPUTS += [S_IMAGE_SAVING]

    F_SAVE_PARAMETERS_FILE = "save_parameters_file",
    F_SAVE_FOLDER = "save_folder",
    F_SAVE_GENERATED_IMAGE = "save_generated_image",
    F_EMBED_WORKFLOW_IN_GENERATED = "embed_workflow_in_generated",
    F_GENERATED_IMAGE_NAME = "generated_image_name",
    F_SAVE_HIGH_RES_IMAGE = "save_high_res_image",
    F_EMBED_WORKFLOW_IN_HIGH_RES = "embed_workflow_in_high_res",
    F_HIGH_RES_IMAGE_NAME = "high_res_image_name",
    F_SAVE_UPSCALED_IMAGE = "save_upscaled_image",
    F_EMBED_WORKFLOW_IN_UPSCALED = "embed_workflow_in_upscaled",
    F_UPSCALED_IMAGE_NAME = "upscaled_image_name",

    # UI: Operating Mode
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_OPERATING_MODE = "operating_mode"
    ALL_UI_INPUTS += [S_OPERATING_MODE]

    F_WORKFLOW_MODE = "workflow_mode"
    F_PROMPTING_MODE = "prompting_mode"
    F_BATCH_SIZE = "batch_size"

    # UI: Operating Mode
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_IMG2IMG_INPAINTING = "img2img_inpainting"
    ALL_UI_INPUTS += [S_IMG2IMG_INPAINTING]

    F_DENOISE = "denoise"
    F_INPAINT_MASK_BLUR = "inpaint_mask_blur"
    F_INPAINT_MASK_MODE = "inpaint_mask_mode"

    # UI: Custom Prompt Mode
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_CUSTOM_PROMPTING = "custom_prompting"
    ALL_UI_INPUTS += [S_CUSTOM_PROMPTING]

    # UI: Prompt Styles
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_PROMPT_STYLING = "prompt_styling"
    ALL_UI_INPUTS += [S_PROMPT_STYLING]

    # UI: High Resolution
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_HIGH_RESOLUTION = "high_resolution"
    ALL_UI_INPUTS += [S_HIGH_RESOLUTION]

    F_HIRES_MODE = "hires_mode"
    F_HIRES_SCALE = "hires_scale"
    F_HIRES_DENOISE = "hires_denoise"
    F_HIRES_SOFTNESS = "hires_softness"
    F_HIRES_DETAIL_BOOST = "hires_detail_boost"
    F_HIRES_CONTRAST_FACTOR = "hires_contrast_factor"
    F_HIRES_SATURATION_FACTOR = "hires_saturation_factor"
    F_HIRES_LATENT_DETAILER = "hires_latent_detailer"
    F_FINAL_UPSCALE_SIZE = "final_upscale_size"

    # UI: Condition Mixing
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_CONDITION_MIXING = "condition_mixing"
    ALL_UI_INPUTS += [S_CONDITION_MIXING]

    # UI: Upscale Models
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_UPSCALE_MODELS = "upscale_models"
    ALL_UI_INPUTS += [S_UPSCALE_MODELS]

    F_HIGH_RES_UPSCALER = "high_res_upscaler"
    F_PRIMARY_UPSCALER = "primary_upscaler"
    F_SECONDARY_UPSCALER = "secondary_upscaler"
    F_DETAIL_PROCESSOR = "detail_processor"

    # UI: Loras
    # --------------------------------------------------------------------------------
    # output structure and field names
    S_LORAS = "loras"
    ALL_UI_INPUTS += [S_LORAS]

    F_LORA_STACK = "lora_stack"

    # per lora settings
    F_LORA_NAME = "lora_name"
    F_LORA_STRENGTH = "lora_strength"

    S_FREEU = "freeu"
    ALL_UI_INPUTS += [S_FREEU]

    F_FREEU_MODE = "freeu_mode"
    F_FREEU_B1 = "freeu_b1"
    F_FREEU_B2 = "freeu_b2"
    F_FREEU_S1 = "freeu_s1"
    F_FREEU_S2 = "freeu_s2"
    F_FREEU_VERSION = "freeu_version"
