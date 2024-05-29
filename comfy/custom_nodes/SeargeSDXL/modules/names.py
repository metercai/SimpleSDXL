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


# ====================================================================================================
# Names to be used for data streams and the structures & fields in them
# ====================================================================================================

class Names:
    PLACEHOLDER = "placeholder"

    S_EXAMPLE_STRUCTURE = "example_structure"
    F_EXAMPLE_FIELD = "example_field"

    # ----------------------------------------
    # structures and fields
    # ----------------------------------------

    # magic box
    B_MAGIC_BOX_ENABLED = "magic_box_enabled"

    # pre-processor
    S_MAGIC_BOX_HIDDEN = "hidden_fields"
    F_MAGIC_BOX_PROMPT = "prompt"
    F_MAGIC_BOX_EXTRA_PNGINFO = "pnginfo"

    S_MAGIC_BOX_VERSION = "version_info"
    F_MAGIC_BOX_EXTENSION = "extension_version"
    F_MAGIC_BOX_WORKFLOW = "workflow_version"

    S_CONDITION_ZEROING = "condition_zeroing"
    F_ZERO_POSITIVES = "zero_positives"
    F_ZERO_NEGATIVES = "zero_negatives"

    # vae decoder stage outputs
    S_VAE_DECODED = "vae_decoded"
    F_DECODED_IMAGE = "image"
    F_POST_PROCESSED = "post_processed"

    # checkpoint loader
    S_LOADED_MODELS = "loaded_models"
    F_BASE_MODEL = "base_model"
    F_BASE_CLIP = "base_clip"
    F_BASE_VAE = "base_vae"
    F_REFINER_MODEL = "refiner_model"
    F_REFINER_CLIP = "refiner_clip"
    F_REFINER_VAE = "refiner_vae"
    F_VAE_MODEL = "vae_model"
    F_HIRES_UPSCALER = "hires_upscaler"
    F_PRIMARY_UPSCALER = "primary_upscaler"
    F_SECONDARY_UPSCALER = "secondary_upscaler"
    F_DETAIL_PROCESSOR = "detail_processor"
    F_CLIP_VISION_MODEL = "clip_vision_model"
    F_CN_CANNY_MODEL = "cn_canny_model"
    F_CN_DEPTH_MODEL = "cn_depth_model"
    F_CN_RECOLOR_MODEL = "cn_recolor_model"
    F_CN_SKETCH_MODEL = "cn_sketch_model"
    F_CN_CUSTOM_MODEL = "cn_custom_model"

    # apply loras
    S_LOADED_LORAS = "loaded_loras"
    F_LORA_NAMES = "lora_names"

    # clip conditioning
    S_PROCESSED_PROMPTS = "processed_prompts"
    F_BASE_POSITIVE_MAIN_PROMPT = "base_positive_main_prompt"
    F_BASE_POSITIVE_SECONDARY_PROMPT = "base_positive_secondary_prompt"
    F_BASE_POSITIVE_STYLE_PROMPT = "base_positive_style_prompt"
    F_BASE_NEGATIVE_MAIN_PROMPT = "base_negative_main_prompt"
    F_BASE_NEGATIVE_SECONDARY_PROMPT = "base_negative_secondary_prompt"
    F_BASE_NEGATIVE_STYLE_PROMPT = "base_negative_style_prompt"
    F_REFINER_POSITIVE_PROMPT = "refiner_positive_prompt"
    F_REFINER_POSITIVE_STYLE_PROMPT = "refiner_positive_style_prompt"
    F_REFINER_NEGATIVE_PROMPT = "refiner_negative_prompt"
    F_REFINER_NEGATIVE_STYLE_PROMPT = "refiner_negative_style_prompt"

    S_CONDITIONING = "conditioning"
    F_BASE_POSITIVE = "base_positive"
    F_BASE_POSITIVE_STYLE = "base_positive_style"
    F_BASE_NEGATIVE = "base_negative"
    F_BASE_NEGATIVE_STYLE = "base_negative_style"
    F_REFINER_POSITIVE = "refiner_positive"
    F_REFINER_POSITIVE_STYLE = "refiner_positive_style"
    F_REFINER_NEGATIVE = "refiner_negative"
    F_REFINER_NEGATIVE_STYLE = "refiner_negative_style"

    # apply controlnet
    S_CONTROLNET_OUTPUT = "controlnet_output"
    F_CN_BASE_POSITIVE = "cn_base_positive"
    F_CN_BASE_NEGATIVE = "cn_base_negative"

    # latent inputs
    S_LATENT_INPUTS = "latent_inputs"
    F_LATENT_IMAGE = "latent_image"

    # sampler
    S_SAMPLED_IMAGE = "sampled_image"
    F_LATENT_SAMPLED = "latent_sampled"

    # latent detailer
    S_LATENT_DETAILED = "latent_detailed"
    F_DETAILED_SAMPLED = "detailed_sampled"

    # vae decode sampled
    S_VAE_DECODED_SAMPLED = "vae_decoded_sampled"
    F_DECODED_SAMPLED_IMAGE = "sampled_image"
    F_SAMPLED_POST_PROCESSED = "sampled_post_processed"

    # high resolution
    S_HIRES_OUTPUT = "hires_output"
    F_LATENT_HIRES = "latent_hires"

    # hires detailer
    S_HIRES_DETAILED = "hires_detailed"
    F_DETAILED_HIRES = "detailed_hires"

    # vae decode hires
    S_VAE_DECODED_HIRES = "vae_decoded_hires"
    F_DECODED_HIRES_IMAGE = "hires_image"
    F_HIRES_POST_PROCESSED = "hires_post_processed"

    # upscaling
    S_UPSCALED = "upscaled"
    F_UPSCALED_IMAGE = "upscaled_image"

    # image saving
    S_SAVED_FILES = "saved_files"
    F_GENERATED_IMAGE_PATH = "generated_image_path"
    F_HIGH_RES_IMAGE_PATH = "high_res_image_path"
    F_UPSCALED_IMAGE_PATH = "upscaled_image_path"
    F_PARAMETER_FILE_PATH = "parameter_file_path"

    # ----------------------------------------
    # cache names
    # ----------------------------------------

    # pre-processor
    C_SOURCE_IMAGE = "source_image"
    C_IMAGE_SIZE = "image_size"
    C_SOURCE_MASK = "source_mask"
    C_BLURRY_MASK = "blurry_mask"

    # checkpoint loader
    C_BASE_CHECKPOINT = "base_checkpoint"
    C_REFINER_CHECKPOINT = "refiner_checkpoint"
    C_VAE_CHECKPOINT = "vae_checkpoint"
    C_HIRES_UPSCALE_MODEL = "hires_upscale_checkpoint"
    C_PRIMARY_UPSCALE_MODEL = "primary_upscale_checkpoint"
    C_SECONDARY_UPSCALE_MODEL = "secondary_upscale_checkpoint"
    C_DETAIL_PROCESSOR_MODEL = "detail_processor_checkpoint"
    C_CLIP_VISION_MODEL = "clip_vision_checkpoint"
    C_CN_CANNY_MODEL = "cn_canny_checkpoint"
    C_CN_DEPTH_MODEL = "cn_depth_checkpoint"
    C_CN_RECOLOR_MODEL = "cn_recolor_checkpoint"
    C_CN_SKETCH_MODEL = "cn_sketch_checkpoint"
    C_CN_CUSTOM_MODEL = "cn_custom_checkpoint"

    # apply loras
    C_APPLIED_LORAS = "applied_loras"

    # clip conditioning
    C_PROCESSED_PROMPTS = "processed_prompts"
    C_BASE_CONDITIONING = "base_conditioning"
    C_REFINER_CONDITIONING = "refiner_conditioning"

    # apply controlnet
    C_APPLIED_CONTROLNET = "applied_controlnet"

    # latent inputs
    C_LATENT_FROM_IMAGE = "latent_from_image"
    C_IMAGE_MASK = "image_mask"
    C_LATENT_WITH_MASK = "latent_with_mask"
    C_EMPTY_LATENT = "empty_latent"

    # sampler
    C_SAMPLED = "sampled"

    # latent detailer
    C_SAMPLED_DETAILER = "sampled_detailer"

    # vae decode sampled
    C_VAE_DECODED = "vae_decoded"
    C_POST_PROCESSED = "post_processed"

    # high resolution
    C_HIRES_LATENT = "hires_latent"
    C_HIRES_LATENT_SIMPLE = "hires_latent_simple"
    C_HIRES_LATENT_NORMAL = "hires_latent_normal"

    # hires detailer
    C_HIRES_DETAILER = "hires_detailer"

    # vae decode sampled
    C_VAE_DECODED_HIRES = "vae_decoded_hires"
    C_POST_PROCESSED_HIRES = "post_processed_hires"

    # upscaling
    C_UPSCALED_IMAGE = "upscaled_image"

    # ----------------------------------------
    # pipeline stream names
    # ----------------------------------------

    P_IMAGE = "image"
    P_MASK = "mask"
    P_LATENT = "latent"

    P_BASE_MODEL = "base_model"
    P_BASE_CLIP = "base_clip"
    P_BASE_VAE = "base_vae"

    P_REFINER_MODEL = "refiner_model"
    P_REFINER_CLIP = "refiner_clip"
    P_REFINER_VAE = "refiner_vae"

    P_VAE_MODEL = "vae_model"

    P_HIRES_UPSCALER = "hires_upscaler"
    P_PRIMARY_UPSCALER = "primary_upscaler"
    P_SECONDARY_UPSCALER = "secondary_upscaler"
    P_DETAIL_PROCESSOR = "detail_processor"

    P_CLIP_VISION_MODEL = "clip_vision_model"
    P_CN_CANNY_MODEL = "cn_canny_model"
    P_CN_DEPTH_MODEL = "cn_depth_model"
    P_CN_RECOLOR_MODEL = "cn_recolor_model"
    P_CN_SKETCH_MODEL = "cn_sketch_model"
    P_CN_CUSTOM_MODEL = "cn_custom_model"

    P_PROCESSED_PROMPTS = "processed_prompts"
    P_BASE_CONDITIONING = "base_conditioning"
    P_REFINER_CONDITIONING = "refiner_conditioning"
