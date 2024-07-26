

import gc
import json
import os
import random
import re

import torch
import folder_paths
import comfy.model_management as mm
from . import mz_kolors_core


def MZ_ChatGLM3TextEncode_call(args):

    text = args.get("text")
    chatglm3_model = args.get("chatglm3_model")

    prompt_embeds, pooled_output = mz_kolors_core.chatglm3_text_encode(
        chatglm3_model,
        text,
    )

    from torch import nn
    hid_proj: nn.Linear = args.get("hid_proj")

    if hid_proj.weight.dtype != prompt_embeds.dtype:
        with torch.cuda.amp.autocast(dtype=hid_proj.weight.dtype):
            prompt_embeds = hid_proj(prompt_embeds)
    else:
        prompt_embeds = hid_proj(prompt_embeds)

    return ([[
        prompt_embeds,
        {"pooled_output": pooled_output},
    ]], )


def load_unet_state_dict(sd):  # load unet in diffusers or regular format
    from comfy import model_management, model_detection
    import comfy.utils

    # Allow loading unets from checkpoint files
    checkpoint = False
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd
        checkpoint = True

    parameters = comfy.utils.calculate_parameters(sd)
    unet_dtype = model_management.unet_dtype(model_params=parameters)
    load_device = model_management.get_torch_device()

    from torch import nn
    hid_proj: nn.Linear = None
    if True:
        model_config = model_detection.model_config_from_diffusers_unet(sd)
        if model_config is None:
            return None

        diffusers_keys = comfy.utils.unet_to_diffusers(
            model_config.unet_config)

        new_sd = {}
        for k in diffusers_keys:
            if k in sd:
                new_sd[diffusers_keys[k]] = sd.pop(k)
            else:
                print("{} {}".format(diffusers_keys[k], k))

        encoder_hid_proj_weight = sd.pop("encoder_hid_proj.weight")
        encoder_hid_proj_bias = sd.pop("encoder_hid_proj.bias")
        hid_proj = nn.Linear(
            encoder_hid_proj_weight.shape[1], encoder_hid_proj_weight.shape[0])
        hid_proj.weight.data = encoder_hid_proj_weight
        hid_proj.bias.data = encoder_hid_proj_bias
        hid_proj = hid_proj.to(load_device)

    offload_device = model_management.unet_offload_device()
    unet_dtype = model_management.unet_dtype(
        model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys in unet: {}".format(left_over))
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device), hid_proj


def MZ_KolorsUNETLoader_call(kwargs):

    from . import hook_comfyui_kolors_v1
    with hook_comfyui_kolors_v1.apply_kolors():
        unet_name = kwargs.get("unet_name")
        unet_path = folder_paths.get_full_path("unet", unet_name)
        import comfy.utils
        sd = comfy.utils.load_torch_file(unet_path)
        model, hid_proj = load_unet_state_dict(sd)
        if model is None:
            raise RuntimeError(
                "ERROR: Could not detect model type of: {}".format(unet_path))
        return (model, hid_proj)


def MZ_FakeCond_call(kwargs):
    import torch
    cond = torch.zeros(2, 256, 4096)
    pool = torch.zeros(2, 4096)

    dtype = kwargs.get("dtype")
    if dtype == "fp16":
        print("fp16")
        cond = cond.half()
        pool = pool.half()
    elif dtype == "bf16":
        print("bf16")
        cond = cond.bfloat16()
        pool = pool.bfloat16()
    else:
        print("fp32")
        cond = cond.float()
        pool = pool.float()

    return ([[
        cond,
        {"pooled_output": pool},
    ]],)


NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}

AUTHOR_NAME = "MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - Kolors"


class MZ_ChatGLM3TextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatglm3_model": ("CHATGLM3MODEL", ),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "hid_proj": ("TorchLinear", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME + "/Legacy"

    def encode(self, **kwargs):
        return MZ_ChatGLM3TextEncode_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ChatGLM3"] = MZ_ChatGLM3TextEncode
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ChatGLM3"] = f"{AUTHOR_NAME} - ChatGLM3TextEncode"


class MZ_KolorsUNETLoader():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "unet_name": (folder_paths.get_filename_list("unet"), ),
                }}

    RETURN_TYPES = ("MODEL", "TorchLinear")

    RETURN_NAMES = ("model", "hid_proj")

    FUNCTION = "load_unet"

    CATEGORY = CATEGORY_NAME + "/Legacy"

    def load_unet(self, **kwargs):
        return MZ_KolorsUNETLoader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsUNETLoader"] = MZ_KolorsUNETLoader
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KolorsUNETLoader"] = f"{AUTHOR_NAME} - Kolors UNET Loader"


class MZ_FakeCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0}),
                "dtype": ([
                    "fp32",
                    "fp16",
                    "bf16",
                ],),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("prompt", )
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        return MZ_FakeCond_call(kwargs)


try:
    if os.environ.get("MZ_DEV", None) is not None:
        NODE_CLASS_MAPPINGS["MZ_FakeCond"] = MZ_FakeCond
        NODE_DISPLAY_NAME_MAPPINGS[
            "MZ_FakeCond"] = f"{AUTHOR_NAME} - FakeCond"
except ImportError:
    pass
