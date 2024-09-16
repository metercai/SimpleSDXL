

import json
import os
import sys
from nodes import MAX_RESOLUTION
import comfy.utils
import shutil
import comfy.samplers
import folder_paths


WEB_DIRECTORY = "./web"

AUTHOR_NAME = u"MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - FluxExt"


import importlib

NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}

from . import mz_fluxext_core
import importlib


class MZ_Flux1PartialLoad_Patch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "double_blocks_cuda_size": ("INT", {"min": 0, "max": 16, "default": 7}),
            "single_blocks_cuda_size": ("INT", {"min": 0, "max": 37, "default": 7}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = f"{CATEGORY_NAME}"

    def load_unet(self, **kwargs):
        from . import mz_fluxext_core
        importlib.reload(mz_fluxext_core)
        return mz_fluxext_core.Flux1PartialLoad_Patch(kwargs)


NODE_CLASS_MAPPINGS["MZ_Flux1PartialLoad_Patch"] = MZ_Flux1PartialLoad_Patch
NODE_DISPLAY_NAME_MAPPINGS["MZ_Flux1PartialLoad_Patch"] = f"{AUTHOR_NAME} - Flux1PartialLoad_Patch"

import nodes


class MZ_Flux1CheckpointLoaderNF4_cpuDynOffload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "double_blocks_cuda_size": ("INT", {"min": 0, "max": 16, "default": 7}),
            "single_blocks_cuda_size": ("INT", {"min": 0, "max": 37, "default": 7}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = f"{CATEGORY_NAME}"

    def load_checkpoint(self, ckpt_name, **kwargs):
        CheckpointLoaderNF4 = nodes.NODE_CLASS_MAPPINGS.get(
            "CheckpointLoaderNF4", None)
        if CheckpointLoaderNF4 is None:
            # 必须安装 https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4
            raise Exception(
                "Please install comfyanonymous/ComfyUI_bitsandbytes_NF4 to use this node.")

        model, clip, vae = CheckpointLoaderNF4().load_checkpoint(ckpt_name)
        return mz_fluxext_core.Flux1PartialLoad_Patch({
            "model": model,
            "double_blocks_cuda_size": kwargs.get("double_blocks_cuda_size", 7),
            "single_blocks_cuda_size": kwargs.get("single_blocks_cuda_size", 7),
        })[0], clip, vae


NODE_CLASS_MAPPINGS["MZ_Flux1CheckpointLoaderNF4_cpuDynOffload"] = MZ_Flux1CheckpointLoaderNF4_cpuDynOffload
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_Flux1CheckpointLoaderNF4_cpuDynOffload"] = f"{AUTHOR_NAME} - Flux1CheckpointLoaderNF4_cpuDynOffload"


class MZ_Flux1CheckpointLoader_cpuDynOffload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "double_blocks_cuda_size": ("INT", {"min": 0, "max": 16, "default": 7}),
            "single_blocks_cuda_size": ("INT", {"min": 0, "max": 37, "default": 7}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = f"{CATEGORY_NAME}"

    def load_checkpoint(self, ckpt_name, **kwargs):
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(
            ckpt_name=ckpt_name)
        return mz_fluxext_core.Flux1PartialLoad_Patch({
            "model": model,
            "double_blocks_cuda_size": kwargs.get("double_blocks_cuda_size", 7),
            "single_blocks_cuda_size": kwargs.get("single_blocks_cuda_size", 7),
        })[0], clip, vae


NODE_CLASS_MAPPINGS["MZ_Flux1CheckpointLoader_cpuDynOffload"] = MZ_Flux1CheckpointLoader_cpuDynOffload
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_Flux1CheckpointLoader_cpuDynOffload"] = f"{AUTHOR_NAME} - Flux1CheckpointLoader_cpuDynOffload"


class MZ_Flux1UnetLoader_cpuDynOffload:
    @classmethod
    def INPUT_TYPES(s):
        args = nodes.UNETLoader().INPUT_TYPES()
        args["required"]["double_blocks_cuda_size"] = (
            "INT", {"min": 0, "max": 16, "default": 7})
        args["required"]["single_blocks_cuda_size"] = (
            "INT", {"min": 0, "max": 37, "default": 7})
        return args

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = f"{CATEGORY_NAME}"

    def load_unet(self, **kwargs):
        model = nodes.UNETLoader().load_unet(
            **{k: v for k, v in kwargs.items() if k != "double_blocks_cuda_size" and k != "single_blocks_cuda_size"})[0]

        return mz_fluxext_core.Flux1PartialLoad_Patch({
            "model": model,
            "double_blocks_cuda_size": kwargs.get("double_blocks_cuda_size", 7),
            "single_blocks_cuda_size": kwargs.get("single_blocks_cuda_size", 7),
        })


NODE_CLASS_MAPPINGS["MZ_Flux1UnetLoader_cpuDynOffload"] = MZ_Flux1UnetLoader_cpuDynOffload
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_Flux1UnetLoader_cpuDynOffload"] = f"{AUTHOR_NAME} - Flux1UnetLoader_cpuDynOffload"
