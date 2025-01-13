import functools
import importlib
import json
import unittest

import comfy.model_management
import comfy.model_patcher
import comfy.sd
import folder_paths
import torch

from . import patchers, utils

HAS_VELOCATOR = importlib.util.find_spec("xelerate") is not None


def get_quant_inputs():
    return {
        "quant_type": (
            [
                "int8_dynamic",
                "e4m3_e4m3_dynamic",
                "e4m3_e4m3_dynamic_per_tensor",
                "int8_weightonly",
                "e4m3_weightonly",
                "e4m3_e4m3_weightonly",
                "e4m3_e4m3_weightonly_per_tensor",
                "nf4_weightonly",
                "af4_weightonly",
                "int4_weightonly",
            ],
        ),
        "filter_fn": (
            "STRING",
            {
                "default": "fnmatch_matches_fqn",
            },
        ),
        "filter_fn_kwargs": (
            "STRING",
            {
                "multiline": True,
                "default": '{"pattern": ["*"]}',
            },
        ),
        "kwargs": (
            "STRING",
            {
                "multiline": True,
                # "default": "{}",
            },
        ),
    }


class VelocatorLoadAndQuantizeDiffusionModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                **utils.get_weight_dtype_inputs(),
                "lowvram": ("BOOLEAN", {"default": True}),
                "full_load": ("BOOLEAN", {"default": True}),
                "quantize": ("BOOLEAN", {"default": True}),
                "quantize_on_load_device": ("BOOLEAN", {"default": True}),
                **get_quant_inputs(),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "wavespeed/velocator"

    def load_unet(
        self,
        unet_name,
        weight_dtype,
        lowvram,
        full_load,
        quantize,
        quantize_on_load_device,
        quant_type,
        filter_fn,
        filter_fn_kwargs,
        kwargs,
    ):
        model_options = {}
        if lowvram:
            model_options["initial_device"] = torch.device("cpu")
        model_options = utils.parse_weight_dtype(model_options, weight_dtype)

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        quantize_fn = None
        if quantize:
            assert HAS_VELOCATOR, "velocator is not installed"
            from xelerate.ao.quant import quantize

            kwargs = json.loads(kwargs) if kwargs else {}

            if lowvram and quantize_on_load_device:
                preprocessor = lambda t: (
                    t.to(patchers.QuantizedModelPatcher._load_device)
                    if patchers.QuantizedModelPatcher._load_device is not None
                    else t
                )
                kwargs["preprocessor"] = preprocessor
                postprocessor = lambda t: (t.to(torch.device("cpu")))
                kwargs["postprocessor"] = postprocessor

            quantize_fn = functools.partial(
                quantize,
                quant_type=quant_type,
                filter_fn=filter_fn,
                filter_fn_kwargs=(
                    json.loads(filter_fn_kwargs) if filter_fn_kwargs else {}
                ),
                **kwargs,
            )

        with patchers.QuantizedModelPatcher._override_defaults(
            quantize_fn=quantize_fn,
            lowvram=lowvram,
            full_load=full_load,
        ), utils.disable_load_models_gpu(), unittest.mock.patch.object(
            comfy.model_patcher, "ModelPatcher", patchers.QuantizedModelPatcher
        ):
            model = comfy.sd.load_diffusion_model(
                unet_path, model_options=model_options
            )

        return (model,)


class VelocatorLoadAndQuantizeClip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": ([""] + folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": ([""] + folder_paths.get_filename_list("text_encoders"),),
                "clip_name3": ([""] + folder_paths.get_filename_list("text_encoders"),),
                "type": ([member.name.lower() for member in comfy.sd.CLIPType],),
                **utils.get_weight_dtype_inputs(),
                "lowvram": ("BOOLEAN", {"default": True}),
                "full_load": ("BOOLEAN", {"default": True}),
                "quantize": ("BOOLEAN", {"default": True}),
                "quantize_on_load_device": ("BOOLEAN", {"default": True}),
                **get_quant_inputs(),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "wavespeed/velocator"

    def load_clip(
        self,
        clip_name1,
        clip_name2,
        clip_name3,
        type,
        weight_dtype,
        lowvram,
        full_load,
        quantize,
        quantize_on_load_device,
        quant_type,
        filter_fn,
        filter_fn_kwargs,
        kwargs,
    ):
        model_options = {}
        if lowvram:
            model_options["initial_device"] = torch.device("cpu")
        model_options = utils.parse_weight_dtype(model_options, weight_dtype)

        clip_paths = []
        clip_type = None
        for clip_type_ in comfy.sd.CLIPType:
            if clip_type_.name.lower() == type:
                clip_type = clip_type_
                break
        assert clip_type is not None, f"Invalid clip type: {type}"
        for clip_name in [clip_name1, clip_name2, clip_name3]:
            if clip_name:
                clip_path = folder_paths.get_full_path_or_raise(
                    "text_encoders", clip_name
                )
                clip_paths.append(clip_path)

        quantize_fn = None
        if quantize:
            assert HAS_VELOCATOR, "velocator is not installed"
            from xelerate.ao.quant import quantize

            kwargs = json.loads(kwargs) if kwargs else {}

            if lowvram and quantize_on_load_device:
                preprocessor = lambda t: (
                    t.to(patchers.QuantizedModelPatcher._load_device)
                    if patchers.QuantizedModelPatcher._load_device is not None
                    else t
                )
                kwargs["preprocessor"] = preprocessor
                postprocessor = lambda t: (t.to(torch.device("cpu")))
                kwargs["postprocessor"] = postprocessor

            quantize_fn = functools.partial(
                quantize,
                quant_type=quant_type,
                filter_fn=filter_fn,
                filter_fn_kwargs=(
                    json.loads(filter_fn_kwargs) if filter_fn_kwargs else {}
                ),
                **kwargs,
            )

        with patchers.QuantizedModelPatcher._override_defaults(
            quantize_fn=quantize_fn,
            lowvram=lowvram,
            full_load=full_load,
        ), utils.disable_load_models_gpu(), unittest.mock.patch.object(
            comfy.model_patcher, "ModelPatcher", patchers.QuantizedModelPatcher
        ):
            clip = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options,
            )

        return (clip,)


class VelocatorQuantizeModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "object_to_patch": (
                    "STRING",
                    {
                        "default": "diffusion_model",
                    },
                ),
                **get_quant_inputs(),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "wavespeed/velocator"

    def patch(
        self,
        model,
        object_to_patch,
        quantize,
        quant_type,
        filter_fn,
        filter_fn_kwargs,
        kwargs,
    ):
        assert HAS_VELOCATOR, "velocator is not installed"

        from xelerate.ao.quant import quantize

        if quantize:
            comfy.model_management.unload_all_models()
            comfy.model_management.load_models_gpu(
                [model], force_patch_weights=True, force_full_load=True
            )

            filter_fn_kwargs = json.loads(filter_fn_kwargs) if filter_fn_kwargs else {}
            kwargs = json.loads(kwargs) if kwargs else {}

            model = model.clone()
            model.add_object_patch(
                object_to_patch,
                quantize(
                    model.get_model_object(object_to_patch),
                    quant_type=quant_type,
                    filter_fn=filter_fn,
                    filter_fn_kwargs=filter_fn_kwargs,
                    **kwargs,
                ),
            )

        return (model,)


class VelocatorCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (utils.any_typ,),
                "is_patcher": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "object_to_patch": (
                    "STRING",
                    {
                        "default": "diffusion_model",
                    },
                ),
                "memory_format": (
                    ["channels_last", "contiguous_format", "preserve_format"],
                ),
                "fullgraph": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "dynamic": ("BOOLEAN", {"default": False}),
                "mode": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "cache-all:max-autotune:low-precision",
                    },
                ),
                "options": (
                    "STRING",
                    {
                        "multiline": True,
                        # "default": "{}",
                    },
                ),
                "disable": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "backend": (
                    "STRING",
                    {
                        "default": "velocator",
                    },
                ),
            }
        }

    RETURN_TYPES = (utils.any_typ,)
    FUNCTION = "patch"

    CATEGORY = "wavespeed/velocator"

    def patch(
        self,
        model,
        is_patcher,
        object_to_patch,
        memory_format,
        fullgraph,
        dynamic,
        mode,
        options,
        disable,
        backend,
    ):
        assert HAS_VELOCATOR, "velocator is not installed"

        from xelerate.compilers.xelerate_compiler import xelerate_compile
        from xelerate.utils.memory_format import apply_memory_format

        compile_function = xelerate_compile

        memory_format = getattr(torch, memory_format)

        mode = mode if mode else None
        options = json.loads(options) if options else None
        if backend == "velocator":
            backend = "xelerate"

        if is_patcher:
            patcher = model.clone()
        else:
            patcher = model.patcher
            patcher = patcher.clone()

        patcher.add_object_patch(
            object_to_patch,
            compile_function(
                apply_memory_format(
                    patcher.get_model_object(object_to_patch),
                    memory_format=memory_format,
                ),
                fullgraph=fullgraph,
                dynamic=dynamic,
                mode=mode,
                options=options,
                disable=disable,
                backend=backend,
            ),
        )

        if is_patcher:
            return (patcher,)
        else:
            model.patcher = patcher
            return (model,)
