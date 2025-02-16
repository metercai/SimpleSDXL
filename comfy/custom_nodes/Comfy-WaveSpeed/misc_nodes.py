import folder_paths
import importlib
import json
import comfy.sd

from . import utils


class EnhancedLoadDiffusionModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                **utils.get_weight_dtype_inputs(),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "wavespeed"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        model_options = utils.parse_weight_dtype(model_options, weight_dtype)

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


class EnhancedCompileModel:

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
                "compiler": (
                    "STRING",
                    {
                        "default": "torch.compile",
                    }
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
                        "default": "",
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
                        "default": "inductor",
                    },
                ),
            }
        }

    RETURN_TYPES = (utils.any_typ,)
    FUNCTION = "patch"

    CATEGORY = "wavespeed"

    def patch(
        self,
        model,
        is_patcher,
        object_to_patch,
        compiler,
        fullgraph,
        dynamic,
        mode,
        options,
        disable,
        backend,
    ):
        utils.patch_optimized_module()
        utils.patch_same_meta()

        import_path, function_name = compiler.rsplit(".", 1)
        module = importlib.import_module(import_path)
        compile_function = getattr(module, function_name)

        mode = mode if mode else None
        options = json.loads(options) if options else None

        if compiler == "torch.compile" and backend == "inductor" and dynamic:
            # TODO: Fix this
            # File "pytorch/torch/_inductor/fx_passes/post_grad.py", line 643, in same_meta
            #   and statically_known_true(sym_eq(val1.size(), val2.size()))
            #   AttributeError: 'SymInt' object has no attribute 'size'
            pass

        if is_patcher:
            patcher = model.clone()
        else:
            patcher = model.patcher
            patcher = patcher.clone()

        patcher.add_object_patch(
            object_to_patch,
            compile_function(
                patcher.get_model_object(object_to_patch),
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
