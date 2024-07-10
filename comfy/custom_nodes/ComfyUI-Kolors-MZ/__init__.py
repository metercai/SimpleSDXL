import inspect
import json
import os
import folder_paths
import importlib


NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}

AUTHOR_NAME = "MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - Kolors"


class MZ_FakeCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("prompt", )
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_FakeCond_call(kwargs)


# NODE_CLASS_MAPPINGS["MZ_FakeCond"] = MZ_FakeCond
# NODE_DISPLAY_NAME_MAPPINGS[
#     "MZ_FakeCond"] = f"{AUTHOR_NAME} - FakeCond"


class MZ_ChatGLM3Loader:
    @classmethod
    def INPUT_TYPES(s):
        from .mz_kolors_utils import Utils
        llm_dir = os.path.join(Utils.get_models_path(), "LLM")
        print("llm_dir:", llm_dir)
        llm_models = Utils.listdir_models(llm_dir)

        return {"required": {
            "chatglm3_checkpoint": (llm_models,),
        }}

    RETURN_TYPES = ("CHATGLM3MODEL",)
    RETURN_NAMES = ("chatglm3_model",)
    FUNCTION = "load_chatglm3"
    CATEGORY = CATEGORY_NAME

    def load_chatglm3(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ChatGLM3Loader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ChatGLM3Loader"] = MZ_ChatGLM3Loader
NODE_DISPLAY_NAME_MAPPINGS["MZ_ChatGLM3Loader"] = f"{AUTHOR_NAME} - ChatGLM3Loader"


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
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ChatGLM3TextEncode_call(kwargs)


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

    CATEGORY = CATEGORY_NAME

    def load_unet(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsUNETLoader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsUNETLoader"] = MZ_KolorsUNETLoader
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KolorsUNETLoader"] = f"{AUTHOR_NAME} - Kolors UNET Loader"
