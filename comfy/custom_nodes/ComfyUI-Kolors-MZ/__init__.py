import inspect
import json
import os
import folder_paths
import importlib


NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}

MAX_RESOLUTION = 16384

AUTHOR_NAME = "MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - Kolors"
folder_paths.add_model_folder_path(
    "LLM", os.path.join(folder_paths.models_dir, "LLM"))


class MZ_ChatGLM3Loader:
    @classmethod
    def INPUT_TYPES(s):
        # from .mz_kolors_utils import Utils
        # llm_dir = os.path.join(Utils.get_models_path(), "LLM")
        # print("llm_dir:", llm_dir)
        llm_models = folder_paths.get_filename_list("LLM")

        # 筛选safetensors结尾的文件
        llm_models = [
            model for model in llm_models if model.endswith("safetensors")]

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


class MZ_ChatGLM3TextEncodeV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatglm3_model": ("CHATGLM3MODEL", ),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ChatGLM3TextEncodeV2_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ChatGLM3_V2"] = MZ_ChatGLM3TextEncodeV2
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ChatGLM3_V2"] = f"{AUTHOR_NAME} - ChatGLM3TextEncodeV2"


class MZ_ChatGLM3Embeds2Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kolors_embeds": ("KOLORS_EMBEDS", ),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)

    FUNCTION = "embeds2conditioning"
    CATEGORY = CATEGORY_NAME

    def embeds2conditioning(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ChatGLM3Embeds2Conditioning_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ChatGLM3Embeds2Conditioning"] = MZ_ChatGLM3Embeds2Conditioning
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ChatGLM3Embeds2Conditioning"] = f"{AUTHOR_NAME} - ChatGLM3Embeds2Conditioning"


# for 2048 resolution
class MZ_ChatGLM3TextEncodeAdvanceV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatglm3_model": ("CHATGLM3MODEL", ),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ChatGLM3TextEncodeV2_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ChatGLM3_Advance_V2"] = MZ_ChatGLM3TextEncodeAdvanceV2
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ChatGLM3_Advance_V2"] = f"{AUTHOR_NAME} - ChatGLM3TextEncodeAdvanceV2"


class MZ_KolorsCheckpointLoaderSimple():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = CATEGORY_NAME

    def load_checkpoint(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsCheckpointLoaderSimple_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsCheckpointLoaderSimple"] = MZ_KolorsCheckpointLoaderSimple
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KolorsCheckpointLoaderSimple"] = f"{AUTHOR_NAME} - KolorsCheckpointLoaderSimple"


class MZ_KolorsControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            # "seed": ("INT", {"default": 0, "min": 0, "max": 1000000}),
        }}

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("ControlNet",)
    FUNCTION = "load_controlnet"

    CATEGORY = CATEGORY_NAME

    def load_controlnet(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsControlNetLoader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsControlNetLoader"] = MZ_KolorsControlNetLoader
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KolorsControlNetLoader"] = f"{AUTHOR_NAME} - KolorsControlNetLoader"


class MZ_KolorsUNETLoaderV2():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "unet_name": (folder_paths.get_filename_list("unet"), ),
                }}

    RETURN_TYPES = ("MODEL",)

    RETURN_NAMES = ("model",)

    FUNCTION = "load_unet"

    CATEGORY = CATEGORY_NAME

    def load_unet(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsUNETLoaderV2_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsUNETLoaderV2"] = MZ_KolorsUNETLoaderV2
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KolorsUNETLoaderV2"] = f"{AUTHOR_NAME} - KolorsUNETLoaderV2"


class MZ_KolorsControlNetPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET", ),
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsControlNetPatch_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsControlNetPatch"] = MZ_KolorsControlNetPatch
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KolorsControlNetPatch"] = f"{AUTHOR_NAME} - KolorsControlNetPatch"


class MZ_KolorsCLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip_name": (folder_paths.get_filename_list("clip_vision"), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"

    CATEGORY = CATEGORY_NAME + "/Legacy"

    def load_clip(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsCLIPVisionLoader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsCLIPVisionLoader"] = MZ_KolorsCLIPVisionLoader
NODE_DISPLAY_NAME_MAPPINGS["MZ_KolorsCLIPVisionLoader"] = f"{AUTHOR_NAME} - KolorsCLIPVisionLoader - Legacy"


class MZ_ApplySDXLSamplingSettings():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("MODEL", )

    FUNCTION = "apply_sampling_settings"
    CATEGORY = CATEGORY_NAME

    def apply_sampling_settings(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ApplySDXLSamplingSettings_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ApplySDXLSamplingSettings"] = MZ_ApplySDXLSamplingSettings
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ApplySDXLSamplingSettings"] = f"{AUTHOR_NAME} - ApplySDXLSamplingSettings"


class MZ_ApplyCUDAGenerator():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("MODEL", )

    FUNCTION = "apply_cuda_generator"
    CATEGORY = CATEGORY_NAME

    def apply_cuda_generator(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ApplyCUDAGenerator_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ApplyCUDAGenerator"] = MZ_ApplyCUDAGenerator
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ApplyCUDAGenerator"] = f"{AUTHOR_NAME} - ApplyCUDAGenerator"


from .ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterAdvanced, IPAdapterModelLoader, IPAdapterInsightFaceLoader, IPAdapterFaceID

IPAdapterModelLoader.CATEGORY = CATEGORY_NAME + "/IPAdapter"
NODE_CLASS_MAPPINGS["MZ_IPAdapterModelLoaderKolors"] = IPAdapterModelLoader
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_IPAdapterModelLoaderKolors"] = f"IPAdapterModelLoader(kolors) - Legacy"

IPAdapterAdvanced.CATEGORY = CATEGORY_NAME + "/IPAdapter"
NODE_CLASS_MAPPINGS["MZ_IPAdapterAdvancedKolors"] = IPAdapterAdvanced
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_IPAdapterAdvancedKolors"] = f"IPAdapterAdvanced(kolors) - Legacy"

IPAdapterInsightFaceLoader.CATEGORY = CATEGORY_NAME + "/IPAdapter"
NODE_CLASS_MAPPINGS["MZ_IPAdapterInsightFaceLoader"] = IPAdapterInsightFaceLoader

NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_IPAdapterInsightFaceLoader"] = f"IPAdapterInsightFaceLoader(kolors) - Legacy"

IPAdapterFaceID.CATEGORY = CATEGORY_NAME + "/IPAdapter"
NODE_CLASS_MAPPINGS["MZ_IPAdapterFaceID"] = IPAdapterFaceID

NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_IPAdapterFaceID"] = f"IPAdapterFaceID(kolors) - Legacy"

from . import mz_kolors_legacy
NODE_CLASS_MAPPINGS.update(mz_kolors_legacy.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(mz_kolors_legacy.NODE_DISPLAY_NAME_MAPPINGS)
