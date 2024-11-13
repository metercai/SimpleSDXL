import torch

from nodes import CLIPTextEncode

from comfy.sd import CLIP, VAE

import comfy.model_patcher

import comfy.sd

import folder_paths

MAX_RESOLUTION=8192
#------------------------------------------------------------------------------
def load_checkpoint(ckpt_name,output_vae=True, output_clip=True):

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))

        model = out[0]
        clip = out[1]
        vae = out[2]

        return model, clip, vae

#------------------------------------------------------------------------------
def load_lora(model, clip, lora_name, strength_model, strength_clip):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

#------------------------------------------------------------------------------
# 添加一个辅助函数，用于交换宽度和高度
def swap_width_height(width, height):
    return height, width

#------------------------------------------------------------------------------
class Co_Input_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                    "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "swap": ("BOOLEAN", {"default": False}),  # 添加交换宽度和高度的按钮
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "positive": ("STRING", {"default": "+正向提示词", "multiline": True}),           
                    "negative": ("STRING", {"default": "-负向提示词", "multiline": True}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }}
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "STRING", "INT", "INT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("clip_skip", "width", "height", "batch_size", "positive", "negative", "seed", "steps", "cfg", "denoise",  )
    
    FUNCTION = "Zho_co_input"

    CATEGORY = "Zho模块组/Standard标准组"

    def Zho_co_input(self, clip_skip, width, height, batch_size, positive, negative, seed, steps, cfg, denoise, swap=False ):
        # 如果用户选择交换宽度和高度，则调用交换函数
        if swap:
            width, height = swap_width_height(width, height)

        return (clip_skip, width, height, batch_size, positive, negative, seed, steps, cfg, denoise,  )

#------------------------------------------------------------------------------
class Co_Loader_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                    "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                    "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"), ),
                    "lora_name": (["None"] + folder_paths.get_filename_list("loras"), ),
                    "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    }}
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    RETURN_NAMES = ("Model", "CLIP", "VAE",)
    
    FUNCTION = "Zho_co_model_loader"

    CATEGORY = "Zho模块组/Standard标准组"

    def Zho_co_model_loader(self, ckpt_name, clip_skip, vae_name, lora_name, strength_model, strength_clip):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None
        
        #主模型
        model, clip, vae = load_checkpoint(model)
        
        #Lora模型
        if lora_name != "None":
            model, clip = load_lora(model, clip, lora_name, strength_model, strength_clip)

        #Clip跳过层        
        clip = clip.clone()
        clip.clip_layer(clip_skip)

        #VAE
        if vae_name != "Baked VAE":        
           vae_path = folder_paths.get_full_path("vae", vae_name)
           vae = comfy.sd.VAE(ckpt_path=vae_path)

        return (model, clip, vae, )

#------------------------------------------------------------------------------
class Parameter_Loader_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip": ("CLIP", ),
                    "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "positive": ("STRING", {"default": "Positive", "multiline": True}),           
                    "negative": ("STRING", {"default": "Negative", "multiline": True}),
                    }}
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("Positive", "Negative", "LATENT",)
    
    FUNCTION = "Zho_co_parameter_loader"

    CATEGORY = "Zho模块组/Standard标准组"

    def Zho_co_parameter_loader(self, positive, negative, width, height, batch_size, clip):

        #潜空间图像
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])

        positive_encoded = negative_encoded = None
        positive_encoded = CLIPTextEncode().encode(clip, positive)[0]
        negative_encoded = CLIPTextEncode().encode(clip, negative)[0]

        return (positive_encoded, negative_encoded, {"samples":latent}, )

#------------------------------------------------------------------------------
# 添加一个辅助函数，用于交换宽度和高度
def swap_width_height(width, height):
    return height, width

# 修改初始潜空间_Zho类
class EmptyLatent_Swap_Zho:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "swap": ("BOOLEAN", {"default": False}),  # 添加交换宽度和高度的按钮
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "Zho模块组/Standard标准组"

    def generate(self, width, height, batch_size=1, swap=False):
        # 如果用户选择交换宽度和高度，则调用交换函数
        if swap:
            width, height = swap_width_height(width, height)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent}, )

#------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Co_Input_Zho": Co_Input_Zho,
    "Co_Loader_Zho": Co_Loader_Zho,
    "Parameter_Loader_Zho": Parameter_Loader_Zho,
    "EmptyLatent_Swap_Zho": EmptyLatent_Swap_Zho,
}
