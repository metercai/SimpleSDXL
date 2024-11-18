import torch

from nodes import CLIPTextEncode

from comfy.sd import CLIP, VAE

import comfy.model_patcher

import comfy.sd

import folder_paths

MAX_RESOLUTION=8192
#------------------------------------------------------------------------------
def load_checkpoint(主模型,output_vae=True, output_clip=True):

        ckpt_path = folder_paths.get_full_path("checkpoints", 主模型)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))

        model = out[0]
        clip = out[1]
        vae = out[2]

        return model, clip, vae

#------------------------------------------------------------------------------
def load_lora(model, clip, Lora模型, Lora模型强度, Loraclip强度):

        lora_path = folder_paths.get_full_path("loras", Lora模型)
        lora = None
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, Lora模型强度, Loraclip强度)
        return (model_lora, clip_lora)

#------------------------------------------------------------------------------
# 添加一个辅助函数，用于交换宽度和高度
def swap_width_height(width, height):
    return height, width

#------------------------------------------------------------------------------
class 综合输入器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "Clip跳过层": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                    "宽度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "高度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "交换宽高": ("BOOLEAN", {"default": False}),  # 添加交换宽度和高度的按钮
                    "生成数量": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "正向提示词": ("STRING", {"default": "正向提示词", "multiline": True}),           
                    "负向提示词": ("STRING", {"default": "负向提示词", "multiline": True}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "步数": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "CFG": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }}
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "STRING", "INT", "INT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("Clip跳过层", "宽度", "高度", "生成数量", "正向提示词", "负向提示词", "seed", "步数", "CFG", "降噪值",  )
    
    FUNCTION = "Zho_co_input"

    CATEGORY = "Zho模块组/Standard标准组"

    def Zho_co_input(self, Clip跳过层, 宽度, 高度, 生成数量, 正向提示词, 负向提示词, seed, 步数, CFG, denoise, 交换宽高=False ):
        # 如果用户选择交换宽度和高度，则调用交换函数
        if 交换宽高:
            宽度, 高度 = swap_width_height(宽度, 高度)

        return (Clip跳过层, 宽度, 高度, 生成数量, 正向提示词, 负向提示词, seed, 步数, CFG, denoise,  )

#------------------------------------------------------------------------------
class 模型加载器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "主模型": (folder_paths.get_filename_list("checkpoints"), ),
                    "Clip跳过层": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                    "VAE模型": (["自带VAE"] + folder_paths.get_filename_list("vae"), ),
                    "Lora模型": (["无"] + folder_paths.get_filename_list("loras"), ),
                    "Lora模型强度": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "Loraclip强度": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    }}
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    RETURN_NAMES = ("主模型", "CLIP模型", "VAE模型",)
    
    FUNCTION = "Zho_co_model_loader"

    CATEGORY = "Zho模块组/Standard标准组"

    def Zho_co_model_loader(self, 主模型, Clip跳过层, VAE模型, Lora模型, Lora模型强度, Loraclip强度):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None
        
        #主模型
        model, clip, vae = load_checkpoint(主模型)
        
        #Lora模型
        if Lora模型 != "无":
            model, clip = load_lora(model, clip, Lora模型, Lora模型强度, Loraclip强度)

        #Clip跳过层        
        clip = clip.clone()
        clip.clip_layer(Clip跳过层)

        #VAE
        if VAE模型 != "自带VAE":        
           vae_path = folder_paths.get_full_path("vae", VAE模型)
           vae = comfy.sd.VAE(ckpt_path=vae_path)

        return (model, clip, vae, )

#------------------------------------------------------------------------------
class 参数加载器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip": ("CLIP", ),
                    "宽度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "高度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "生成数量": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "正向提示词": ("STRING", {"default": "正向提示词", "multiline": True}),           
                    "负向提示词": ("STRING", {"default": "负向提示词", "multiline": True}),
                    }}
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("正向条件", "负向条件", "LATENT",)
    
    FUNCTION = "Zho_co_parameter_loader"

    CATEGORY = "Zho模块组/Standard标准组"

    def Zho_co_parameter_loader(self, 正向提示词, 负向提示词, 宽度, 高度, 生成数量, clip):

        #潜空间图像
        latent = torch.zeros([生成数量, 4, 高度 // 8, 宽度 // 8])

        positive_encoded = negative_encoded = None
        positive_encoded = CLIPTextEncode().encode(clip, 正向提示词)[0]
        negative_encoded = CLIPTextEncode().encode(clip, 负向提示词)[0]

        return (positive_encoded, negative_encoded, {"samples":latent}, )

#------------------------------------------------------------------------------
# 添加一个辅助函数，用于交换宽度和高度
def swap_width_height(width, height):
    return height, width

# 修改初始潜空间_Zho类
class 初始潜空间_交换_Zho:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "宽度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "高度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "批次数": ("INT", {"default": 1, "min": 1, "max": 64}),
            "交换宽高": ("BOOLEAN", {"default": False}),  # 添加交换宽度和高度的按钮
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "generate"

    CATEGORY = "Zho模块组/Standard标准组"

    def generate(self, 宽度, 高度, 批次数=1, 交换宽高=False):
        # 如果用户选择交换宽度和高度，则调用交换函数
        if 交换宽高:
            宽度, 高度 = swap_width_height(宽度, 高度)

        latent = torch.zeros([批次数, 4, 高度 // 8, 宽度 // 8])
        return ({"samples": latent}, )

#------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "综合输入器_Zho": 综合输入器_Zho,
    "模型加载器_Zho": 模型加载器_Zho,
    "参数加载器_Zho": 参数加载器_Zho,
    "初始潜空间_交换_Zho": 初始潜空间_交换_Zho,
}
