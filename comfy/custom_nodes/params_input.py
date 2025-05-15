
MAX_RESOLUTION=32768
class GeneralInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"default": "正向提示词", "multiline": True}),           
                    "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                    "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                    "additional_prompt": ("STRING", {"default": "", "multiline": False}),
                    "var_number": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                    "wavespeed_strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }}
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "INT", "FLOAT", "INT", "INT", "STRING", "INT", "FLOAT",)
    RETURN_NAMES = ("prompt", "negative_prompt", "width", "height", "cfg", "steps", "denoise", "seed", "clip_skip", "additional_prompt",  "var_number", "wavespeed_strength",)
    
    FUNCTION = "general_input"

    CATEGORY = "api/input"

    def general_input(self, prompt, negative_prompt, width, height, cfg, steps, denoise, seed, clip_skip, additional_prompt, var_number, wavespeed_strength ):
        

        return (prompt, negative_prompt, width, height, cfg, steps, denoise, seed, clip_skip, additional_prompt, var_number, wavespeed_strength, )

class EnhanceUovInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "uov_method": (["Disabled", "Vary (Subtle)", "Vary (Strong)", "Upscale (1.5x)", "Upscale (2x)", "Upscale (Fast 2x)"], {"default": "Disabled"}),
                    "uov_processing_order": (["Before First Enhancemen", "After Last Enhancement"], {"default": "Before First Enhancemen"}),
                    "uov_prompt_type": (["Original Prompts", "Last Filled Enhancement Prompts"], {"default": ""}),
                    }}
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("uov_method", "uov_processing_order", "uov_prompt_type",)

    FUNCTION = "enhance_uov_input"

    CATEGORY = "api/input"

    def enhance_uov_input(self, uov_method, uov_processing_order, uov_prompt_type):

        return (uov_method, uov_processing_order, uov_prompt_type)


class EnhanceRegionInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "mask_dino_prompt_text": ("STRING", {"default": "识别目标", "multiline": False}),
                    "prompt": ("STRING", {"default": "", "multiline": False}),           
                    "negative_prompt": ("STRING", {"default": "", "multiline": False}),
                    "mask_model": (["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "sam"], {"default": "sam"}),
                    "mask_cloth_category": (["full", "upper", "lower"], {"default": "full"}),
                    "mask_sam_model": (["vit_b", "vit_l", "vit_h"], {"default": "vit_b"}),
                    "mask_text_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1, "step": 0.05}),
                    "mask_box_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1, "step": 0.05}),
                    "mask_sam_max_detections": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                    "mask_invert": ("BOOLEAN", {"default": False}),
                    "inpaint_disable_initial_latent": ("BOOLEAN", {"default": False}),
                    "inpaint_engine": (["v2.6", "v2.5", "None"], {"default": "v2.6", "multiline": False}),
                    "inpaint_strength": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.001}),
                    "inpaint_respective_field": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "inpaint_erode_or_dilate": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                    }}
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "INT", "BOOLEAN", "BOOLEAN", "STRING", "FLOAT", "FLOAT", "INT",)
    RETURN_NAMES = ("mask_dino_prompt_text", "prompt", "negative_prompt", "mask_model", "mask_cloth_category", "mask_sam_model", "mask_text_threshold", "mask_box_threshold", "mask_sam_max_detections", "mask_invert", "inpaint_disable_initial_latent", "inpaint_engine", "inpaint_strength",  "inpaint_respective_field", "inpaint_erode_or_dilate",)
    
    FUNCTION = "enhance_region_input"

    CATEGORY = "api/input"

    def enhance_region_input(self, mask_dino_prompt_text, prompt, negative_prompt, mask_model, mask_cloth_category, mask_sam_model, mask_text_threshold, mask_box_threshold, mask_sam_max_detections, mask_invert, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, inpaint_erode_or_dilate ):

        return (mask_dino_prompt_text, prompt, negative_prompt, mask_model, mask_cloth_category, mask_sam_model, mask_text_threshold, mask_box_threshold, mask_sam_max_detections, mask_invert, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, inpaint_erode_or_dilate, )
        
NODE_CLASS_MAPPINGS = {
    "GeneralInput": GeneralInput,
    "EnhanceUovInput": EnhanceUovInput,
    "EnhanceRegionInput": EnhanceRegionInput,
}
