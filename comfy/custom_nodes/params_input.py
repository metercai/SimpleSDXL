import torch
import hashlib
import numpy as np
import folder_paths
import node_helpers

from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
from PIL import Image, ImageOps, ImageSequence

MAX_SEED_NUM = 1125899906842624
MAX_RESOLUTION=32768
class GeneralInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"default": "", "multiline": True}),           
                    "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                    "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "refiner_step": ("INT", {"default": 16, "min": 1, "max": 10000}),
                    "sampler": (SAMPLER_NAMES, {"default": SAMPLER_NAMES[0]}), 
                    "scheduler": (SCHEDULER_NAMES, {"default": SCHEDULER_NAMES[0]}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                    "inpaint_disable_initial_latent": ("BOOLEAN", {"default": False}),
                    "wavespeed_strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "save_final_enhanced_image_only": ("BOOLEAN", {"default": False}),
                    }}
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "INT", "INT", SAMPLER_NAMES, SCHEDULER_NAMES, "FLOAT", "INT",  "BOOLEAN", "FLOAT", "BOOLEAN", )
    RETURN_NAMES = ("prompt", "negative_prompt", "width", "height", "cfg", "steps", "refiner_step", "sampler", "scheduler", "denoise", "clip_skip", "inpaint_disable_initial_latent", "wavespeed_strength", "save_final_enhanced_image_only", )
    
    FUNCTION = "general_input"

    CATEGORY = "api/input"

    def general_input(self, prompt, negative_prompt, width, height, cfg, steps, refiner_step, sampler, scheduler, denoise, clip_skip, inpaint_disable_initial_latent, wavespeed_strength, save_final_enhanced_image_only):

        return (prompt, negative_prompt, width, height, cfg, steps, refiner_step, sampler, scheduler, denoise, clip_skip, inpaint_disable_initial_latent, wavespeed_strength, save_final_enhanced_image_only)


class SceneInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"default": "", "multiline": True}),
                    "additional_prompt": ("STRING", {"default": "", "multiline": False}),
                    "ip_image": ("STRING", {"default": "None", "multiline": False}),
                    "ip_image1": ("STRING", {"default": "None", "multiline": False}),
                    "inpaint_image": ("STRING", {"default": "None", "multiline": False}),
                    "inpaint_mask": ("STRING", {"default": "None", "multiline": False}),
                    "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "var_number": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                }}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT", "INT", )
    RETURN_NAMES = ("prompt", "additional_prompt", "ip_image", "ip_image1", "inpaint_image", "inpaint_mask", "width", "height", "var_number", )

    FUNCTION = "scene_input"

    CATEGORY = "api/input"

    def scene_input(self, prompt, additional_prompt, ip_image, ip_image1, inpaint_image, inpaint_mask, width, height, var_number):

        return (prompt, additional_prompt, ip_image, ip_image1, inpaint_image, inpaint_mask, width, height, var_number)

class SeedInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "seed_input"

    CATEGORY = "api/input"

    def seed_input(self, seed=0):
        return seed,

class EnhanceUovInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "uov_method": (["disabled", "vary (subtle)", "vary (strong)", "upscale (1.5x)", "upscale (2x)", "upscale (fast 2x)"], {"default": "disabled"}),
                "uov_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "uov_processing_order": (["Before First Enhancement", "After Last Enhancement"], {"default": "Before First Enhancement"}),
                "uov_prompt_type": (["Original Prompts", "Last Filled Enhancement Prompts"], {"default": ""}),
                "uov_multiple": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "uov_tiled_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "uov_tiled_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "uov_tiled_steps": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
                }}
    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "STRING", "FLOAT", "INT", "INT", "INT", )
    RETURN_NAMES = ("uov_method", "uov_denoise", "uov_processing_order", "uov_prompt_type", "uov_multiple", "uov_tiled_width", "uov_tiled_height", "uov_tiled_steps", )

    FUNCTION = "enhance_uov_input"

    CATEGORY = "api/input"

    def enhance_uov_input(self, uov_method, uov_denoise, uov_processing_order, uov_prompt_type, uov_multiple, uov_tiled_width, uov_tiled_height, uov_tiled_steps):

        return (uov_method, uov_denoise, uov_processing_order, uov_prompt_type, uov_multiple, uov_tiled_width, uov_tiled_height, uov_tiled_steps)


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
                    "inpaint_engine": ("STRING", {"default": "v2.6", "multiline": False}),
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
       

class LoadInputImage:
    @classmethod
    def INPUT_TYPES(s):
        files = folder_paths.get_input_directory_files()
        return {"required": {
                    "image_name": ("STRING", {}),
                    "image": (["None"]+sorted(files), {"default":"None", "image_upload": True}),
                }}

    CATEGORY = "api/input"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", )
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height", )

    FUNCTION = "load_image"

    def load_image(self, image_name, image):
        if image_name:
            image = image_name.strip()
        if image != 'None':
            image_path = folder_paths.get_annotated_filepath(image)
            img = node_helpers.pillow(Image.open, image_path)
        else:
            width = 1024
            height = 1024
            img = np.zeros((width, height), dtype=np.uint8)
            return (img, img, width, height)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, w, h)

    @classmethod
    def IS_CHANGED(s, image_name, image):
        if image_name:
            image = image_name.strip()
        m = hashlib.sha256()
        if image!='None':
            image_path = folder_paths.get_annotated_filepath(image)
            with open(image_path, 'rb') as f:
                m.update(f.read())
        else:
            m.update(image)
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image_name, image):
        if image_name:
            image = image_name.strip()
        if image!='None' and not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

NODE_CLASS_MAPPINGS = {
    "GeneralInput": GeneralInput,
    "SceneInput": SceneInput,
    "SeedInput": SeedInput,
    "LoadInputImage": LoadInputImage,
    "EnhanceUovInput": EnhanceUovInput,
    "EnhanceRegionInput": EnhanceRegionInput,
}
