# -*- coding: utf-8 -*-
import folder_paths, os
from comfy.supported_models import FluxInpaint, models
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator
from nodes import UNETLoader
from PIL import Image, ImageOps, ImageSequence
import node_helpers
import numpy as np
import torch
import hashlib

class InContextEditInstruction(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "editText": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "Your edit instruction."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    RETURN_NAMES = ("In_context",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "In-context_Editing_Framework"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, editText):
        text = f"A diptych with two side-by-side images of the same scene. On the right, the scene is the same as on the left but {editText}"
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)
        print(text)
        return (clip.encode_from_tokens_scheduled(tokens), )
    
    
class DiptychCreate:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                "optional": {
                    "image_input": ("IMAGE",)
                }
                }

    CATEGORY = "In-context_Editing_Framework"

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("diptych","maskDiptych", "original_image")
    FUNCTION = "create"
    
    #TODO:作为单独的节点
    def calculate_optimal_dimensions(image):
        # Extract the original dimensions
        original_width, original_height = image.size
        
        # Set constants
        MIN_ASPECT_RATIO = 9 / 16
        MAX_ASPECT_RATIO = 16 / 9
        FIXED_DIMENSION = 1024

        # Calculate the aspect ratio of the original image
        original_aspect_ratio = original_width / original_height

        # Determine which dimension to fix
        if original_aspect_ratio > 1:  # Wider than tall
            width = FIXED_DIMENSION
            height = round(FIXED_DIMENSION / original_aspect_ratio)
        else:  # Taller than wide
            height = FIXED_DIMENSION
            width = round(FIXED_DIMENSION * original_aspect_ratio)

        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Enforce aspect ratio limits
        calculated_aspect_ratio = width / height
        if calculated_aspect_ratio > MAX_ASPECT_RATIO:
            width = (height * MAX_ASPECT_RATIO // 8) * 8
        elif calculated_aspect_ratio < MIN_ASPECT_RATIO:
            height = (width / MIN_ASPECT_RATIO // 8) * 8

        # Ensure width and height remain above the minimum dimensions
        width = max(width, 576) if width == FIXED_DIMENSION else width
        height = max(height, 576) if height == FIXED_DIMENSION else height

        return width, height
    def create(self, image, image_input=None):
        if image_input is not None:
            # Convert tensor to PIL Image
            i = 255. * image_input.cpu().numpy().squeeze()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            original_image = img.copy()
        else:
            image_path = folder_paths.get_annotated_filepath(image)
            img = node_helpers.pillow(Image.open, image_path)
            img = node_helpers.pillow(ImageOps.exif_transpose, img)
            original_image = img.copy()


        # if img.mode == 'I':
        #     img = img.point(lambda i: i * (1 / 255))
        # image = img.convert("RGB")
        # width, height = None, None
        width, height = img.size
        img = img.resize((512, int(512 * height / width)))
        width, height = img.size
        # width, height = calculate_optimal_dimensions(image)


        
        # 创建双联图
        combined_image = Image.new("RGB", (width * 2, height))
        combined_image.paste(img, (0, 0))  # 左边放原图
        combined_image.paste(img, (width, 0))  # 右边放原图
        #image1 = Image.new("RGB", (width, height), (128, 128, 128))  # 红色图像
        #combined_image.paste(image1, (width, 0))
        # 创建mask：左边黑色，右边白色
        mask_array = np.zeros((height, width * 2), dtype=np.uint8)
        mask_array[:, width:] = 255  # 右边区域设为白色
        mask = Image.fromarray(mask_array)  
        combined_image = np.array(combined_image).astype(np.float32) / 255.0    #转换为归一化的浮点数数组
        combined_image = torch.from_numpy(combined_image)[None,]    #将其转换为 PyTorch 张量，并添加批量维度和通道维度
        
        # 将 mask 转换为归一化的浮点数数组
        mask_array = np.array(mask).astype(np.float32) / 255.0

        # 将 mask 转换为 PyTorch 张量，并添加批量维度和通道维度
        mask_tensor = torch.from_numpy(mask_array)[None, None, ...]
        original_image = np.array(original_image).astype(np.float32) / 255.0
        original_image = torch.from_numpy(original_image)[None,]
        return (combined_image, mask_tensor, original_image)

    @classmethod
    def IS_CHANGED(s, image, image_input=None):
        if image_input is None:
            image_path = folder_paths.get_annotated_filepath(image)
            m = hashlib.sha256()
            with open(image_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        else:
            return hashlib.sha256(image_input.tobytes()).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
    
class ICEFConditioning:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"In_context": ("CONDITIONING",),
                             "negative": ("CONDITIONING",),
                             "vae": ("VAE",),
                             "diptych": ("IMAGE",),
                             "maskDiptych": ("MASK",),
                            #  "maskDiptych": ("BOOLEAN", {"default": True,
                            #                             "tooltip": "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."}),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("In_context", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "In-context_Editing_Framework"

    def encode(self, In_context, negative, diptych, vae, maskDiptych):
        x = (diptych.shape[1] // 8) * 8
        y = (diptych.shape[2] // 8) * 8
        print('before')
        maskDiptych = torch.nn.functional.interpolate(maskDiptych.reshape((-1, 1, maskDiptych.shape[-2], maskDiptych.shape[-1])),size=(diptych.shape[1], diptych.shape[2]), mode="bilinear")

        print('after')
        orig_pixels = diptych
        diptych = orig_pixels.clone()
        if diptych.shape[1] != x or diptych.shape[2] != y:
            x_offset = (diptych.shape[1] % 8) // 2
            y_offset = (diptych.shape[2] % 8) // 2
            diptych = diptych[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            maskDiptych = maskDiptych[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
        # why?
        m = (1.0 - maskDiptych.round()).squeeze(1)
        for i in range(3):
            diptych[:,:,:,i] -= 0.5
            diptych[:,:,:,i] *= m
            diptych[:,:,:,i] += 0.5
            
        concat_latent = vae.encode(diptych)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        out_latent["noise_mask"] = maskDiptych


        # c = node_helpers.conditioning_set_values(In_context, {"concat_latent_image": concat_latent,
        #                                                             "concat_mask": maskDiptych})


        out = []
        for conditioning in [In_context, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": maskDiptych})
            out.append(c)
        return (out[0], out[1], out_latent)#return (c, out_latent)
