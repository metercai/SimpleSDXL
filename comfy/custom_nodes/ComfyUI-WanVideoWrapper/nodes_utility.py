import torch
import numpy as np
from comfy.utils import common_upscale

try:
    from server import PromptServer
except:
    PromptServer = None

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

class WanVideoImageResizeToClosest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
            "generation_width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "generation_height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "aspect_ratio_preservation": (["keep_input", "stretch_to_new", "crop_to_new"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", )
    RETURN_NAMES = ("image","width","height",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Resizes image to the closest supported resolution based on aspect ratio and max pixels, according to the original code"

    def process(self, image, generation_width, generation_height, aspect_ratio_preservation ):
    
        H, W = image.shape[1], image.shape[2]
        max_area = generation_width * generation_height

        crop = "disabled"

        if aspect_ratio_preservation == "keep_input":
            aspect_ratio = H / W
        elif aspect_ratio_preservation == "stretch_to_new" or aspect_ratio_preservation == "crop_to_new":
            aspect_ratio = generation_height / generation_width
            if aspect_ratio_preservation == "crop_to_new":
                crop = "center"
                
        lat_h = round(
        np.sqrt(max_area * aspect_ratio) // VAE_STRIDE[1] //
        PATCH_SIZE[1] * PATCH_SIZE[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // VAE_STRIDE[2] //
            PATCH_SIZE[2] * PATCH_SIZE[2])
        h = lat_h * VAE_STRIDE[1]
        w = lat_w * VAE_STRIDE[2]

        resized_image = common_upscale(image.movedim(-1, 1), w, h, "lanczos", crop).movedim(1, -1)

        return (resized_image, w, h)

class ExtractStartFramesForContinuations:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video_frames": ("IMAGE", {"tooltip": "Input video frames to extract the start frames from."}),
                "num_frames": ("INT", {"default": 10, "min": 1, "max": 1024, "step": 1, "tooltip": "Number of frames to get from the start of the video."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("start_frames",)
    FUNCTION = "get_start_frames"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Extracts the first N frames from a video sequence for continuations."

    def get_start_frames(self, input_video_frames, num_frames):
        if input_video_frames is None or input_video_frames.shape[0] == 0:
            log.warning("Input video frames are empty. Returning an empty tensor.")
            if input_video_frames is not None:
                return (torch.empty((0,) + input_video_frames.shape[1:], dtype=input_video_frames.dtype),)
            else:
                # Return a tensor with 4 dimensions, as expected for an IMAGE type.
                return (torch.empty((0, 64, 64, 3), dtype=torch.float32),)

        total_frames = input_video_frames.shape[0]
        num_to_get = min(num_frames, total_frames)

        if num_to_get < num_frames:
            log.warning(f"Requested {num_frames} frames, but input video only has {total_frames} frames. Returning first {num_to_get} frames.")

        start_frames = input_video_frames[:num_to_get]

        return (start_frames.cpu().float(),)

class WanVideoVACEStartToEndFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "empty_frame_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "White level of empty frame to use"}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "control_images": ("IMAGE",),
                "inpaint_mask": ("MASK", {"tooltip": "Inpaint mask to use for the empty frames"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Index to start from"}),
                "end_index": ("INT", {"default": -1, "min": -10000, "max": 10000, "step": 1, "tooltip": "Index to end at"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to create start/end frame batch and masks for VACE"

    def process(self, num_frames, empty_frame_level, start_image=None, end_image=None, control_images=None, inpaint_mask=None, start_index=0, end_index=-1):
        
        B, H, W, C = start_image.shape if start_image is not None else end_image.shape
        device = start_image.device if start_image is not None else end_image.device

        # Convert negative end_index to positive
        if end_index < 0:
            end_index = num_frames + end_index
        
        # Create output batch with empty frames
        out_batch = torch.ones((num_frames, H, W, 3), device=device) * empty_frame_level
        
        # Create mask tensor with proper dimensions
        masks = torch.ones((num_frames, H, W), device=device)
        
        # Pre-process all images at once to avoid redundant work
        if end_image is not None and (end_image.shape[1] != H or end_image.shape[2] != W):
            end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
        
        if control_images is not None and (control_images.shape[1] != H or control_images.shape[2] != W):
            control_images = common_upscale(control_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
        
        # Place start image at start_index
        if start_image is not None:
            frames_to_copy = min(start_image.shape[0], num_frames - start_index)
            if frames_to_copy > 0:
                out_batch[start_index:start_index + frames_to_copy] = start_image[:frames_to_copy]
                masks[start_index:start_index + frames_to_copy] = 0
        
        # Place end image at end_index
        if end_image is not None:
            # Calculate where to start placing end images
            end_start = end_index - end_image.shape[0] + 1
            if end_start < 0:  # Handle case where end images won't all fit
                end_image = end_image[abs(end_start):]
                end_start = 0
                
            frames_to_copy = min(end_image.shape[0], num_frames - end_start)
            if frames_to_copy > 0:
                out_batch[end_start:end_start + frames_to_copy] = end_image[:frames_to_copy]
                masks[end_start:end_start + frames_to_copy] = 0
        
        # Apply control images to remaining frames that don't have start or end images
        if control_images is not None:
            # Create a mask of frames that are still empty (mask == 1)
            empty_frames = masks.sum(dim=(1, 2)) > 0.5 * H * W
            
            if empty_frames.any():
                # Only apply control images where they exist
                control_length = control_images.shape[0]
                for frame_idx in range(num_frames):
                    if empty_frames[frame_idx] and frame_idx < control_length:
                        out_batch[frame_idx] = control_images[frame_idx]
        
        # Apply inpaint mask if provided
        if inpaint_mask is not None:
            inpaint_mask = common_upscale(inpaint_mask.unsqueeze(1), W, H, "nearest-exact", "disabled").squeeze(1).to(device)
            
            # Handle different mask lengths efficiently
            if inpaint_mask.shape[0] > num_frames:
                inpaint_mask = inpaint_mask[:num_frames]
            elif inpaint_mask.shape[0] < num_frames:
                repeat_factor = (num_frames + inpaint_mask.shape[0] - 1) // inpaint_mask.shape[0]  # Ceiling division
                inpaint_mask = inpaint_mask.repeat(repeat_factor, 1, 1)[:num_frames]

            # Apply mask in one operation
            masks = inpaint_mask * masks

        return (out_batch.cpu().float(), masks.cpu().float())


class CreateCFGScheduleFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 30, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of steps to schedule cfg for"} ),
            "cfg_scale_start": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "cfg_scale_end": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "interpolation": (["linear", "ease_in", "ease_out"], {"default": "linear", "tooltip": "Interpolation method to use for the cfg scale"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "Start percent of the steps to apply cfg"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "End percent of the steps to apply cfg"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float_list",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to generate a list of floats that can be used to schedule cfg scale for the steps, outside the set range cfg is set to 1.0"

    def process(self, steps, cfg_scale_start, cfg_scale_end, interpolation, start_percent, end_percent, unique_id):

        # Create a list of floats for the cfg schedule
        cfg_list = [1.0] * steps
        start_idx = min(int(steps * start_percent), steps - 1)
        end_idx = min(int(steps * end_percent), steps - 1)
        
        for i in range(start_idx, end_idx + 1):
            if i >= steps:
                break
                
            if end_idx == start_idx:
                t = 0
            else:
                t = (i - start_idx) / (end_idx - start_idx)
            
            if interpolation == "linear":
                factor = t
            elif interpolation == "ease_in":
                factor = t * t
            elif interpolation == "ease_out":
                factor = t * (2 - t)
            
            cfg_list[i] = round(cfg_scale_start + factor * (cfg_scale_end - cfg_scale_start), 2)
        
        # If start_percent > 0, always include the first step
        if start_percent > 0:
            cfg_list[0] = 1.0

        if unique_id and PromptServer is not None:
            try:                
                PromptServer.instance.send_progress_text(
                    f"{cfg_list}",
                    unique_id
                )
            except:
                pass

        return (cfg_list,)
    
class CreateScheduleFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 30, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of steps to schedule cfg for"} ),
            "start_value": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "end_value": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "default_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.01, "tooltip": "Default value to use for the steps"}),
            "interpolation": (["linear", "ease_in", "ease_out"], {"default": "linear", "tooltip": "Interpolation method to use for the cfg scale"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "Start percent of the steps to apply cfg"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "End percent of the steps to apply cfg"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float_list",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to generate a list of floats that can be used to schedule things like cfg and lora scale per step"

    def process(self, steps, start_value, end_value, default_value,interpolation, start_percent, end_percent, unique_id):

        # Create a list of floats for the cfg schedule
        cfg_list = [default_value] * steps
        start_idx = min(int(steps * start_percent), steps - 1)
        end_idx = min(int(steps * end_percent), steps - 1)
        
        for i in range(start_idx, end_idx + 1):
            if i >= steps:
                break
                
            if end_idx == start_idx:
                t = 0
            else:
                t = (i - start_idx) / (end_idx - start_idx)
            
            if interpolation == "linear":
                factor = t
            elif interpolation == "ease_in":
                factor = t * t
            elif interpolation == "ease_out":
                factor = t * (2 - t)

            cfg_list[i] = round(start_value + factor * (end_value - start_value), 2)

        # If start_percent > 0, always include the first step
        if start_percent > 0:
            cfg_list[0] = default_value

        if unique_id and PromptServer is not None:
            try:                
                PromptServer.instance.send_progress_text(
                    f"{cfg_list}",
                    unique_id
                )
            except:
                pass

        return (cfg_list,)
    

class DummyComfyWanModelObject:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "shift": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Sigma shift value"}),
            }
        }

    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model",)
    FUNCTION = "create"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to create empty Wan model to use with BasicScheduler -node to get sigmas"

    def create(self, shift):
        from comfy.model_sampling import ModelSamplingDiscreteFlow
        class DummyModel:
            def get_model_object(self, name):
                if name == "model_sampling":
                    model_sampling = ModelSamplingDiscreteFlow()
                    model_sampling.set_parameters(shift=shift)
                    return model_sampling
                return None
        return (DummyModel(),)
    
class WanVideoLatentReScale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "samples": ("LATENT",),
                    "direction": (["comfy_to_wrapper", "wrapper_to_comfy"], {"tooltip": "Direction to rescale latents, from comfy to wrapper or vice versa"}),
                }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Rescale latents to match the expected range for encoding or decoding between native ComfyUI VAE and the WanVideoWrapper VAE."

    def encode(self, samples, direction):
        samples = samples.copy()
        latents = samples["samples"]

        if latents.shape[1] == 48:
            mean = [
                    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
                    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
                    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
                    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
                    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
                    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
                ]
            std = [
                    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
                    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
                    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
                ]
        else:
            mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ]
            std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ]
        mean = torch.tensor(mean).view(1, latents.shape[1], 1, 1, 1)
        std = torch.tensor(std).view(1, latents.shape[1], 1, 1, 1)
        inv_std = (1.0 / std).view(1, latents.shape[1], 1, 1, 1)
        if direction == "comfy_to_wrapper":
            latents = (latents - mean.to(latents)) * inv_std.to(latents)
        elif direction == "wrapper_to_comfy":
            latents = latents / inv_std.to(latents) + mean.to(latents)

        samples["samples"] = latents

        return (samples,)
    
class WanVideoSigmaToStep:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "sigma": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("step",)
    FUNCTION = "convert"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Simply passes a float value as an integer, used to set start/end steps with sigma threshold"

    def convert(self, sigma):
        return (sigma,)
    
NODE_CLASS_MAPPINGS = {
    "WanVideoImageResizeToClosest": WanVideoImageResizeToClosest,
    "WanVideoVACEStartToEndFrame": WanVideoVACEStartToEndFrame,
    "ExtractStartFramesForContinuations": ExtractStartFramesForContinuations,
    "CreateCFGScheduleFloatList": CreateCFGScheduleFloatList,
    "DummyComfyWanModelObject": DummyComfyWanModelObject,
    "WanVideoLatentReScale": WanVideoLatentReScale,
    "CreateScheduleFloatList": CreateScheduleFloatList,
    "WanVideoSigmaToStep": WanVideoSigmaToStep
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoImageResizeToClosest": "WanVideo Image Resize To Closest",
    "WanVideoVACEStartToEndFrame": "WanVideo VACE Start To End Frame",
    "ExtractStartFramesForContinuations": "Extract Start Frames For Continuations",
    "CreateCFGScheduleFloatList": "Create CFG Schedule Float List",
    "DummyComfyWanModelObject": "Dummy Comfy Wan Model Object",
    "WanVideoLatentReScale": "WanVideo Latent ReScale",
    "CreateScheduleFloatList": "Create Schedule Float List",
    "WanVideoSigmaToStep": "WanVideo Sigma To Step"
}