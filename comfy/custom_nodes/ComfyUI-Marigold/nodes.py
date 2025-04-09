import os
import torch
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

from .marigold.model.marigold_pipeline import MarigoldPipeline
from .marigold.util.ensemble import ensemble_depths
from .marigold.util.image_util import chw2hwc, colorize_depth_maps


import comfy.utils
import model_management 
import folder_paths

def colorizedepth(depth_map, colorize_method):
    depth_map = depth_map.cpu().numpy()
    percentile = 0.03
    min_depth_pct = np.percentile(depth_map, percentile)
    max_depth_pct = np.percentile(depth_map, 100 - percentile)
    
    depth_colored = colorize_depth_maps(
        depth_map, min_depth_pct, max_depth_pct, cmap=colorize_method
    ).squeeze()  # [3, H, W], value in (0, 1)
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    return depth_colored_hwc

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    elif dtype_str == 'fp8':
        return torch.float8_e4m3fn
    
    else:
        raise NotImplementedError
    
script_directory = os.path.dirname(os.path.abspath(__file__))
empty_text_embed = torch.load(os.path.join(script_directory, "empty_text_embed.pt"), map_location="cpu")

class MarigoldDepthEstimation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
            "n_repeat": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
            "regularizer_strength": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 4096, "step": 0.001}),
            "reduction_method": (
            [   
                'median',
                'mean',  
            ], {
               "default": 'median'
            }),
            "max_iter": ("INT", {"default": 5, "min": 1, "max": 4096, "step": 1}),
            "tol": ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 1e-1, "step": 1e-6}),
            
            "invert": ("BOOLEAN", {"default": True}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "n_repeat_batch_size": ("INT", {"default": 2, "min": 1, "max": 4096, "step": 1}),
            "use_fp16": ("BOOLEAN", {"default": True}),
            "scheduler": (
            [   
                'DDIMScheduler',
                'DDPMScheduler',
                'PNDMScheduler',
                'DEISMultistepScheduler',
                'LCMScheduler',
            ], {
               "default": 'DDIMScheduler'
            }),
            "normalize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model": (
            [   
                'Marigold',
                'marigold-lcm-v1-0',  
            ], {
               "default": 'Marigold'
            }),
            }
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("ensembled_image",)
    FUNCTION = "process"
    CATEGORY = "Marigold"
    DESCRIPTION = """
Diffusion-based monocular depth estimation:  
https://github.com/prs-eth/Marigold  
  
- denoise_steps: steps per depth map, increase for accuracy in exchange of processing time
- n_repeat: amount of iterations to be ensembled into single depth map
- n_repeat_batch_size: how many of the n_repeats are processed as a batch,  
if you have the VRAM this can match the n_repeats for faster processing  
- model: Marigold or it's LCM version marigold-lcm-v1-0  
For the LCM model use around 4 steps and the LCMScheduler  
- scheduler: Different schedulers give bit different results  
- invert: marigold by default produces depth map where black is front,  
for controlnets etc. we want the opposite.  
- regularizer_strength, reduction_method, max_iter, tol (tolerance) are settings   
for the ensembling process, generally do not touch.  
- use_fp16: if true, use fp16, if false use fp32  
fp16 uses much less VRAM, but in some cases can lead to loss of quality.  
"""

    def process(self, image, seed, denoise_steps, n_repeat, regularizer_strength, reduction_method, max_iter, tol,invert, keep_model_loaded, n_repeat_batch_size, use_fp16, scheduler, normalize, model="Marigold"):
        batch_size = image.shape[0]
        precision = torch.float16 if use_fp16 else torch.float32
        device = model_management.get_torch_device()
        torch.manual_seed(seed)

        image = image.permute(0, 3, 1, 2).to(device).to(dtype=precision)
        if normalize:
            image = image * 2.0 - 1.0

        diffusers_model_path = os.path.join(folder_paths.models_dir,'diffusers')
        #load the diffusers model
        if model == "Marigold":
            folders_to_check = [
                os.path.join(script_directory,"checkpoints","Marigold_v1_merged",),
                os.path.join(script_directory,"checkpoints","Marigold",),
                os.path.join(diffusers_model_path,"Marigold_v1_merged"),
                os.path.join(diffusers_model_path,"Marigold")
            ]
        elif model == "marigold-lcm-v1-0":
            folders_to_check = [
                os.path.join(diffusers_model_path,"marigold-lcm-v1-0"),
                os.path.join(diffusers_model_path,"checkpoints","marigold-lcm-v1-0")
            ]
        self.custom_config = {
            "model": model,
            "use_fp16": use_fp16,
            "scheduler": scheduler,
        }
        if not hasattr(self, 'marigold_pipeline') or self.marigold_pipeline is None or self.current_config != self.custom_config:
            self.current_config = self.custom_config
            # Load the model only if it hasn't been loaded before
            checkpoint_path = None
            for folder in folders_to_check:
                if os.path.exists(folder):
                    checkpoint_path = folder
                    break
            to_ignore = ["*.bin", "*fp16*"]

            if checkpoint_path is None:
                if model == "Marigold":
                    try:
                        from huggingface_hub import snapshot_download
                        checkpoint_path = os.path.join(diffusers_model_path, "Marigold")
                        snapshot_download(repo_id="Bingxin/Marigold", ignore_patterns=to_ignore, local_dir=checkpoint_path, local_dir_use_symlinks=False)  
                    except:
                        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_path}")
                if model == "marigold-lcm-v1-0":
                    try:
                        from huggingface_hub import snapshot_download
                        checkpoint_path = os.path.join(diffusers_model_path, "marigold-lcm-v1-0")
                        snapshot_download(repo_id="prs-eth/marigold-lcm-v1-0", ignore_patterns=to_ignore, local_dir=checkpoint_path, local_dir_use_symlinks=False)  
                    except:
                        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_path}")

            self.marigold_pipeline = MarigoldPipeline.from_pretrained(checkpoint_path, enable_xformers=False, empty_text_embed=empty_text_embed, noise_scheduler_type=scheduler)
            self.marigold_pipeline = self.marigold_pipeline.to(device).half() if use_fp16 else self.marigold_pipeline.to(device)
            self.marigold_pipeline.unet.eval()  # Set the model to evaluation mode
        pbar = comfy.utils.ProgressBar(batch_size * n_repeat)

        out = []

        with torch.no_grad():
            for i in range(batch_size):
                # Duplicate the current image n_repeat times
                duplicated_batch = image[i].unsqueeze(0).repeat(n_repeat, 1, 1, 1)
                
                # Process the duplicated batch in sub-batches
                depth_maps = []
                for j in range(0, n_repeat, n_repeat_batch_size):
                    # Get the current sub-batch
                    sub_batch = duplicated_batch[j:j + n_repeat_batch_size]
                    
                    # Process the sub-batch
                    depth_maps_sub_batch = self.marigold_pipeline(sub_batch, num_inference_steps=denoise_steps, show_pbar=False)
                    
                    # Process each depth map in the sub-batch if necessary
                    for depth_map in depth_maps_sub_batch:
                        depth_map = torch.clip(depth_map, -1.0, 1.0)
                        depth_map = (depth_map + 1.0) / 2.0
                        depth_maps.append(depth_map)
                        pbar.update(1)
                
                depth_predictions = torch.cat(depth_maps, dim=0).squeeze()
                del duplicated_batch, depth_maps_sub_batch
                torch.cuda.empty_cache()  # clear vram cache for ensembling

                # Test-time ensembling
                if n_repeat > 1:
                    depth_map, pred_uncert = ensemble_depths(
                        depth_predictions,
                        regularizer_strength=regularizer_strength,
                        max_iter=max_iter,
                        tol=tol,
                        reduction=reduction_method,
                        max_res=None,
                        device=device,
                    )
                    print(depth_map.shape)
                    depth_map = depth_map.unsqueeze(2).repeat(1, 1, 3)
                    print(depth_map.shape)
                else:
                    depth_map = depth_map.permute(1, 2, 0)
                    depth_map = depth_map.repeat(1, 1, 3)
                    print(depth_map.shape)
                
                out.append(depth_map)
                del depth_map, depth_predictions
      
        if invert:
            outstack = 1.0 - torch.stack(out, dim=0).cpu().to(torch.float32)
        else:
            outstack = torch.stack(out, dim=0).cpu().to(torch.float32)

        if not keep_model_loaded:
            self.marigold_pipeline = None
            model_management.soft_empty_cache()
        return (outstack,)        
        
class MarigoldDepthEstimationVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "first_frame_denoise_steps": ("INT", {"default": 4, "min": 1, "max": 4096, "step": 1}),
            "first_frame_n_repeat": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
            "n_repeat_batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),          
            "invert": ("BOOLEAN", {"default": True}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "scheduler": (
            [   
                'DDIMScheduler',
                'DDPMScheduler',
                'PNDMScheduler',
                'DEISMultistepScheduler',
                'LCMScheduler',
            ], {
               "default": 'DEISMultistepScheduler'
            }),
            "normalize": ("BOOLEAN", {"default": True}),
            "denoise_steps": ("INT", {"default": 4, "min": 1, "max": 4096, "step": 1}),
            "flow_warping": ("BOOLEAN", {"default": True}),
            "flow_depth_mix": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            "noise_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "dtype": (
            [   
                'fp16',
                'bf16',
                'fp32',
            ], {
               "default": 'fp16'
            }),
            },
            "optional": {
                "model": (
            [   
                'Marigold',
                'marigold-lcm-v1-0',  
            ], {
               "default": 'Marigold'
            }),
            }
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("ensembled_image",)
    FUNCTION = "process"
    CATEGORY = "Marigold"
    DESCRIPTION = """
Diffusion-based monocular depth estimation:  
https://github.com/prs-eth/Marigold  

This node is experimental version that includes optical flow  
for video consistency between frames.  

- denoise_steps: steps per depth map, increase for accuracy in exchange of processing time
- n_repeat: amount of iterations to be ensembled into single depth map
- n_repeat_batch_size: how many of the n_repeats are processed as a batch,  
if you have the VRAM this can match the n_repeats for faster processing  
- model: Marigold or it's LCM version marigold-lcm-v1-0  
For the LCM model use around 4 steps and the LCMScheduler  
- scheduler: Different schedulers give bit different results  
- invert: marigold by default produces depth map where black is front,  
for controlnets etc. we want the opposite.  
- regularizer_strength, reduction_method, max_iter, tol (tolerance) are settings   
for the ensembling process, generally do not touch.  
"""

    def process(self, image, seed, first_frame_denoise_steps, denoise_steps, first_frame_n_repeat, keep_model_loaded, invert,
                n_repeat_batch_size, dtype, scheduler, normalize, flow_warping, flow_depth_mix, noise_ratio, model="Marigold"):
        batch_size = image.shape[0]

        precision = convert_dtype(dtype)
       
        device = model_management.get_torch_device()
        torch.manual_seed(seed)

        image = image.permute(0, 3, 1, 2).to(device).to(dtype=precision)
        if normalize:
            image = image * 2.0 - 1.0
        
        if flow_warping:
            from .marigold.util.flow_estimation import FlowEstimator
            flow_estimator = FlowEstimator(os.path.join(script_directory, "gmflow", "gmflow_things-e9887eda.pth"), device)
        diffusers_model_path = os.path.join(folder_paths.models_dir,'diffusers')
        if model == "Marigold":
            folders_to_check = [
                os.path.join(script_directory,"checkpoints","Marigold_v1_merged",),
                os.path.join(script_directory,"checkpoints","Marigold",),
                os.path.join(diffusers_model_path,"Marigold_v1_merged"),
                os.path.join(diffusers_model_path,"Marigold")
            ]
        elif model == "marigold-lcm-v1-0":
            folders_to_check = [
                os.path.join(diffusers_model_path,"marigold-lcm-v1-0"),
                os.path.join(diffusers_model_path,"checkpoints","marigold-lcm-v1-0")
            ]
        self.custom_config = {
            "model": model,
            "dtype": dtype,
            "scheduler": scheduler,
        }
        if not hasattr(self, 'marigold_pipeline') or self.marigold_pipeline is None or self.current_config != self.custom_config:
            self.current_config = self.custom_config
            # Load the model only if it hasn't been loaded before
            checkpoint_path = None
            for folder in folders_to_check:
                potential_path = os.path.join(script_directory, folder)
                if os.path.exists(potential_path):
                    checkpoint_path = potential_path
                    break
            to_ignore = ["*.bin", "*fp16*"]            
            if checkpoint_path is None:
                if model == "Marigold":
                    try:
                        from huggingface_hub import snapshot_download
                        checkpoint_path = os.path.join(diffusers_model_path, "Marigold")
                        snapshot_download(repo_id="Bingxin/Marigold", ignore_patterns=to_ignore, local_dir=checkpoint_path, local_dir_use_symlinks=False)  
                    except:
                        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_path}")
                if model == "marigold-lcm-v1-0":
                    try:
                        from huggingface_hub import snapshot_download
                        checkpoint_path = os.path.join(diffusers_model_path, "marigold-lcm-v1-0")
                        snapshot_download(repo_id="prs-eth/marigold-lcm-v1-0", ignore_patterns=to_ignore, local_dir=checkpoint_path, local_dir_use_symlinks=False)  
                    except:
                        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_path}")
            
            self.marigold_pipeline = MarigoldPipeline.from_pretrained(checkpoint_path, enable_xformers=False, empty_text_embed=empty_text_embed, noise_scheduler_type=scheduler)
            self.marigold_pipeline = self.marigold_pipeline.to(precision).to(device)
            self.marigold_pipeline.unet.eval()
        pbar = comfy.utils.ProgressBar(batch_size)

        out = []
        for i in range(batch_size):
            if flow_warping:
                current_image = image[i]
                prev_image = image[i-1]
                flow = flow_estimator.estimate_flow(prev_image.to(torch.float32), current_image.to(torch.float32))
            if i == 0 or not flow_warping:
                # Duplicate the current image n_repeat times
                duplicated_batch = image[i].unsqueeze(0).repeat(first_frame_n_repeat, 1, 1, 1)
                # Process the duplicated batch in sub-batches
                depth_maps = []
                for j in range(0, first_frame_n_repeat, n_repeat_batch_size):
                    # Get the current sub-batch
                    sub_batch = duplicated_batch[j:j + n_repeat_batch_size]
                    
                    # Process the sub-batch
                    depth_maps_sub_batch = self.marigold_pipeline(sub_batch, num_inference_steps=first_frame_denoise_steps, show_pbar=False)
                    
                    # Process each depth map in the sub-batch if necessary
                    for depth_map in depth_maps_sub_batch:
                        depth_map = torch.clip(depth_map, -1.0, 1.0)
                        depth_map = (depth_map + 1.0) / 2.0
                        depth_maps.append(depth_map)
                
                depth_predictions = torch.cat(depth_maps, dim=0).squeeze()
                
                del duplicated_batch, depth_maps_sub_batch
                comfy.model_management.soft_empty_cache()

                # Test-time ensembling
                if first_frame_n_repeat > 1:
                    depth_map, pred_uncert = ensemble_depths(
                        depth_predictions,
                        regularizer_strength=0.02,
                        max_iter=5,
                        tol=1e-3,
                        reduction="median",
                        max_res=None,
                        device=device,
                    )
                    prev_depth_map = torch.clip(depth_map, 0.0, 1.0)                
                    depth_map = depth_map.unsqueeze(2).repeat(1, 1, 3)
                    out.append(depth_map)
                    pbar.update(1)
                else:
                    prev_depth_map = torch.clip(depth_map[0], 0.0, 1.0)                
                    depth_map = depth_map[0].unsqueeze(2).repeat(1, 1, 3)
                    out.append(depth_map)
                    pbar.update(1)
                
            else:
                #idea and original implementation from https://github.com/pablodawson/Marigold-Video
                warped_depth_map = FlowEstimator.warp_with_flow(flow, prev_depth_map).to(precision).to(device)                   
                warped_depth_map = (warped_depth_map + 1.0) / 2.0
                assert warped_depth_map.min() >= -1.0 and warped_depth_map.max() <= 1.0
                depth_predictions = self.marigold_pipeline(current_image.unsqueeze(0), init_depth_latent=warped_depth_map.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0),
                                                            noise_ratio=noise_ratio, num_inference_steps=denoise_steps, show_pbar=False)
                
                warped_depth_map = warped_depth_map / warped_depth_map.max()
                depth_out = flow_depth_mix * depth_predictions + (1 - flow_depth_mix) * warped_depth_map
                depth_out = torch.clip(depth_out, 0.0, 1.0)

                prev_depth_map = depth_out.squeeze()
                depth_out = depth_out.squeeze().unsqueeze(2).repeat(1, 1, 3)
                
                out.append(depth_out)
                pbar.update(1)

                del depth_predictions, warped_depth_map
        if invert:
            outstack = 1.0 - torch.stack(out, dim=0).cpu().to(torch.float32)
        else:
            outstack = torch.stack(out, dim=0).cpu().to(torch.float32)
        if not keep_model_loaded:
            self.marigold_pipeline = None
            model_management.soft_empty_cache()
        return (outstack,)
    
class ColorizeDepthmap:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "colorize_method": (
            [   
                'Spectral',
                'terrain', 
                'viridis',
                'plasma',
                'inferno',
                'magma',
                'cividis',
                'twilight',
                'rainbow',
                'gist_rainbow',
                'gist_ncar',
                'gist_earth',
                'turbo',
                'jet',
                'afmhot',
                'copper',
                'seismic',
                'hsv',
                'brg',

            ], {
               "default": 'Spectral'
            }),
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "color"

    CATEGORY = "Marigold"

    def color(self, image, colorize_method):
        colored_images = []
        for i in range(image.shape[0]):  # Iterate over the batch dimension
            depth_map = image[i].squeeze().permute(2, 0, 1)
            depth_map = depth_map[0]
            depth_map = colorizedepth(depth_map, colorize_method)
            depth_map = torch.from_numpy(depth_map) / 255
            depth_map = depth_map.unsqueeze(0)
            colored_images.append(depth_map)
        
        # Stack the list of tensors along a new dimension
        colored_images = torch.cat(colored_images, dim=0)
        return (colored_images,)

import folder_paths

class SaveImageOpenEXR:
    def __init__(self):
        try:
            import OpenEXR
            import Imath
            self.OpenEXR = OpenEXR
            self.Imath = Imath
            self.use_openexr = True
        except ImportError:
            print("No OpenEXR module found, trying OpenCV...")
            self.use_openexr = False
            try:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                import cv2
                self.cv2 = cv2
            except ImportError:
                raise ImportError("No OpenEXR or OpenCV module found, can't save EXR")
        
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "images": ("IMAGE", ),
            "filename_prefix": ("STRING", {"default": "ComfyUI_EXR"})
            },
            
            }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES =("file_url",)
    FUNCTION = "saveexr"
    OUTPUT_NODE = True
    CATEGORY = "Marigold"

    def saveexr(self, images, filename_prefix):
        import re
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        def file_counter():
            max_counter = 0
            # Loop through the existing files
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = re.fullmatch(f"{filename}_(\d+)_?\.[a-zA-Z0-9]+", existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter
            return max_counter
        
        for image in images:
            # Ensure the tensor is on the CPU and convert it to a numpy array
            image_np = image.cpu().numpy()
            image_np = image_np.astype(np.float32)

            if self.use_openexr:
                # Assuming the image is in the format of floating point 32 bit (change PIXEL_TYPE if not)
                PIXEL_TYPE = self.Imath.PixelType(self.Imath.PixelType.FLOAT)
                height, width, channels = image_np.shape

                # Prepare the EXR header
                header = self.OpenEXR.Header(width, height)
                half_chan = self.Imath.Channel(PIXEL_TYPE)
                header['channels'] = dict([(c, half_chan) for c in "RGB"])

                # Split the channels for OpenEXR
                R = image_np[:, :, 0].tostring()
                G = image_np[:, :, 1].tostring()
                B = image_np[:, :, 2].tostring()

                # Increment the counter by 1 to get the next available value
                counter = file_counter() + 1
                file = f"{filename}_{counter:05}.exr"

                # Write the EXR file
                exr_file = self.OpenEXR.OutputFile(os.path.join(full_output_folder, file), header)
                exr_file.writePixels({'R': R, 'G': G, 'B': B})
                exr_file.close()
            else:            
                counter = file_counter() + 1
                file = f"{filename}_{counter:05}.exr"
                exr_file = os.path.join(full_output_folder, file)
                self.cv2.imwrite(exr_file, image_np)

        return (f"/view?filename={file}&subfolder=&type=output",)

class RemapDepth:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE",),
            "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
            "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
            "clamp": ("BOOLEAN", {"default": True}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remap"

    CATEGORY = "Marigold"
        
    def remap(self, image, min, max, clamp):
        if image.dtype == torch.float16:
            image = image.to(torch.float32)
        image = min + image * (max - min)
        if clamp:
            image = torch.clamp(image, min=0.0, max=1.0)
        return (image, )

NODE_CLASS_MAPPINGS = {
    "MarigoldDepthEstimation": MarigoldDepthEstimation,
    "MarigoldDepthEstimationVideo": MarigoldDepthEstimationVideo,
    "ColorizeDepthmap": ColorizeDepthmap,
    "SaveImageOpenEXR": SaveImageOpenEXR,
    "RemapDepth": RemapDepth
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldDepthEstimation": "MarigoldDepthEstimation",
    "MarigoldDepthEstimationVideo": "MarigoldDepthEstimationVideo",
    "ColorizeDepthmap": "ColorizeDepthmap",
    "SaveImageOpenEXR": "SaveImageOpenEXR",
    "RemapDepth": "RemapDepth"
}