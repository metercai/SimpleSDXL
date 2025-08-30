import os
import torch
import gc
from ..utils import log, dict_to_device
import numpy as np
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__)) 
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

local_model_path = os.path.join(folder_paths.models_dir, "nlf", "nlf_l_multi_0.3.2.torchscript")

from .motion4d import SMPL_VQVAE, VectorQuantizer, Encoder, Decoder
from .mtv import prepare_motion_embeddings

class DownloadAndLoadNLFModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": (
                    [
                    "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript"
                    ],
                )
             },
        }

    RETURN_TYPES = ("NLFMODEL",)
    RETURN_NAMES = ("nlf_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, url):
        
        if not os.path.exists(local_model_path):
            log.info(f"Downloading NLF model to: {local_model_path}")
            import requests
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            response = requests.get(url)
            if response.status_code == 200:
                with open(local_model_path, "wb") as f:
                    f.write(response.content)
            else:
                print("Failed to download file:", response.status_code)

        model = torch.jit.load(local_model_path).eval()

        return (model,)

class LoadNLFModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": local_model_path}),
            },
        }

    RETURN_TYPES = ("NLFMODEL",)
    RETURN_NAMES = ("nlf_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, path):
        model = torch.jit.load(path).eval()

        return model,

class LoadVQVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
        }

    RETURN_TYPES = ("VQVAE",)
    RETURN_NAMES = ("vqvae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model_name):
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        # Get motion tokenizer
        motion_encoder = Encoder(
            in_channels=3,
            mid_channels=[128, 512],
            out_channels=3072,
            downsample_time=[2, 2],
            downsample_joint=[1, 1]
        )
        motion_quant = VectorQuantizer(nb_code=8192, code_dim=3072)
        motion_decoder = Decoder(
            in_channels=3072,
            mid_channels=[512, 128],
            out_channels=3,
            upsample_rate=2.0,
            frame_upsample_rate=[2.0, 2.0],
            joint_upsample_rate=[1.0, 1.0]
        )
     
        vqvae = SMPL_VQVAE(motion_encoder, motion_decoder, motion_quant).to(device)
        vqvae.load_state_dict(vae_sd, strict=True)

        return vqvae,

class MTVCrafterEncodePoses:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vqvae": ("VQVAE", {"tooltip": "VQVAE model"}),
                "poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            },
        }

    RETURN_TYPES = ("MTVCRAFTERMOTION", "NLFPRED")
    RETURN_NAMES = ("mtvcrafter_motion", "pose_results")
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, vqvae, poses):

        # import pickle
        # with open(os.path.join(script_directory, "data", "sampled_data.pkl"), 'rb') as f:
        #     data_list = pickle.load(f)
        # if not isinstance(data_list, list):
        #     data_list = [data_list]
        # print(data_list)

        # smpl_poses = data_list[1]['pose']

        global_mean = np.load(os.path.join(script_directory, "data", "mean.npy")) #global_mean.shape: (24, 3)
        global_std = np.load(os.path.join(script_directory, "data", "std.npy"))

        smpl_poses = []
        for pose in poses['joints3d_nonparam'][0]:
            smpl_poses.append(pose[0].cpu().numpy())
        smpl_poses = np.array(smpl_poses)

        norm_poses = torch.tensor((smpl_poses - global_mean) / global_std).unsqueeze(0)
        print(f"norm_poses shape: {norm_poses.shape}, dtype: {norm_poses.dtype}")

        vqvae.to(device)
        motion_tokens, vq_loss = vqvae(norm_poses.to(device), return_vq=True)
        
        recon_motion = vqvae(norm_poses.to(device))[0][0].to(dtype=torch.float32).cpu().detach() * global_std + global_mean
        vqvae.to(offload_device)

        poses_dict = {
            'mtv_motion_tokens': motion_tokens,
            'global_mean': global_mean,
            'global_std': global_std
        }
       
        return poses_dict, recon_motion


class NLFPredict:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("NLFMODEL",),
            "images": ("IMAGE", {"tooltip": "Input images for the model"}),
            },
        }

    RETURN_TYPES = ("NLFPRED", )
    RETURN_NAMES = ("pose_results",)
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def predict(self, model, images):
        
        model.to(device)
        pred = model.detect_smpl_batched(images.permute(0, 3, 1, 2).to(device))
        model.to(offload_device)

        pred = dict_to_device(pred, offload_device)

        pose_results = {
            'joints3d_nonparam': [],
        }
        # Collect pose data
        for key in pose_results.keys():
            if key in pred:
                pose_results[key].append(pred[key])
            else:
                pose_results[key].append(None)
       
        return (pose_results,)

class DrawNLFPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def predict(self, poses, width, height):
        from .draw_pose import get_control_conditions
        print(type(poses))
        if isinstance(poses, dict):
            pose_input = poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in poses else poses
        else:
            pose_input = poses
        control_conditions = get_control_conditions(pose_input, height, width)

        return (control_conditions,)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadNLFModel": DownloadAndLoadNLFModel,
    "NLFPredict": NLFPredict,
    "DrawNLFPoses": DrawNLFPoses,
    "LoadVQVAE": LoadVQVAE,
    "MTVCrafterEncodePoses": MTVCrafterEncodePoses
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadNLFModel": "(Download)Load NLF Model",
    "NLFPredict": "NLF Predict",
    "DrawNLFPoses": "Draw NLF Poses",
    "LoadVQVAE": "Load VQVAE",
    "MTVCrafterEncodePoses": "MTV Crafter Encode Poses"
}
