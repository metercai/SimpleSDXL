import types

import torch
from torch import nn, Tensor
from torchvision import transforms
from torchvision.transforms import functional
import os
import logging
import folder_paths
import comfy
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .encoders_flux import IDFormer, PerceiverAttentionCA

from .PulidFluxHook import pulid_forward_orig, set_model_dit_patch_replace, pulid_enter, pulid_patch_double_blocks_after
from .patch_util import PatchKeys, add_model_patch_option, set_model_patch

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

MODELS_DIR = os.path.join(folder_paths.models_dir, "pulid")
CLIP_DIR = os.path.join(folder_paths.models_dir, "clip")
CONTROLNET_DIR = os.path.join(folder_paths.models_dir, "controlnet")

if "pulid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["pulid"]
folder_paths.folder_names_and_paths["pulid"] = (current_paths, folder_paths.supported_pt_extensions)

class PulidFluxModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.double_interval = 2
        self.single_interval = 4

        # Init encoder
        self.pulid_encoder = IDFormer()

        # Init attention
        num_ca = 19 // self.double_interval + 38 // self.single_interval
        if 19 % self.double_interval != 0:
            num_ca += 1
        if 38 % self.single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA() for _ in range(num_ca)
        ])

    def from_pretrained(self, path: str):
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

    def get_embeds(self, face_embed, clip_embeds):
        return self.pulid_encoder(face_embed, clip_embeds)

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor

def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

wrappers_name = "PULID_wrappers"

class PulidFluxModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pulid_file": (folder_paths.get_filename_list("pulid"), )}}

    RETURN_TYPES = ("PULIDFLUX",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    def load_model(self, pulid_file):
        model_path = folder_paths.get_full_path("pulid", pulid_file)

        # Also initialize the model, takes longer to load but then it doesn't have to be done every time you change parameters in the apply node
        model = PulidFluxModel()

        logging.info("Loading PuLID-Flux model.")
        model.from_pretrained(path=model_path)

        return (model,)

class PulidFluxInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "pulid"

    def load_insightface(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)

class PulidFluxEvaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_eva_clip"
    CATEGORY = "pulid"

    def load_eva_clip(self):
        from .eva_clip.factory import create_model_and_transforms

        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True, cache_dir=CLIP_DIR)

        model = model.visual

        eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            model["image_mean"] = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            model["image_std"] = (eva_transform_std,) * 3

        return (model,)

class ApplyPulidFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid_flux": ("PULIDFLUX", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_mask": ("MASK", ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_pulid_flux"
    CATEGORY = "pulid"

    def apply_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, image, weight, start_at, end_at, attn_mask=None, unique_id=None):
        model = model.clone()

        device = comfy.model_management.get_torch_device()
        # Why should I care what args say, when the unet model has a different dtype?!
        # Am I missing something?!
        #dtype = comfy.model_management.unet_dtype()
        dtype = model.model.diffusion_model.dtype
        # Because of 8bit models we must check what cast type does the unet uses
        # ZLUDA (Intel, AMD) & GPUs with compute capability < 8.0 don't support bfloat16 etc.
        # Issue: https://github.com/balazik/ComfyUI-PuLID-Flux/issues/6
        if model.model.manual_cast_dtype is not None:
            dtype = model.model.manual_cast_dtype

        eva_clip.to(device, dtype=dtype)
        pulid_flux.to(device, dtype=dtype)

        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device, dtype=dtype)

        image = tensor_to_image(image)

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
            model_rootpath=CONTROLNET_DIR,
        )

        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device, model_rootpath=CONTROLNET_DIR)

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []

        # Analyse multiple images at multiple sizes and combine largest area embeddings
        for i in range(image.shape[0]):
            # get insightface embeddings
            iface_embeds = None
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face_info = face_analysis.get(image[i])
                if face_info:
                    # Only use the maximum face
                    # Removed the reverse=True from original code because we need the largest area not the smallest one!
                    # Sorts the list in ascending order (smallest to largest),
                    # then selects the last element, which is the largest face
                    face_info = sorted(face_info, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    iface_embeds = torch.from_numpy(face_info.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                # No face detected, skip this image
                logging.warning(f'Warning: No face detected in image {str(i)}')
                continue

            # get eva_clip embeddings
            face_helper.clean_all()
            face_helper.read_image(image[i])
            face_helper.get_face_landmarks_5(only_keep_largest=True)
            face_helper.align_warp_face()

            if len(face_helper.cropped_faces) == 0:
                # No face detected, skip this image
                continue

            # Get aligned face image
            align_face = face_helper.cropped_faces[0]
            # Convert bgr face image to tensor
            align_face = image_to_tensor(align_face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            parsing_out = face_helper.face_parse(functional.normalize(align_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(align_face)
            # Only keep the face features
            face_features_image = torch.where(bg, white_image, to_gray(align_face))

            # Transform img before sending to eva_clip
            # Apparently MPS only supports NEAREST interpolation?
            face_features_image = functional.resize(face_features_image, eva_clip.image_size, transforms.InterpolationMode.BICUBIC if 'cuda' in device.type else transforms.InterpolationMode.NEAREST).to(device, dtype=dtype)
            face_features_image = functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)

            # eva_clip
            id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            # Combine embeddings
            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)

            # Pulid_encoder
            cond.append(pulid_flux.get_embeds(id_cond, id_vit_hidden))

        if not cond:
            # No faces detected, return the original model
            logging.warning("PuLID warning: No faces detected in any of the given images, returning unmodified model.")
            return (model,)

        # average embeddings
        cond = torch.cat(cond).to(device, dtype=dtype)
        if cond.shape[0] > 1:
            cond = torch.mean(cond, dim=0, keepdim=True)

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        patch_kwargs = {
            "pulid_model": pulid_flux,
            "weight": weight,
            "embedding": cond,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "mask": attn_mask
        }

        ca_idx = 0
        for i in range(19):
            if i % pulid_flux.double_interval == 0:
                patch_kwargs["ca_idx"] = ca_idx
                set_model_dit_patch_replace(model, patch_kwargs, ("double_block", i))
                ca_idx += 1
        for i in range(38):
            if i % pulid_flux.single_interval == 0:
                patch_kwargs["ca_idx"] = ca_idx
                set_model_dit_patch_replace(model, patch_kwargs, ("single_block", i))
                ca_idx += 1

        if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name)) == 0:
            # Just add it once when connecting in series
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name, pulid_outer_sample_wrappers_with_override)
        if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name)) == 0:
            # Just add it once when connecting in series
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name, pulid_apply_model_wrappers)

        device = torch.device("cpu")
        eva_clip.to(device, dtype=dtype)
        pulid_flux.to(device, dtype=dtype)
        del eva_clip, face_analysis, pulid_flux
        torch.cuda.empty_cache()

        return (model,)


class FixPulidFluxPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "fix_pulid_patch"
    CATEGORY = "pulid"

    def fix_pulid_patch(self, model):
        model = model.clone()

        if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name)) > 0:
            model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name)

        if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name)) > 0:
            model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name)
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name, pulid_outer_sample_wrappers)

            set_model_patch(model, PatchKeys.options_key, pulid_enter, PatchKeys.dit_enter)
            set_model_patch(model, PatchKeys.options_key, pulid_patch_double_blocks_after, PatchKeys.dit_double_blocks_after)

        return (model,)


def set_hook(diffusion_model, target_forward_orig):
    # comfy.ldm.flux.model.Flux.old_forward_orig_for_pulid = comfy.ldm.flux.model.Flux.forward_orig
    # comfy.ldm.flux.model.Flux.forward_orig = pulid_forward_orig
    diffusion_model.old_forward_orig_for_pulid = diffusion_model.forward_orig
    diffusion_model.forward_orig = types.MethodType(target_forward_orig, diffusion_model)

def clean_hook(diffusion_model):
    # if hasattr(comfy.ldm.flux.model.Flux, 'old_forward_orig_for_pulid'):
    #     comfy.ldm.flux.model.Flux.forward_orig = comfy.ldm.flux.model.Flux.old_forward_orig_for_pulid
    #     del comfy.ldm.flux.model.Flux.old_forward_orig_for_pulid
    if hasattr(diffusion_model, 'old_forward_orig_for_pulid'):
        diffusion_model.forward_orig = diffusion_model.old_forward_orig_for_pulid
        del diffusion_model.old_forward_orig_for_pulid

def pulid_outer_sample_wrappers_with_override(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = wrapper_executor.class_obj
    PULID_model_patch = add_model_patch_option(cfg_guider, PatchKeys.pulid_patch_key_attrs)
    PULID_model_patch['latent_image_shape'] = latent_image.shape

    diffusion_model = cfg_guider.model_patcher.model.diffusion_model
    set_hook(diffusion_model, pulid_forward_orig)
    try :
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
    finally:
        del PULID_model_patch['latent_image_shape']
        clean_hook(diffusion_model)

    return out

def pulid_outer_sample_wrappers(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = wrapper_executor.class_obj
    PULID_model_patch = add_model_patch_option(cfg_guider, PatchKeys.pulid_patch_key_attrs)
    PULID_model_patch['latent_image_shape'] = latent_image.shape
    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
    finally:
        del PULID_model_patch['latent_image_shape']

    return out

def pulid_apply_model_wrappers(wrapper_executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
    base_model = wrapper_executor.class_obj
    PULID_model_patch = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    PULID_model_patch['timesteps'] = base_model.model_sampling.timestep(t).float()
    try:
        transformer_options[PatchKeys.running_net_model] = base_model.diffusion_model
        out = wrapper_executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    finally:
        if PatchKeys.running_net_model in transformer_options:
            del transformer_options[PatchKeys.running_net_model]
        del PULID_model_patch['timesteps']

    return out

NODE_CLASS_MAPPINGS = {
    "PulidFluxModelLoader": PulidFluxModelLoader,
    "PulidFluxInsightFaceLoader": PulidFluxInsightFaceLoader,
    "PulidFluxEvaClipLoader": PulidFluxEvaClipLoader,
    "ApplyPulidFlux": ApplyPulidFlux,
    "FixPulidFluxPatch": FixPulidFluxPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulidFluxModelLoader": "Load PuLID Flux Model",
    "PulidFluxInsightFaceLoader": "Load InsightFace (PuLID Flux)",
    "PulidFluxEvaClipLoader": "Load Eva Clip (PuLID Flux)",
    "ApplyPulidFlux": "Apply PuLID Flux",
    "FixPulidFluxPatch": "Fix PuLID Flux Patch",
}
