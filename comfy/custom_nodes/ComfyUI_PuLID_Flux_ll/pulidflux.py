import types
import zipfile

import cv2
import torch
from insightface.utils.download import download_file
from insightface.utils.storage import BASE_REPO_URL
from insightface.utils import face_align
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional
import os
import logging
import folder_paths
import comfy
from insightface.app import FaceAnalysis
from .face_restoration_helper import FaceRestoreHelper, get_face_by_index, draw_on

from comfy import model_management
from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .encoders_flux import IDFormer, PerceiverAttentionCA

from .PulidFluxHook import pulid_forward_orig, set_model_dit_patch_replace, pulid_enter, pulid_patch_double_blocks_after
from .patch_util import PatchKeys, add_model_patch_option, set_model_patch


def set_extra_config_model_path(extra_config_models_dir_key, models_dir_name:str):
    models_dir_default = os.path.join(folder_paths.models_dir, models_dir_name)
    if extra_config_models_dir_key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[extra_config_models_dir_key] = (
            [os.path.join(folder_paths.models_dir, models_dir_name)], folder_paths.supported_pt_extensions)
    else:
        if not os.path.exists(models_dir_default):
            os.makedirs(models_dir_default, exist_ok=True)
        folder_paths.add_model_folder_path(extra_config_models_dir_key, models_dir_default, is_default=True)

set_extra_config_model_path("pulid", "pulid")
set_extra_config_model_path("insightface", "insightface")
#set_extra_config_model_path("facexlib", "facexlib")

INSIGHTFACE_DIR = folder_paths.get_folder_paths("insightface")[0]
FACEXLIB_DIR = folder_paths.get_folder_paths("controlnet")[0]

# MODELS_DIR = os.path.join(folder_paths.models_dir, "pulid")
# if "pulid" not in folder_paths.folder_names_and_paths:
#     current_paths = [MODELS_DIR]
# else:
#     current_paths, _ = folder_paths.folder_names_and_paths["pulid"]
# folder_paths.folder_names_and_paths["pulid"] = (current_paths, folder_paths.supported_pt_extensions)

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
        offload_device = model_management.unet_offload_device()
        load_device = model_management.get_torch_device()

        model = PulidFluxModel()

        logging.info("Loading PuLID-Flux model.")
        model.from_pretrained(path=model_path)

        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        del model

        return (model_patcher,)

def download_insightface_model(sub_dir, name, force=False, root='~/.insightface'):
    # Copied and modified from insightface.utils.storage.download
    # Solve https://github.com/deepinsight/insightface/issues/2711
    _root = os.path.expanduser(root)
    dir_path = os.path.join(_root, sub_dir, name)
    if os.path.exists(dir_path) and not force:
        return dir_path
    print('download_path:', dir_path)
    zip_file_path = os.path.join(_root, sub_dir, name + '.zip')
    model_url = "%s/%s.zip"%(BASE_REPO_URL, name)
    download_file(model_url,
             path=zip_file_path,
             overwrite=True)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # zip file has contains ${name}
    real_dir_path = os.path.join(_root, sub_dir)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(real_dir_path)
    #os.remove(zip_file_path)
    return dir_path

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
        name = "antelopev2"
        download_insightface_model("models", name, root=INSIGHTFACE_DIR)
        model = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider', ]) # alternative to buffalo_l
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

        clip_file_path = folder_paths.get_full_path("text_encoders", 'EVA02_CLIP_L_336_psz14_s6B.pt')
        if clip_file_path is None:
            clip_dir = os.path.join(folder_paths.models_dir, "clip")
        else:
            clip_dir = os.path.dirname(clip_file_path)
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True, local_dir=clip_dir)

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
                "options": ("OPTIONS",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_pulid_flux"
    CATEGORY = "pulid"

    def apply_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, image, weight, start_at, end_at, attn_mask=None, options={}, unique_id=None):
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
        pulid_flux.model.to(dtype=dtype)
        model_management.load_models_gpu([pulid_flux], force_full_load=True)
        # model_management.load_model_gpu(pulid_flux)

        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            # attn_mask = attn_mask.to(device, dtype=dtype)

        image = tensor_to_image(image)

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            parsing_model='bisenet',
            save_ext='png',
            device=device,
            model_rootpath=FACEXLIB_DIR
        )

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []

        input_face_sort = options.get('input_faces_order', "large-small")
        input_face_index = options.get('input_faces_index', 0)
        input_face_align_mode = options.get('input_faces_align_mode', 1)
        # Analyse multiple images at multiple sizes and combine largest area embeddings
        for i in range(image.shape[0]):
            # get insightface embeddings
            bboxes = []
            iface_embeds = None
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face_info = face_analysis.get(image[i])
                if face_info:
                    face_info, index, sorted_faces = get_face_by_index(face_info, face_sort_rule=input_face_sort, face_index=input_face_index)
                    bboxes = [face.bbox for face in sorted_faces]
                    iface_embeds = torch.from_numpy(face_info.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                # No face detected, skip this image
                logging.warning(f'Warning: No face detected in image {str(i)}')
                continue

            if input_face_align_mode == 1:
                image_size = 512
                M = face_align.estimate_norm(face_info.kps, image_size=image_size)
                align_face = cv2.warpAffine(image[i], M, (image_size, image_size), borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(135, 133, 132))
                # align_face = face_align.norm_crop(image[i], landmark=face_info.kps, image_size=image_size)
                del M
            else:
                # get eva_clip embeddings
                face_helper.clean_all()
                face_helper.read_image(image[i])
                face_helper.get_face_landmarks_5(ref_sort_bboxes=bboxes, face_index=input_face_index)
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
            cond.append(pulid_flux.model.get_embeds(id_cond, id_vit_hidden))

        eva_clip.to(torch.device('cpu'))
        if not cond:
            # No faces detected, return the original model
            logging.warning("PuLID warning: No faces detected in any of the given images, returning unmodified model.")
            del eva_clip, face_analysis, pulid_flux, face_helper, attn_mask
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
            if i % pulid_flux.model.double_interval == 0:
                patch_kwargs["ca_idx"] = ca_idx
                set_model_dit_patch_replace(model, patch_kwargs, ("double_block", i))
                ca_idx += 1
        for i in range(38):
            if i % pulid_flux.model.single_interval == 0:
                patch_kwargs["ca_idx"] = ca_idx
                set_model_dit_patch_replace(model, patch_kwargs, ("single_block", i))
                ca_idx += 1

        if len(model.get_additional_models_with_key("pulid_flux_model_patcher")) == 0:
            model.set_additional_models("pulid_flux_model_patcher", [pulid_flux])

        if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name)) == 0:
            # Just add it once when connecting in series
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name, pulid_outer_sample_wrappers_with_override)
        if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name)) == 0:
            # Just add it once when connecting in series
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name, pulid_apply_model_wrappers)

        del eva_clip, face_analysis, pulid_flux, face_helper, attn_mask
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


class PulidFluxOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_faces_order": (
                    ["left-right","right-left","top-bottom","bottom-top","small-large","large-small"],
                    {
                        "default": "large-small",
                        "tooltip": "left-right: Sort the left boundary of bbox by column from left to right.\n"
                                   "right-left: Reverse order of left-right (Sort the left boundary of bbox by column from right to left).\n"
                                   "top-bottom: Sort the top boundary of bbox by row from top to bottom.\n"
                                   "bottom-top: Reverse order of top-bottom (Sort the top boundary of bbox by row from bottom to top).\n"
                                   "small-large: Sort the area of bbox from small to large.\n"
                                   "large-small: Sort the area of bbox from large to small."
                    }
                ),
                "input_faces_index": ("INT",
                                      {
                                          "default": 0, "min": 0, "max": 1000, "step": 1,
                                          "tooltip": "If the value is greater than the size of bboxes, will set value to 0."
                                      }),
                "input_faces_align_mode": ("INT",
                                      {
                                          "default": 1, "min": 0, "max": 1, "step": 1,
                                          "tooltip": "Align face mode.\n"
                                                     "0: align_face and embed_face use different detectors. The results maybe different.\n"
                                                     "1: align_face and embed_face use the same detector."
                                      }),
            }
        }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "execute"
    CATEGORY = "pulid"

    def execute(self,input_faces_order, input_faces_index, input_faces_align_mode=1):
        options: dict = {
            "input_faces_order": input_faces_order,
            "input_faces_index": input_faces_index,
            "input_faces_align_mode": input_faces_align_mode,
        }
        return (options, )


class PulidFluxFaceDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE",),
                "options": ("OPTIONS",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("embed_face", "align_face", "face_bbox_image",)
    FUNCTION = "execute"
    CATEGORY = "pulid"
    OUTPUT_IS_LIST = (True, True, True,)

    def execute(self, face_analysis, image, options):

        device = comfy.model_management.get_torch_device()

        input_face_sort = options.get('input_faces_order', "large-small")
        input_face_index = options.get('input_faces_index', 0)
        input_face_align_mode = options.get('input_faces_align_mode', 1)

        if input_face_align_mode == 0:
            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                parsing_model='bisenet',
                save_ext='png',
                device=device,
                model_rootpath=FACEXLIB_DIR
            )

        # Analyse multiple images at multiple sizes and combine largest area embeddings
        embed_faces=[]
        align_faces=[]
        draw_embed_face_bbox=[]
        image = tensor_to_image(image)
        for i in range(image.shape[0]):
            bboxes = []
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face_info = face_analysis.get(image[i])
                if face_info:
                    face_info, index, sorted_faces = get_face_by_index(face_info, face_sort_rule=input_face_sort,
                                                         face_index=input_face_index)
                    bboxes = [face.bbox for face in sorted_faces]
                    embed_faces.append(crop_image(image[i], face_info.bbox, margin=10))
                    draw_embed_face_bbox.append(image_to_tensor(draw_on(image[i], sorted_faces)).unsqueeze(0))
                    break
            else:
                # No face detected, skip this image
                logging.warning(f'Warning: No face detected in image {str(i)}')
                continue

            if input_face_align_mode == 1:
                image_size = 512
                M = face_align.estimate_norm(face_info.kps, image_size=image_size)
                align_face = cv2.warpAffine(image[i], M, (image_size, image_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
                # align_face = face_align.norm_crop(image[i], landmark=face_info.kps, image_size=image_size)
                del M
            else:
                # get eva_clip embeddings
                face_helper.clean_all()
                face_helper.read_image(image[i])
                face_helper.get_face_landmarks_5(ref_sort_bboxes=bboxes, face_index=input_face_index)
                face_helper.align_warp_face()

                if len(face_helper.cropped_faces) == 0:
                    # No face detected, skip this image
                    continue

                # Get aligned face image
                align_face = face_helper.cropped_faces[0]
                del face_helper
            align_faces.append(image_to_tensor(align_face).unsqueeze(0))
            del bboxes, align_face
        del image
        if len(embed_faces) == 0:
            # No face detected, skip this image
            logging.warning(f'Warning: No embed face detected in image')
        if  len(align_faces) == 0:
            logging.warning(f'Warning: No align face detected in image')
        return embed_faces, align_faces, draw_embed_face_bbox,


def crop_image(image, bbox, margin=0):
    if len(image.shape) == 3:
        image = image[None, ...]
    image = image_to_tensor(image)
    x, y, x1, y1 = bbox.astype(int)
    w = x1 - x
    h = y1 - y
    image_height = image.shape[1]
    image_width = image.shape[2]
    # 左上角坐标
    x = min(x, image_width)
    y = min(y, image_height)
    # 右下角坐标
    to_x = min(w + x + margin, image_width)
    to_y = min(h + y + margin, image_height)
    # 防止越界
    x = max(0, x - margin)
    y = max(0, y - margin)
    to_x = max(0, to_x)
    to_y = max(0, to_y)
    # 按区域截取图片
    crop_img = image[:, y:to_y, x:to_x, :]
    return crop_img


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
        del diffusion_model, cfg_guider

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
        del PULID_model_patch['timesteps'], base_model

    return out

NODE_CLASS_MAPPINGS = {
    "PulidFluxModelLoader": PulidFluxModelLoader,
    "PulidFluxInsightFaceLoader": PulidFluxInsightFaceLoader,
    "PulidFluxEvaClipLoader": PulidFluxEvaClipLoader,
    "ApplyPulidFlux": ApplyPulidFlux,
    "FixPulidFluxPatch": FixPulidFluxPatch,
    "PulidFluxOptions": PulidFluxOptions,
    "PulidFluxFaceDetector": PulidFluxFaceDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulidFluxModelLoader": "Load PuLID Flux Model",
    "PulidFluxInsightFaceLoader": "Load InsightFace (PuLID Flux)",
    "PulidFluxEvaClipLoader": "Load Eva Clip (PuLID Flux)",
    "ApplyPulidFlux": "Apply PuLID Flux",
    "FixPulidFluxPatch": "Fix PuLID Flux Patch",
    "PulidFluxOptions": "Pulid Flux Options",
    "PulidFluxFaceDetector": "Pulid Flux Face Detector",
}
