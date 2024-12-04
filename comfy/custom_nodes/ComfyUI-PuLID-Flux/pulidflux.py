
import torch
from torch import nn, Tensor
from torchvision import transforms
from torchvision.transforms import functional
import os
import logging
import folder_paths
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
import comfy.model_management
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import logging
from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .encoders_flux import IDFormer, PerceiverAttentionCA

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

def forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
) -> Tensor:
    device = comfy.model_management.get_torch_device()
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    ca_idx = 0
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"])
                return out

            out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe}, {"original_block": block_wrap})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        if control is not None: # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

        # PuLID attention
        if self.pulid_data:
            if i % self.pulid_double_interval == 0:
                # Will calculate influence of all pulid nodes at once
                for _, node_data in self.pulid_data.items():
                    if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                        img = img + node_data['weight'] * self.pulid_ca[ca_idx].to(device)(node_data['embedding'], img)
                ca_idx += 1

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"])
                return out

            out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe}, {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe)

        if control is not None: # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

        # PuLID attention
        if self.pulid_data:
            real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
            if i % self.pulid_single_interval == 0:
                # Will calculate influence of all nodes at once
                for _, node_data in self.pulid_data.items():
                    if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                        real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx].to(device)(node_data['embedding'], real_img)
                ca_idx += 1
            img = torch.cat((txt, real_img), 1)

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img

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

        # 显式释放显存
        torch.cuda.empty_cache()

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

        # 显式释放显存
        torch.cuda.empty_cache()

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
        #global CLIP_DIR
        from .eva_clip.factory import create_model_and_transforms

        with torch.no_grad():
            model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True, cache_dir=CLIP_DIR)

            model = model.visual

            eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
            eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
            if not isinstance(eva_transform_mean, (list, tuple)):
                model["image_mean"] = (eva_transform_mean,) * 3
            if not isinstance(eva_transform_std, (list, tuple)):
                model["image_std"] = (eva_transform_std,) * 3

        # 显式释放显存
        torch.cuda.empty_cache()

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
                "source_face_selection": (["largest_face","center_face"], {"default": "center_face"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "face_used")
    FUNCTION = "apply_pulid_flux"
    CATEGORY = "pulid"

    def __init__(self):
        self.pulid_data_dict = None

    def apply_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, image, weight, start_at, end_at, attn_mask=None, unique_id=None, source_face_selection="center_face"):
        device = comfy.model_management.get_torch_device()
        dtype = model.model.diffusion_model.dtype

        with torch.no_grad():
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
            align_faces = []

            for i in range(image.shape[0]):
                iface_embeds = None
                for size in [(size, size) for size in range(640, 256, -64)]:
                    face_analysis.det_model.input_size = size
                    face_info = face_analysis.get(image[i])
                    if face_info:
                        face_info = sorted(face_info, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                        iface_embeds = torch.from_numpy(face_info.embedding).unsqueeze(0).to(device, dtype=dtype)
                        break
                else:
                    logging.warning(f'Warning: No face detected in image {str(i)}')
                    continue

                face_helper.clean_all()
                face_helper.read_image(image[i])
                if source_face_selection == "largest_face":
                    face_helper.get_face_landmarks_5(only_keep_largest=True)
                else:
                    face_helper.get_face_landmarks_5(only_center_face=True)
                face_helper.align_warp_face()

                if len(face_helper.cropped_faces) == 0:
                    continue

                align_face = face_helper.cropped_faces[0]
                align_face = image_to_tensor(align_face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                parsing_out = face_helper.face_parse(functional.normalize(align_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
                parsing_out = parsing_out.argmax(dim=1, keepdim=True)
                bg = sum(parsing_out == i for i in bg_label).bool()
                white_image = torch.ones_like(align_face)
                face_features_image = torch.where(bg, white_image, to_gray(align_face))

                face_features_image = functional.resize(face_features_image, eva_clip.image_size, transforms.InterpolationMode.BICUBIC if 'cuda' in device.type else transforms.InterpolationMode.NEAREST).to(device, dtype=dtype)
                face_features_image = functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)

                id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
                id_cond_vit = id_cond_vit.to(device, dtype=dtype)
                for idx in range(len(id_vit_hidden)):
                    id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

                id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

                id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)

                cond.append(pulid_flux.get_embeds(id_cond, id_vit_hidden))
                align_faces.append(align_face.permute(0, 2, 3, 1))

            if not cond:
                logging.warning("PuLID warning: No faces detected in any of the given images, returning unmodified model.")
                return (model,)

            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                cond = torch.mean(cond, dim=0, keepdim=True)

            sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
            sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

            flux_model = model.model.diffusion_model
            if not hasattr(flux_model, "pulid_ca"):
                flux_model.pulid_ca = pulid_flux.pulid_ca
                flux_model.pulid_double_interval = pulid_flux.double_interval
                flux_model.pulid_single_interval = pulid_flux.single_interval
                flux_model.pulid_data = {}
                new_method = forward_orig.__get__(flux_model, flux_model.__class__)
                setattr(flux_model, 'forward_orig', new_method)

            flux_model.pulid_data[unique_id] = {
                'weight': weight,
                'embedding': cond,
                'sigma_start': sigma_start,
                'sigma_end': sigma_end,
            }

            self.pulid_data_dict = {'data': flux_model.pulid_data, 'unique_id': unique_id}

            align_faces = torch.cat(align_faces)
            
            device = torch.device("cpu")
            eva_clip.to(device, dtype=dtype)
            pulid_flux.to(device, dtype=dtype)
            torch.cuda.empty_cache()
            del eva_clip
            del pulid_flux
            return (model, align_faces)


    def __del__(self):
        # Destroy the data for this node
        if self.pulid_data_dict:
            del self.pulid_data_dict['data'][self.pulid_data_dict['unique_id']]
            del self.pulid_data_dict


NODE_CLASS_MAPPINGS = {
    "PulidFluxModelLoader": PulidFluxModelLoader,
    "PulidFluxInsightFaceLoader": PulidFluxInsightFaceLoader,
    "PulidFluxEvaClipLoader": PulidFluxEvaClipLoader,
    "ApplyPulidFlux": ApplyPulidFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulidFluxModelLoader": "Load PuLID Flux Model",
    "PulidFluxInsightFaceLoader": "Load InsightFace (PuLID Flux)",
    "PulidFluxEvaClipLoader": "Load Eva Clip (PuLID Flux)",
    "ApplyPulidFlux": "Apply PuLID Flux",
}



