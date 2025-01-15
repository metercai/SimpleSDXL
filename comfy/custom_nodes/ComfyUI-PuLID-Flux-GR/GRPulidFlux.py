import comfy.model_management as model_management
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
from .face_restoration_helper import FaceRestoreHelper
import gc
import torch.nn.functional as F
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

from .online_train2 import online_train

class GRPulidFluxModel(nn.Module):
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
    attn_mask: Tensor = None,  # Add attn_mask here
    *args,
    **kwargs
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

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    ca_idx = 0
    blocks_replace = patches_replace.get("dit", {})

    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"])
                return out

            out = blocks_replace[("double_block", i)](
                {"img": img, "txt": txt, "vec": vec, "pe": pe}, {"original_block": block_wrap})
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
                    condition_start = node_data['sigma_start'] >= timesteps
                    condition_end = timesteps >= node_data['sigma_end']
                    condition = torch.logical_and(
                        condition_start, condition_end).all()
                    
                    if condition:
                        img = img + node_data['weight'] * self.pulid_ca[ca_idx].to(device)(node_data['embedding'], img)
            
                ca_idx += 1
    
    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
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
                    condition_start = node_data['sigma_start'] >= timesteps
                    condition_end = timesteps >= node_data['sigma_end']

                    # Combine conditions and reduce to a single boolean
                    condition = torch.logical_and(condition_start, condition_end).all()

                    if condition:
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

def resize_with_pad(img, target_size): # image: 1, h, w, 3
    img = img.permute(0, 3, 1, 2)
    H, W = target_size
    
    h, w = img.shape[2], img.shape[3]
    scale_h = H / h
    scale_w = W / w
    scale = min(scale_h, scale_w)

    new_h = int(min(h * scale,H))
    new_w = int(min(w * scale,W))
    new_size = (new_h, new_w)
    
    img = F.interpolate(img, size=new_size, mode='bicubic', align_corners=False)
    
    pad_top = (H - new_h) // 2
    pad_bottom = (H - new_h) - pad_top
    pad_left = (W - new_w) // 2
    pad_right = (W - new_w) - pad_left
    img = F.pad(img, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return img.permute(0, 2, 3, 1)

def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x

class GRPulidFluxModelLoader:
    def __init__(self):
        pass
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pulid_file": (folder_paths.get_filename_list("pulid"), )}}

    RETURN_TYPES = ("PULIDFLUX",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    def load_model(self, pulid_file):
        model_path = folder_paths.get_full_path("pulid", pulid_file)

        # Also initialize the model, takes longer to load but then it doesn't have to be done every time you change parameters in the apply node
        self.model = GRPulidFluxModel()

        logging.info("Loading PuLID-Flux model.")
        self.model.from_pretrained(path=model_path)

        return (self.model,)

    def __del__(self):
        if self.model:
            logging.info("Cleaning up PuLID-Flux model resources...")
            try:
                if hasattr(self.model, "to") and callable(self.model.to):
                    logging.info("Moving model to CPU before cleanup.")
                    self.model.to("cpu")
                del self.model
            except Exception as e:
                logging.warning(f"Error while cleaning up model resources: {e}")
            finally:
                self.model = None
        gc.collect()
        if torch.cuda.is_available():
            logging.info("Clearing CUDA memory cache.")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class GRPulidFluxInsightFaceLoader:
    def __init__(self):
        pass
        self.model = None

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
        self.model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        self.model.prepare(ctx_id=0, det_size=(640, 640))

        return (self.model,)

    def cleanup(self):
        if self.model:
            logging.info("Cleaning up InsightFace model resources...")
            try:
                if hasattr(self.model, "to") and callable(self.model.to):
                    logging.info("Moving model to CPU before cleanup (if applicable).")
                    self.model.to("cpu")
                del self.model
            except Exception as e:
                logging.warning(f"Error during manual cleanup of InsightFace model: {e}")
            finally:
                self.model = None
        gc.collect()
        if torch.cuda.is_available():
            logging.info("Clearing CUDA memory cache.")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def __del__(self):
        if self.model:
            logging.info("Automatically cleaning up InsightFace model resources...")
            try:
                if hasattr(self.model, "to") and callable(self.model.to):
                    logging.info("Moving model to CPU before cleanup (if applicable).")
                    self.model.to("cpu")
                del self.model
            except Exception as e:
                logging.warning(f"Error during automatic cleanup of InsightFace model: {e}")
            finally:
                self.model = None
        gc.collect()
        if torch.cuda.is_available():
            logging.info("Clearing CUDA memory cache.")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class GRPulidFluxEvaClipLoader:
    def __init__(self):
        pass
        self.model = None
    gc.collect()

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

        self.model = model.visual

        eva_transform_mean = getattr(self.model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            self.model["image_mean"] = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            self.model["image_std"] = (eva_transform_std,) * 3

        return (self.model,)

    def cleanup(self):
        if self.model:
            logging.info("Cleaning up EVA-CLIP model resources...")
            try:
                if hasattr(self.model, "to") and callable(self.model.to):
                    logging.info("Moving model to CPU before cleanup (if applicable).")
                    self.model.to("cpu")
                del self.model
            except Exception as e:
                logging.warning(f"Error during manual cleanup of EVA-CLIP model: {e}")
            finally:
                self.model = None
        gc.collect()
        if torch.cuda.is_available():
            logging.info("Clearing CUDA memory cache.")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def __del__(self):
        if self.model:
            logging.info("Automatically cleaning up EVA-CLIP model resources...")
            try:
                if hasattr(self.model, "to") and callable(self.model.to):
                    logging.info("Moving model to CPU before cleanup (if applicable).")
                    self.model.to("cpu")
                del self.model
            except Exception as e:
                logging.warning(f"Error during automatic cleanup of EVA-CLIP model: {e}")
            finally:
                self.model = None
        gc.collect()
        if torch.cuda.is_available():
            logging.info("Clearing CUDA memory cache.")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class GRApplyPulidFlux:
    def __init__(self):
        pass
        self.pulid_data_dict = None

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
                "use_face_number": ("BOOLEAN",),
                "face_number": ("INT", {"default": "None", "min": 0, "max": 15, "step": 1}),
                "blur": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 1, "step": 0.01}),
                "face_select":  (["center_face","largest_face","smallest_face","most_prominent","normal"],),
                "fusion": (["mean","concat","max","norm_id","max_token","auto_weight","train_weight"],),
                "fusion_weight_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1 }),
                "fusion_weight_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1 }),
                "train_step": ("INT", {"default": 1000, "min": 0, "max": 20000, "step": 1 }),
                "use_gray": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
            "optional": {
                "attn_mask": ("MASK", ),
                "prior_image": ("IMAGE",), # for train weight, as the target
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_pulid_flux"
    CATEGORY = "pulid"

    def __init__(self):
        self.pulid_data_dict = None

    def apply_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, image, weight, start_at, end_at, prior_image=None, use_face_number=False,face_number=None, blur=0.01,face_select="center_face",fusion="mean", fusion_weight_max=1.0, fusion_weight_min=0.0, train_step=1000, use_gray=True, attn_mask=None, unique_id=None):
        device = comfy.model_management.get_torch_device()
        # Why should I care what args say, when the unet model has a different dtype?!
        # Am I missing something?!
        #dtype = comfy.model_management.unet_dtype()
        dtype = model.model.diffusion_model.dtype
        # For 8bit use bfloat16 (because ufunc_add_CUDA is not implemented)
        #if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        #    dtype = torch.bfloat16
        if model.model.manual_cast_dtype is not None:
            dtype = model.model.manual_cast_dtype

        eva_clip.to(device, dtype=dtype)
        pulid_flux.to(device, dtype=dtype)

        # TODO: Add masking support!
        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device, dtype=dtype)

        if prior_image is not None:
            prior_image = resize_with_pad(prior_image.to(image.device, dtype=image.dtype), target_size=(image.shape[1], image.shape[2]))
            image=torch.cat((prior_image,image),dim=0)
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
            if face_select == "center_face":
                face_helper.get_face_landmarks_5(only_center_face=True)
            elif face_select == "largest_face":
                face_helper.get_face_landmarks_5(only_keep_largest=True)
            elif face_select == "smallest_face":
                face_helper.get_face_landmarks_5(only_keep_smallest=True)
            elif face_select == "most_prominent":
                face_helper.get_face_landmarks_5(only_keep_most_prominent=True)
            elif use_face_number:
                face_helper.get_face_landmarks_5(only_center_face=False, blur_ratio=blur, select_by_index=face_number)
            else:
                face_helper.get_face_landmarks_5(only_center_face=False, blur_ratio=blur)
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
            if use_gray:
                _align_face = to_gray(align_face)
            else:
                _align_face = align_face
            face_features_image = torch.where(bg, white_image, _align_face)

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

        # fusion embeddings
        if fusion == "mean":
            cond = torch.cat(cond).to(device, dtype=dtype) # N,32,2048
            if cond.shape[0] > 1:
                cond = torch.mean(cond, dim=0, keepdim=True)
        elif fusion == "concat":
            cond = torch.cat(cond, dim=1).to(device, dtype=dtype)
        elif fusion == "max":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                cond = torch.max(cond, dim=0, keepdim=True)[0]
        elif fusion == "norm_id":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=(1,2))
                norm=norm/torch.sum(norm)
                cond=torch.einsum("wij,w->ij",cond,norm).unsqueeze(0)
        elif fusion == "max_token":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=2)
                _,idx=torch.max(norm,dim=0)
                cond=torch.stack([cond[j,i] for i,j in enumerate(idx)]).unsqueeze(0)
        elif fusion == "auto_weight": # 🤔
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=2)
                order=torch.argsort(norm,descending=False,dim=0)
                regular_weight=torch.linspace(fusion_weight_min,fusion_weight_max,norm.shape[0]).to(device, dtype=dtype)

                _cond=[]
                for i in range(cond.shape[1]):
                    o=order[:,i]
                    _cond.append(torch.einsum('ij,i->j',cond[:,i,:],regular_weight[o]))
                cond=torch.stack(_cond,dim=0).unsqueeze(0)
        elif fusion == "train_weight":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                if train_step > 0:
                    with torch.inference_mode(False):
                        cond = online_train(cond, device=cond.device, step=train_step)
                else:
                    cond = torch.mean(cond, dim=0, keepdim=True)

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        # Patch the Flux model (original diffusion_model)
        # Nah, I don't care for the official ModelPatcher because it's undocumented!
        # I want the end result now, and I don’t mind if I break other custom nodes in the process. 😄
        flux_model = model.model.diffusion_model
        # Let's see if we already patched the underlying flux model, if not apply patch
        if not hasattr(flux_model, "pulid_ca"):
            # Add perceiver attention, variables and current node data (weight, embedding, sigma_start, sigma_end)
            # The pulid_data is stored in Dict by unique node index,
            # so we can chain multiple ApplyPulidFlux nodes!
            flux_model.pulid_ca = pulid_flux.pulid_ca
            flux_model.pulid_double_interval = pulid_flux.double_interval
            flux_model.pulid_single_interval = pulid_flux.single_interval
            flux_model.pulid_data = {}
            # Replace model forward_orig with our own
            new_method = forward_orig.__get__(flux_model, flux_model.__class__)
            setattr(flux_model, 'forward_orig', new_method)

        # Patch is already in place, add data (weight, embedding, sigma_start, sigma_end) under unique node index
        flux_model.pulid_data[unique_id] = {
            'weight': weight,
            'embedding': cond,
            'sigma_start': sigma_start,
            'sigma_end': sigma_end,
        }

        # Keep a reference for destructor (if node is deleted the data will be deleted as well)
        self.pulid_data_dict = {'data': flux_model.pulid_data, 'unique_id': unique_id}

        self._cleanup(eva_clip, pulid_flux, model.model.diffusion_model)
        device = torch.device("cpu")
        eva_clip.to(device, dtype=dtype)
        pulid_flux.to(device, dtype=dtype)
        torch.cuda.empty_cache()
        del eva_clip
        del pulid_flux


        return (model,)

    def _cleanup(self, eva_clip, pulid_flux, flux_model):
        print("Performing cleanup...")
        UnloadAllModels.execute(flux_model=flux_model)
        del eva_clip, pulid_flux
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup completed.")
    
    def cleanup_models():
        print("Running cleanup_models...")
        global current_loaded_models
        to_delete = []
        for i in range(len(current_loaded_models)):
            if current_loaded_models[i].real_model() is None:
                to_delete = [i] + to_delete
    
        for i in to_delete:
            x = current_loaded_models.pop(i)
            del x


    def cleanup_models_gc():
        print("Running cleanup_models_gc...")
        global current_loaded_models
        do_gc = False
        for i in range(len(current_loaded_models)):
            cur = current_loaded_models[i]
            if cur.is_dead():
                print(
                    f"Potential memory leak detected with model {cur.real_model().__class__.__name__}. "
                    f"Performing full garbage collection."
                )
                do_gc = True
                break
        if do_gc:
            gc.collect()
            torch.cuda.empty_cache()
            for i in range(len(current_loaded_models)):
                cur = current_loaded_models[i]
                if cur.is_dead():
                    print(
                        f"WARNING: Memory leak with model {cur.real_model().__class__.__name__}. "
                        f"Ensure it is not being referenced elsewhere."
                    )

class UnloadAllModels:
    @staticmethod
    def execute(flux_model=None):
        print("Unload All Models:")
        print(" - Unloading all models...")
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        if flux_model is not None:
            UnloadAllModels._clear_flux_latents(flux_model)
        UnloadAllModels._cleanup_memory()
        return {"status": "success", "message": "All models unloaded and memory cleaned up."}

    @staticmethod
    def _clear_flux_latents(flux_model):
        try:
            if hasattr(flux_model, "latent_rgb_factors") and hasattr(flux_model, "latent_channels"):
                print(" - Detected a Flux-like model. Clearing latents...")
                if hasattr(flux_model, "latent_rgb_factors"):
                    flux_model.latent_rgb_factors = None
                    print("   - Cleared latent_rgb_factors.")
                if hasattr(flux_model, "latent_rgb_factors_bias"):
                    flux_model.latent_rgb_factors_bias = None
                    print("   - Cleared latent_rgb_factors_bias.")
                if hasattr(flux_model, "taesd_decoder_name"):
                    flux_model.taesd_decoder_name = None
                    print("   - Cleared taesd_decoder_name reference.")
                flux_model.latent_channels = 0
                print("   - Cleared latent_channels.")
            else:
                print("   - Model does not appear to be a Flux-like model. Skipping latent cleanup.")
        except Exception as e:
            print(f"   - Error during Flux latent cleanup: {e}")

    @staticmethod
    def _cleanup_memory():
        try:
            print(" - Clearing system and GPU cache...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("   - Cache cleared successfully.")
        except Exception as e:
            print(f"   - Unable to clear cache: {e}")

NODE_CLASS_MAPPINGS = {
    "GRPulidFluxModelLoader": GRPulidFluxModelLoader,
    "GRPulidFluxInsightFaceLoader": GRPulidFluxInsightFaceLoader,
    "GRPulidFluxEvaClipLoader": GRPulidFluxEvaClipLoader,
    "GRApplyPulidFlux": GRApplyPulidFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRPulidFluxModelLoader": "GR Load PuLID Flux Model",
    "GRPulidFluxInsightFaceLoader": "GR Load InsightFace (PuLID Flux)",
    "GRPulidFluxEvaClipLoader": "GR Load Eva Clip (PuLID Flux)",
    "GRApplyPulidFlux": "GR Apply PuLID Flux",
}
