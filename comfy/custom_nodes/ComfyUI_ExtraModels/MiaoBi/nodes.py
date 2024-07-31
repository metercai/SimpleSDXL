import os
import folder_paths

import comfy.sd
import comfy.diffusers_load
from .tokenizer import MiaoBiTokenizer

class MiaoBiCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip"),),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_mbclip"
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "MiaoBi CLIP Loader"

    def load_mbclip(self, clip_name):
        clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type
        )
        # override tokenizer
        clip.tokenizer.clip_l = MiaoBiTokenizer()
        return (clip,)


class MiaoBiDiffusersLoader:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {
            "required": {
                "model_path": (paths,),
                }
            }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_mbcheckpoint"
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "MiaoBi Checkpoint Loader (Diffusers)"

    def load_mbcheckpoint(self, model_path, output_vae=True, output_clip=True):
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break
        unet, clip, vae = comfy.diffusers_load.load_diffusers(
            model_path,
            output_vae = output_vae,
            output_clip = output_clip,
            embedding_directory = folder_paths.get_folder_paths("embeddings")
        )
        # override tokenizer
        clip.tokenizer.clip_l = MiaoBiTokenizer()
        return (unet, clip, vae)

NODE_CLASS_MAPPINGS = {
    "MiaoBiCLIPLoader": MiaoBiCLIPLoader,
    "MiaoBiDiffusersLoader": MiaoBiDiffusersLoader,
}