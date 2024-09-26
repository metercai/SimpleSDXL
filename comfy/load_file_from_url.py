import os
import folder_paths
from urllib.parse import urlparse
from typing import Optional

import json

def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        if url.find("huggingface.co")>=0:
            url = url.replace('huggingface.co', 'hf-mirror.com')
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)

    return cached_file

def load_model_for_path(models_url, root_name):
    models_root = folder_paths.get_folder_paths(root_name)[0]
    for model_path in models_url:
        model_full_path = os.path.join(models_root, model_path)
        if not os.path.exists(model_full_path):
            model_full_path = load_file_from_url(
                url=models_url[model_path], model_dir=models_root, file_name=model_path
            )

def load_model_for_iclight():
    models_url = dict({
        "iclight_sd15_fc_unet_ldm.safetensors": "https://huggingface.co/huchenlei/IC-Light-ldm/resolve/main/iclight_sd15_fc_unet_ldm.safetensors",
        #"iclight_sd15_fbc_unet_ldm.safetensors": "https://huggingface.co/huchenlei/IC-Light-ldm/resolve/main/iclight_sd15_fbc_unet_ldm.safetensors",
        })
    load_model_for_path(models_url, "unet")

    models_url = dict({
        "realisticVisionV60B1_v51VAE.safetensors": "https://huggingface.co/metercai/SimpleSDXL2/resolve/main/ckpt/realisticVisionV60B1_v51VAE.safetensors"
        })
    load_model_for_path(models_url, "checkpoints")




