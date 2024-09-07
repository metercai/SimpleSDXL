import os
from urllib.parse import urlparse
from typing import Optional

import json
import shutil

urlmapping = {}
urlmapping_path = os.path.abspath(f'./enhanced/urlmapping.json')
with open(urlmapping_path, "r", encoding="utf-8") as json_file:
    urlmapping.update(json.load(json_file))
accelerate_muid = []
accelerate_muid_path = os.path.abspath(f'./enhanced/accelerate_muid.json')
with open(accelerate_muid_path, "r", encoding="utf-8") as json_file:
    accelerate_muid=json.load(json_file)

def load_file_from_muid(
        filename: str,
        muid: str,
        model_dir: str,
        pregress: bool = True,
) -> str:
    os.makedirs(model_dir, exist_ok=True)
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        import requests
        from enhanced.simpleai import models_hub_host as hub
        from shared import token as token_did, sysinfo
        location = sysinfo["location"]

        try:
            if muid in accelerate_muid:
                print(f'Get info from {hub.models_hub_host_2} and downloading: "{filename}" at location: {location}.')
                requests.post(f'{hub.models_hub_host_2}/register_claim/', data = token_did.get_register_claim('SimpleSDXLHub_site2'))
                filename2 = token_did.encrypt(hub.accelerate_DID, muid)
                url2 = f'{hub.models_hub_host_2}/mfile/{token_did.DID}/{filename2}'
            else:
                print(f'Get info from {hub.models_hub_host} and downloading: "{filename}" at location: {location}.')
                requests.post(f'{hub.models_hub_host}/register_claim/', data = token_did.get_register_claim('SimpleSDXLHub'))
                filename2 = token_did.encrypt_default(muid)
                url2 = f'{hub.models_hub_host}/mfile/{token_did.DID}/{filename2}'
        except Exception as e:
            print(f'Connect the models hub site failed!')
            print(e)
        cached_file2 = os.path.abspath(os.path.join(model_dir, filename2))
        from download import download
        download(url2, cached_file2, progressbar=True)
        os.rename(cached_file2, cached_file)
    return cached_file


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
    domain = os.environ.get("HF_MIRROR", "huggingface.co").rstrip('/')
    url = str.replace(url, "huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
        #from enhanced.simpleai import refresh_models_info
        #refresh_models_info()

    return cached_file

