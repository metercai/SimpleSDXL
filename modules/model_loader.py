import os
from urllib.parse import urlparse
from typing import Optional

import json

urlmapping = {}
urlmapping_path = os.path.abspath(f'./enhanced/urlmapping.json')
with open(urlmapping_path, "r", encoding="utf-8") as json_file:
    urlmapping.update(json.load(json_file))


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
        from enhanced.models_hub_host import models_hub_host
        import enhanced.token_did as token_did
        from enhanced.location import location
        try:
            requests.post(f'{models_hub_host}/register_claim/', data = token_did.get_register_claim('SimpleSDXLHub'))
        except Exception as e:
            print(f'Connect the models hub site failed!')
            print(e)
        print(f'Get info from {models_hub_host} and downloading: "{filename}" at location: {location}.')
        filename2 = token_did.encrypt_default(muid)
        url2 = f'{models_hub_host}/mfile/{token_did.DID}/{filename2}'
        #url2 = f'{models_hub_host}/lmfile/{location}/{token_did.DID}/{filename2}'
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
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        from enhanced.location import location
        if location=='CN' and url in urlmapping.keys():
            load_file_from_muid(file_name, urlmapping[url], model_dir, progress)
        else:
            print(f'Downloading: "{url}" to {cached_file}\n')
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)

    return cached_file

