import os
import json
import shared
from urllib.parse import urlparse
from typing import Optional

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
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}')
        print(f'正在下载模型文件: "{url}"。如果速度慢，可终止运行，自行用工具下载后保存到: {cached_file}，然后重启应用。\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
        shared.modelsinfo.refresh_file('add', cached_file, url)
    return cached_file


presets_model_list = {}
presets_mtime = {}

#    "flux_aio" : [
#        ("controlnet", "flux.1-dev_canny_controlnet_v3.safetensors", 1487623552, 0, "controlnet/flux.1-dev_canny_controlnet_v3.safetensors"),
#        ("checkpoints", [['flux','dev','safetensors'], ['flux', 'hyp', 'gguf']], 0, 0, ""),
#        ];

def refresh_model_list(presets, user_did=None):
    from enhanced.simpleai import get_path_in_user_dir
    global presets_model_list, presets_mtime

    path_preset = os.path.abspath(f'./presets/')
    if user_did:
        user_path_preset = get_path_in_user_dir(user_did, 'presets')
    if len(presets)>0:
        for preset in presets:
            if preset.endswith('.'):
                if user_did is None:
                    continue
                preset_file = os.path.join(user_path_preset, f'{preset}json')
                preset = f'{preset}{user_did[:7]}'
            else:
                preset_file = os.path.join(path_preset, f'{preset}.json')
            mtime = os.path.getmtime(preset_file)
            if preset not in presets_mtime:
                presets_mtime[preset] = 0
            if  mtime>presets_mtime[preset]:
                presets_mtime[preset] = mtime
                with open(preset_file, "r", encoding="utf-8") as json_file:
                    config_preset = json.load(json_file)
                if 'model_list' in config_preset:
                    model_list = config_preset['model_list']
                    model_list = [tuple(p.split(',')) for p in model_list]
                    model_list = [(cata, path_file, int(size), hash10, url) for (cata, path_file, size, hash10, url) in model_list]
                    presets_model_list[preset] = model_list
    return
            

def check_models_exists(preset, user_did=None):
    from modules.config import path_models_root
    global presets_model_list

    if preset.endswith('.'):
        if user_did is None:
            return False
        preset = f'{preset}{user_did[:7]}'
    model_list = [] if preset not in presets_model_list else presets_model_list[preset]
    if len(model_list)>0:
        for cata, path_file, size, hash10, url in model_list:
            if isinstance(path_file, list):
                result = shared.modelsinfo.get_model_names(cata, path_file)
                if result is None or len(result)==0:
                    print(f'[ModelInfos] Missing a type of model file in preset({preset}): {cata}, filter={path_file}')
                    return False
            else:
                file_path = shared.modelsinfo.get_model_filepath(cata, path_file)
                if file_path is None or file_path == '' or size != os.path.getsize(file_path):
                    print(f'[ModelInfos] Missing model file in preset({preset}): {cata}, {path_file}')
                    return False
        return True
    return False


default_download_url_prefix = 'https://huggingface.co/metercai/SimpleSDXL2/resolve/main/SimpleModels'
def download_model_files(preset, user_did=None):
    from modules.config import path_models_root, model_cata_map
    global presets_model_list, default_download_url_prefix

    if preset.endswith('.'):
        if user_did is None:
            return False
        preset = f'{preset}{user_did[:7]}'
    model_list = [] if preset not in presets_model_list else presets_model_list[preset]
    if len(model_list)>0:
        for cata, path_file, size, hash10, url in model_list:
            if isinstance(path_file, list):
                if url:
                    parts = urlparse(url)
                    file_name = os.path.basename(parts.path)
                else:
                    return
            else:
                file_name = path_file.replace('\\', '/').replace(os.sep, '/')
            if cata in model_cata_map:
                model_dir=model_cata_map[cata][0]
            else:
                model_dir=os.path.join(path_models_root, cata)
            full_path_file = os.path.abspath(os.path.join(model_dir, file_name))
            model_dir = os.path.dirname(full_path_file)
            file_name = os.path.basename(full_path_file)
            if url is None or url == '':
                url = f'{default_download_url_prefix}/{cata}/{path_file}'
            load_file_from_url(
                url=url,
                model_dir=model_dir,
                file_name=file_name
            )
    return

