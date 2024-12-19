import os
import json
import ast
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
                    model_list = [(cata.strip(), path_file.strip(), int(size), hash10.strip(), url.strip()) for (cata, path_file, size, hash10, url) in model_list]
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
            if path_file[:1]=='[' and path_file[-1:]==']':
                path_file = [path_file[1:-1]]
                result = shared.modelsinfo.get_model_names(cata, path_file, casesensitive=True)
                if result is None or len(result)<size:
                    #print(f'[ModelInfos] Missing model dir in preset({preset}): {cata}, filter={path_file}, len={size}\nresult={result}')
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
            if path_file[:1]=='[' and path_file[-1:]==']':
                if url:
                    parts = urlparse(url)
                    file_name = os.path.basename(parts.path)
                else:
                    continue
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
            if path_file[:1]=='[' and path_file[-1:]==']' and url.endswith('.zip'):
                download_diffusers_model(path_models_root, cata, path_file[1:-1], size, url)
            else:
                load_file_from_url(
                    url=url,
                    model_dir=model_dir,
                    file_name=file_name
                )
    return

def download_diffusers_model(path_models_root, cata, model_name, num, url):
    import zipfile
    import shutil
    from shared import modelsinfo

    path_filter = [model_name]
    result = shared.modelsinfo.get_model_names(cata, path_filter, casesensitive=True)
    if result is None or len(result)<num:
        path_temp = os.path.join(path_models_root, 'temp')
        if not os.path.exists(path_temp):
            os.makedirs(path_temp)
        file_name = os.path.basename(urlparse(url).path)
        load_file_from_url(
            url=url,
            model_dir=path_temp,
            file_name=file_name
        )
        downfile = os.path.join(path_temp, file_name)
        with zipfile.ZipFile(downfile, 'r') as zipf:
            print(f'extractall: {downfile} to {path_temp}')
            zipf.extractall(path_temp)
        shutil.move(os.path.join(path_temp, f'SimpleModels/{cata}/{model_name}'), os.path.join(os.path.join(path_models_root, cata), model_name))
        os.remove(downfile)
        shutil.rmtree(path_temp)
    modelsinfo.refresh_from_path()
    return

