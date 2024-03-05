# put the link of the model you want to download (checkpoint or LoRA)
# link can specify a version id and if not specified the newest version will be downloaded
# url example without version id: https://civitai.com/models/133005/juggernaut-xl
# url example with version id: https://civitai.com/models/133005?modelVersionId=252902
# -----------------------------

import requests
from urllib.parse import urlparse
from urllib.parse import parse_qs
from modules.model_loader import load_file_from_url
from modules.config import paths_checkpoints, path_loras

def download_from_civitai(c_url):
    allowed_types = ['Checkpoint', 'LORA']
    save_loactions = {
        'Checkpoint': paths_checkpoints[0],
        'LORA': path_loras
    }

    #c_url = "https://civitai.com/models/133005/juggernaut-xl"
    parsed_url = urlparse(c_url)
    model_id = parsed_url.path.split('/')[parsed_url.path.split('/').index('models') + 1]
    model_version_id = parse_qs(parsed_url.query).get('modelVersionId')

    url = "https://civitai.com/api/v1/models/" + model_id
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError('model not found')
    data = response.json()

    model_type = data.get('type')
    if model_type not in allowed_types:
        raise RuntimeError('model is not a checkpoint or LoRA')
    model_versions = data.get('modelVersions')
    selected_version = None
    if model_version_id:
        for model_version in model_versions:
            if str(model_version.get('id')) == model_version_id[0]:
                selected_version = model_version
    else:
        selected_version = model_versions[0]

    if selected_version is None:
        raise RuntimeError("this version doesn't exist")
    

    files = selected_version.get('files')
    primary_file = None
    for f in files:
        if f.get('primary'):
            primary_file = f
    download_url = primary_file.get('downloadUrl')
    filename = primary_file.get('name')+".1"

    model_name = data.get('name')
    selected_version_name = selected_version.get('name')
    print(f'downloading {model_name} ({model_type} version {selected_version_name})')
    print(f'from {download_url} to {save_loactions[model_type]}/{filename}')

    load_file_from_url(
        url=download_url,
        model_dir=save_loactions[model_type],
        file_name=filename
    )
    return filename

url = "https://civitai.com/models/128397"
download_from_civitai(url)
