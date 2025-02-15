import os
import anyio
import httpx
from tqdm import tqdm
import threading
import queue
import json
import ast
import shared
from urllib.parse import urlparse
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from torch.hub import download_url_to_file

import logging
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

thread_pool = ThreadPoolExecutor(max_workers=6)
download_tasks = set()
task_lock = threading.Lock()

async def download_file_with_progress(url: str, file_path: str, size: int=0):
    timeout = int(max(60.0, size / (1024 * 1024)))
    logger.info(f'the download file timeout: {timeout}s')
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        try:
            if 'HF_MIRROR' in os.environ:
                url = str.replace(url, "huggingface.co", os.environ["HF_MIRROR"].rstrip('/'), 1)
            model_dir = os.path.dirname(file_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length", 0))
                
                with tqdm(
                    total=total_size, unit="iB", unit_scale=True, desc=''
                ) as progress_bar:
                    partial_file_path = file_path + ".partial"
                    with open(partial_file_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            progress_bar.update(len(chunk))
            os.rename(partial_file_path, file_path)
            shared.modelsinfo.refresh_file('add', file_path, url)
        except httpx.HTTPStatusError as e:
            logger.error(f"下载失败: {e}")
            logger.error(f"请求 URL: {e.request.url}")
            logger.error(f"重定向 URL: {e.response.headers.get('Location')}")
            raise
        except Exception as e:
            logger.error(f"下载过程中发生错误: {e}")
            raise


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        async_task: bool = False,
        size: int = 0,
) -> str:
    global download_queue

    """
    Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    if 'HF_MIRROR' in os.environ:
        url = str.replace(url, "huggingface.co", os.environ["HF_MIRROR"].rstrip('/'), 1)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        #logger.info(f'Downloading: "{url}" to {cached_file}')
        logger.info(f'正在下载文件: "{url}"。如果速度慢，可终止运行，自行用工具下载后保存到: {cached_file}，进入"模型"页点击"本地刷新"按钮。')
        def _download_task():
            try:
                anyio.run(download_file_with_progress, url, cached_file, size)
            except Exception as e:
                print(f'下载任务:{model_name} 失败, 错误为: {e}')
            finally:
                with task_lock:
                    download_tasks.discard(file_name)
                    logger.info(f"下载任务:{file_name} 已完成, 从任务队列中清除.")
        if async_task:
            with task_lock:
                if file_name in download_tasks:
                    print(f"下载任务:{file_name} 已经在任务队列中.")
                    return
                download_tasks.add(file_name)
                print(f"启动新的下载任务:{file_name}.")
            thread_pool.submit(_download_task)
        else:
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
        user_path_preset = get_path_in_user_dir('presets', user_did)
    if len(presets)>0:
        for preset in presets:
            if preset.endswith('.'):
                if user_did is None:
                    continue
                preset_file = os.path.join(user_path_preset, f'{preset}json')
                preset = f'{preset}{user_did[:7]}'
            else:
                preset_file = os.path.join(path_preset, f'{preset}.json')
            try:
                mtime = os.path.getmtime(preset_file)
                if preset not in presets_mtime:
                    presets_mtime[preset] = 0
                if mtime>presets_mtime[preset]:
                    presets_mtime[preset] = mtime
                    with open(preset_file, "r", encoding="utf-8") as json_file:
                        config_preset = json.load(json_file)
                    if 'model_list' in config_preset:
                        model_list = config_preset['model_list']
                        model_list = [tuple(p.split(',')) for p in model_list]
                        model_list = [(cata.strip(), path_file.strip(), int(size), hash10.strip(), url.strip()) for (cata, path_file, size, hash10, url) in model_list]
                        presets_model_list[preset] = model_list
            except Exception as e:
                logger.info(f'load preset file failed: {preset_file}')
                continue
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
                path_file = [f'{path_file[1:-1]}/']
                result = shared.modelsinfo.get_model_names(cata, path_file, casesensitive=True)
                if result is None or len(result)<size:
                    logger.info(f'Missing model dir in preset({preset}): {cata}, filter={path_file}, len={size}\nresult={result}')
                    return False
            else:
                file_path = shared.modelsinfo.get_model_filepath(cata, path_file)
                if file_path is None or file_path == '' or not os.path.exists(file_path) or size != os.path.getsize(file_path):
                    logger.info(f'Missing model file in preset({preset}): {cata}, {path_file}')
                    return False
        return True
    return False

default_download_url_prefix = 'https://huggingface.co/metercai/SimpleSDXL2/resolve/main/SimpleModels'
def download_model_files(preset, user_did=None, async_task=False):
    from modules.config import path_models_root, model_cata_map
    global presets_model_list, default_download_url_prefix, download_queue
    
    if preset.endswith('.'):
        if user_did is None:
            return False
        preset = f'{preset}{user_did[:7]}'
    model_list = [] if preset not in presets_model_list else presets_model_list[preset]
    if len(model_list)>0:
        download_task_list = []
        for cata, path_file, size, hash10, url in model_list:
            download_task = ('', {})
            if path_file[:1]=='[' and path_file[-1:]==']':
                if url:
                    parts = urlparse(url)
                    file_name = os.path.basename(parts.path)
                    result = shared.modelsinfo.get_model_names(cata, [f'{path_file[1:-1]}/'], casesensitive=True)
                    if result and len(result)>=size:
                        continue
                else:
                    continue
            else:
                file_name = path_file.replace('\\', '/').replace(os.sep, '/')
            if cata in model_cata_map:
                model_dir=model_cata_map[cata][0]
            else:
                model_dir=os.path.join(path_models_root, cata)
            full_path_file = os.path.abspath(os.path.join(model_dir, file_name))
            if os.path.exists(full_path_file):
                continue
            logger.info(f'The model file is not exists, ready to download: {file_name}')
            model_dir = os.path.dirname(full_path_file)
            file_name = os.path.basename(full_path_file)
            if url is None or url == '':
                url = f'{default_download_url_prefix}/{cata}/{path_file}'
            if path_file[:1]=='[' and path_file[-1:]==']' and url.endswith('.zip'):
                download_diffusers_model(cata, path_file[1:-1], size, url)
            else:
                if not async_task:
                    load_file_from_url(
                        url=url,
                        model_dir=model_dir,
                        file_name=file_name
                    )
                else:
                    load_file_from_url(
                        url=url,
                        model_dir=model_dir,
                        file_name=file_name,
                        async_task=True,
                        size=size
                    )
    return


def download_diffusers_model(cata, model_name, num, url):
    def _download_task():
        try:
            anyio.run(download_diffusers_model_async, cata, model_name, num, url)
        except Exception as e:
            print(f'下载任务:{model_name} 失败, 错误为: {e}')
        finally:
            with task_lock:
                download_tasks.discard(model_name)
                print(f"下载任务:{model_name} 已完成, 从任务队列中清除.")

    with task_lock:
        if model_name in download_tasks:
            print(f"下载任务:{model_name} 已经在任务队列中.")
            return
        download_tasks.add(model_name)
        print(f'开启新的下载任务:{model_name} in "{cata}" from {url}.')

    thread_pool.submit(_download_task)


async def download_diffusers_model_async(cata, model_name, num, url):
    import zipfile
    import shutil
    from modules.config import path_models_root, model_cata_map

    path_filter = [f'{model_name}/']
    result = shared.modelsinfo.get_model_names(cata, path_filter, casesensitive=True)
    if result is None or len(result)<num:
        path_temp = os.path.join(path_models_root, 'temp')
        if not os.path.exists(path_temp):
            os.makedirs(path_temp)
        file_name = os.path.basename(urlparse(url).path)
        downfile = os.path.join(path_temp, file_name)
        download_url_to_file(url, downfile, progress=True)
        with zipfile.ZipFile(downfile, 'r') as zipf:
            logger.info(f'extractall: {downfile} to {path_temp}')
            zipf.extractall(path_temp)
        shutil.move(os.path.join(path_temp, f'SimpleModels/{cata}/{model_name}'), os.path.join(model_cata_map[cata][0], model_name))
        os.remove(downfile)
        shutil.rmtree(path_temp)
    shared.modelsinfo.refresh_from_path()
    return

