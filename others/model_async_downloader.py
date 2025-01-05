import os
import json
import ast
import shared
import time
from urllib.parse import urlparse
from typing import Optional
import gradio as gr
import threading

def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    # 定义镜像源列表
    MIRRORS = ["huggingface.co", "hf-mirror.com"]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    
    # 使用临时文件进行下载
    temp_file = os.path.abspath(os.path.join(model_dir, f"{file_name}.tmp"))
    final_file = os.path.abspath(os.path.join(model_dir, file_name))
    
    # 如果最终文件已存在，直接返回
    if os.path.exists(final_file):
        return final_file
        
    # 初始化下载状态
    downloaded_size = 0
    if os.path.exists(temp_file):
        downloaded_size = os.path.getsize(temp_file)
    
    # 尝试多个镜像源
    last_error = None
    for mirror in MIRRORS:
        try:
            # 替换URL中的域名
            mirror_url = url.replace("huggingface.co", mirror, 1)
            
            print(f'Downloading: "{mirror_url}" to {final_file}')
            print(f'正在下载模型文件: "{mirror_url}"。如果速度慢，可终止运行，自行用工具下载后保存到: {final_file}，然后重启应用。\n')
            
            import requests
            from tqdm import tqdm
            
            # 设置Range头支持断点续传
            headers = {}
            if downloaded_size > 0:
                headers["Range"] = f"bytes={downloaded_size}-"
            
            # 打开临时文件进行追加写入
            mode = "ab" if downloaded_size > 0 else "wb"
            with open(temp_file, mode) as file:
                # 添加重试逻辑
                max_retries = 5
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        with requests.get(mirror_url, headers=headers, stream=True, timeout=10) as response:
                            # 处理断点续传的206响应
                            if response.status_code == 206:
                                content_range = response.headers.get('content-range')
                                print(f"Content-Range: {content_range}")
                                total = int(content_range.split('/')[1])
                            else:
                                content_length = response.headers.get('content-length', '0')
                                print(f"Content-Length: {content_length}")
                                total = int(content_length)
                            
                            # 如果之前有下载部分，调整进度条
                            print(f"Total size: {total}")
                            
                            with tqdm(
                                desc=file_name,
                                total=total,
                                unit='iB',
                                unit_scale=True,
                                unit_divisor=1024,
                                initial=downloaded_size
                            ) as pbar:
                                for data in response.iter_content(chunk_size=1024):
                                    if cancel_event and cancel_event.is_set():
                                        file.close()
                                        if os.path.exists(temp_file):
                                            os.remove(temp_file)
                                        print(f"\n取消下载: {file_name}")
                                        return None
                                    size = file.write(data)
                                    pbar.update(size)
                            break  # 成功完成下载，退出重试循环
                    except (requests.exceptions.RequestException, ConnectionError) as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"连接中断，5秒后重试 ({retry_count}/{max_retries})...")
                            time.sleep(5)
                            # 更新已下载大小
                            downloaded_size = os.path.getsize(temp_file)
                            if downloaded_size > 0:
                                headers["Range"] = f"bytes={downloaded_size}-"
                        else:
                            raise e
            
            # 下载完成后重命名临时文件
            os.rename(temp_file, final_file)
            shared.modelsinfo.refresh_file('add', final_file, url)
            return final_file
            
        except Exception as e:
            last_error = e
            print(f"镜像源 {mirror} 下载失败: {str(e)}")
            continue
    
    # 所有镜像源都失败，抛出最后一个错误
    if last_error:
        raise last_error
    return None


models_to_download = []

def ready_to_download_url(preset, user_did, cata, file_name, size, url, model_dir):
    global models_to_download

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        models_to_download.append(dict(
            a=a
            ))
        return ''
    else:
        return cached_file


def download_it_from_ready_list(preset, user_did=None):
    from modules.config import path_models_root, model_cata_map
    from modules.model_loader import presets_model_list, default_download_url_prefix

    # 准备要下载的模型列表
    models_to_download = []

    # 处理特殊的预设名称（可能包含用户ID）
    if preset.endswith('.'):
        if user_did is None:
            return False
        preset = f'{preset}{user_did[:7]}'

    # 获取预设对应的模型列表
    model_list = [] if preset not in presets_model_list else presets_model_list[preset]

    if len(model_list) > 0:
        # 首先检查哪些模型需要下载
        for cata, path_file, size, hash10, url in model_list:
            # 确定文件名
            if path_file[:1] == '[' and path_file[-1:] == ']':
                if url:
                    parts = urlparse(url)
                    file_name = os.path.basename(parts.path)
                else:
                    continue
            else:
                file_name = path_file.replace('\\', '/').replace(os.sep, '/')

            # 检查模型是否已存在
            if cata in model_cata_map:
                model_dir = model_cata_map[cata][0]
            else:
                model_dir = os.path.join(path_models_root, cata)

            full_path_file = os.path.abspath(os.path.join(model_dir, file_name))

            # 如果文件不存在，加入待下载列表
            if not os.path.exists(full_path_file):
                models_to_download.append({
                    'cata': cata,
                    'path_file': path_file,
                    'size': size,
                    'url': url,
                    'file_name': file_name,
                    'model_dir': model_dir
                })

        # 如果有需要下载的模型，创建下载确认界面
        if models_to_download:
            download_info = "#### The following model needs to be downloaded\n\n"
            for model in models_to_download:
                size_gb = round(model['size'] / (1024 * 1024 * 1024), 2)
                download_info += f"- {model['file_name']} (Type: {model['cata']}, Size: {size_gb}GB)\n"
                gr.Markdown("<br>")  # 添加空行
            

            # 使用threading.Event来控制下载取消
            cancel_event = threading.Event()

            def confirm_download(confirm):
                if not confirm:
                    cancel_event.set()
                    return "已取消下载。"
                
                download_results = []
                for model in models_to_download:
                    if cancel_event.is_set():
                        return "下载已取消。"

                    try:
                        url = model['url']
                        if url is None or url == '':
                            url = f'{default_download_url_prefix}/{model["cata"]}/{model["path_file"]}'
            
                        if model['path_file'][:1] == '[' and model['path_file'][-1:] == ']' and url.endswith('.zip'):
                            result = download_diffusers_model(path_models_root, model['cata'], 
                                                            model['path_file'][1:-1], model['size'], 
                                                            url, cancel_event)
                            if result is None:
                                return "下载已取消。"
                            download_results.append(f"成功下载: {model['file_name']}")
                        else:
                            result = load_file_from_url(
                                url=url,
                                model_dir=model['model_dir'],
                                file_name=model['file_name'],
                                cancel_event=cancel_event
                            )
                            if result is None:
                                return "下载已取消。"
                            download_results.append(f"成功下载: {model['file_name']}")
                    except Exception as e:
                        download_results.append(f"下载失败: {model['file_name']} - {str(e)}")
                        if not cancel_event.is_set():
                            cancel_event.set()
                            return f"下载过程中出错: {str(e)}"
            
                return "\n".join(download_results)

            def cancel_download():
                cancel_event.set()
                return "正在取消下载..."

            # 创建一个新的Blocks实例来处理下载确认
            with gr.Blocks(title="模型下载", css="style.css") as download_interface:
                with gr.Box(elem_classes="download-box"):
                    gr.Markdown(download_info, elem_classes=["download-status"])
                    with gr.Row(elem_classes=["button-row"]):
                        confirm_btn = gr.Button("Download", variant="primary")
                        cancel_btn = gr.Button("Cancel", variant="secondary")
                    output = gr.Textbox(label="Start with the default model while you wait. Sketch as you go along.", interactive=False, elem_classes=["download-time-estimate"])

                    # 在Blocks上下文中绑定事件
                    confirm_btn.click(
                        fn=confirm_download,
                        inputs=[gr.State(True)],
                        outputs=output,
                    ).then(
                        fn=lambda: gr.update(interactive=False),
                        outputs=[confirm_btn]
                    )

                    cancel_btn.click(
                        fn=cancel_download,
                        outputs=output,
                    ).then(
                        fn=lambda: gr.update(interactive=False),
                        outputs=[cancel_btn]
                    )

                # 启动下载界面
                download_interface.launch(
                    prevent_thread_lock=True,
                    share=False,
                    inbrowser=True,
                    show_api=False,
                    width=500,
                    height=300
                )

    return



