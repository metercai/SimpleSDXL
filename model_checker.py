import os
import sys
import time
import requests
import queue
from tqdm import tqdm
from colorama import init, Fore, Style
import threading
import atexit
import json
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
def load_model_paths():
    global simplemodels_root

    config_path = os.path.normpath(os.path.join(script_dir, "..", "..", "users", "config.txt"))
    path_mapping = {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        models_root = config.get("path_models_root", None)
        if models_root:
            simplemodels_root = os.path.abspath(os.path.join(script_dir, models_root)) if not os.path.isabs(models_root) else models_root
        else:
            simplemodels_root = os.path.normpath(os.path.join(script_dir, "..", "..", "SimpleModels"))
        path_mapping = {
            "checkpoints": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in config.get("path_checkpoints", [])],
            "loras": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                    for p in config.get("path_loras", [])],
            "controlnet": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in config.get("path_controlnet", [])],
            "embeddings": [os.path.abspath(os.path.join(script_dir, p))
                        for p in ([config.get("path_embeddings")] if isinstance(config.get("path_embeddings"), str)
                                else config.get("path_embeddings", []))],
            "vae_approx": [os.path.abspath(os.path.join(script_dir, p))
                        for p in ([config.get("path_vae_approx")] if isinstance(config.get("path_vae_approx"), str)
                                else config.get("path_vae_approx", []))],
            "vae": [os.path.abspath(os.path.join(script_dir, p))
                    for p in ([config.get("path_vae")] if isinstance(config.get("path_vae"), str)
                            else config.get("path_vae", []))],
            "upscale_models": [os.path.abspath(os.path.join(script_dir, p))
                            for p in ([config.get("path_upscale_models")] if isinstance(config.get("path_upscale_models"), str)
                                    else config.get("path_upscale_models", []))],
            "inpaint": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                    for p in config.get("path_inpaint", [])],
            "clip": [os.path.abspath(os.path.join(script_dir, p))
                    for p in ([config.get("path_clip")] if isinstance(config.get("path_clip"), str)
                            else config.get("path_clip", []))],
            "clip_vision": [os.path.abspath(os.path.join(script_dir, p))
                            for p in ([config.get("path_clip_vision")] if isinstance(config.get("path_clip_vision"), str)
                                    else config.get("path_clip_vision", []))],
            "fooocus_expansion": [os.path.abspath(os.path.join(script_dir, config.get("path_fooocus_expansion", "")))],
            "llms": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                    for p in config.get("path_llms", [])],
            "safety_checker": [os.path.abspath(os.path.join(script_dir, config.get("path_safety_checker", "")))],
            "unet": [os.path.abspath(os.path.join(script_dir, config.get("path_unet", "")))],
            "rembg": [os.path.abspath(os.path.join(script_dir, config.get("path_rembg", "")))],
            "layer_model": [os.path.abspath(os.path.join(script_dir, config.get("path_layer_model", "")))],
            "diffusers": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in config.get("path_diffusers", [])],
            "ipadapter": [os.path.abspath(os.path.join(script_dir, config.get("path_ipadapter", "")))],
            "pulid": [os.path.abspath(os.path.join(script_dir, config.get("path_pulid", "")))],
            "insightface": [os.path.abspath(os.path.join(script_dir, config.get("path_insightface", "")))],
            "style_models": [os.path.abspath(os.path.join(script_dir, config.get("path_style_models", "")))],
            "configs": [os.path.abspath(os.path.join(simplemodels_root, "configs"))],
            "prompt_expansion": [os.path.abspath(os.path.join(simplemodels_root, "prompt_expansion"))],
        }

    except Exception as e:
        print(f"{Fore.YELLOW}△配置文件加载失败: {e}，使用默认路径{Style.RESET_ALL}")
        path_mapping = {
            "checkpoints": [os.path.join(simplemodels_root, "checkpoints")],
            "loras": [os.path.join(simplemodels_root, "loras")],
            "controlnet": [os.path.join(simplemodels_root, "controlnet")],
            "embeddings": [os.path.join(simplemodels_root, "embeddings")],
            "vae_approx": [os.path.join(simplemodels_root, "vae_approx")],
            "vae": [os.path.join(simplemodels_root, "vae")],
            "upscale_models": [os.path.join(simplemodels_root, "upscale_models")],
            "inpaint": [os.path.join(simplemodels_root, "inpaint")],
            "clip": [os.path.join(simplemodels_root, "clip")],
            "clip_vision": [os.path.join(simplemodels_root, "clip_vision")],
            "fooocus_expansion": [os.path.join(simplemodels_root, "prompt_expansion", "fooocus_expansion")],
            "llms": [os.path.join(simplemodels_root, "llms")],
            "safety_checker": [os.path.join(simplemodels_root, "safety_checker")],
            "unet": [os.path.join(simplemodels_root, "unet")],
            "rembg": [os.path.join(simplemodels_root, "rembg")],
            "layer_model": [os.path.join(simplemodels_root, "layer_model")],
            "diffusers": [os.path.join(simplemodels_root, "diffusers")],
            "ipadapter": [os.path.join(simplemodels_root, "ipadapter")],
            "pulid": [os.path.join(simplemodels_root, "pulid")],
            "insightface": [os.path.join(simplemodels_root, "insightface")],
            "style_models": [os.path.join(simplemodels_root, "style_models")],
            "configs": [os.path.normpath(os.path.join(simplemodels_root, "configs"))],
            "prompt_expansion": [os.path.normpath(os.path.join(simplemodels_root, "prompt_expansion"))],
        }

    for key in path_mapping:
        path_mapping[key] = [
            os.path.abspath(p) if not os.path.isabs(p) else p
            for p in path_mapping[key]
        ]
        path_mapping[key] = list(set(path_mapping[key]))

    return path_mapping

def cleanup():
    if os.path.exists("downloadlist.txt"):
        os.remove("downloadlist.txt")
        print("已删除 'downloadlist.txt' 文件。")
    if os.path.exists("缺失模型下载链接.txt"):
        os.remove("缺失模型下载链接.txt")
        print("已删除 '缺失模型下载链接.txt' 文件。")
atexit.register(cleanup)

init(autoreset=True)
class DownloadStatus:
    def __init__(self, filename, total_size):
        self.filename = filename
        self.total_size = total_size
        self.progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=filename,
            position=0,
            leave=True
        )

def print_colored(text, color=Fore.WHITE):
    print(f"{color}{text}{Style.RESET_ALL}")

def check_python_embedded():
    python_exe = sys.executable
    print(f"Python解析器路径: {python_exe}")

    if "python_embeded" not in python_exe.lower():
        print_colored("×当前 Python 解释器不在 python_embeded 目录中，请检查运行环境", Fore.RED)
        input("按任意键继续。")
        sys.exit(1)

def check_script_file():
    script_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SimpleSDXL", "entry_with_update.py")

    if os.path.exists(script_file):
        print_colored("√找到主程序目录", Fore.GREEN)
    else:
        print_colored("×未找到主程序目录，请检查脚本位置", Fore.RED)
        input("按任意键继续。")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(script_file))
    directory_level = len(base_dir.split(os.sep))

    if directory_level <= 2:
        print_colored("×主程序目录层级不足，可能会导致脚本结果有误。请按照安装视频指引先建立SimpleAI主文件夹", Fore.RED)
    else: 
        print_colored("√主程序目录层级验证通过", Fore.GREEN)

    paths_to_check = [
        ("当前脚本路径", os.path.abspath(__file__)),
        ("主程序目录", base_dir),
        ("入口文件路径", script_file)
    ]

    has_space = False
    for desc, path in paths_to_check:
        if ' ' in path:
            print_colored(f"!警告：{desc}包含空格 -> {path}", Fore.YELLOW)
            has_space = True

    if has_space:
        print_colored("!路径包含空格可能导致程序异常，建议将SimpleAI安装到无空格路径（如D:\\SimpleAI）", Fore.YELLOW)
        time.sleep(10)

def get_total_virtual_memory():
    try:
        import psutil
        virtual_mem = psutil.virtual_memory().total
        swap_mem = psutil.swap_memory().total
        total_virtual_memory = virtual_mem + swap_mem
        return total_virtual_memory
    except ImportError:
        print_colored("无法导入 psutil 模块，跳过内存检查", Fore.YELLOW)
        return None
    except Exception as e:
        print_colored(f"无法获取系统虚拟内存，可能是性能计数器未开启或其他问题。\n错误详情: {e}", Fore.YELLOW)
        print_colored("请参考https://learn.microsoft.com/zh-cn/troubleshoot/windows-server/performance/rebuild-performance-counter-library-values重新启用系统性能计数器，或忽略此警告继续。", Fore.YELLOW)
        return None

def check_virtual_memory(total_virtual):
    if total_virtual is None:
        print_colored("跳过虚拟内存检查。", Fore.YELLOW)
        return
    total_gb = total_virtual / (1024 ** 3)
    if total_gb < 40:
        print_colored("警告：系统虚拟内存小于40GB，会禁用部分预置包，请参考安装视频教程设置系统虚拟内存。", Fore.YELLOW)
    else:
        print_colored("√系统虚拟内存充足", Fore.GREEN)
    print(f"系统总虚拟内存: {total_gb:.2f} GB")

def find_simplemodels_dir(start_path):
    current_dir = start_path
    while current_dir != os.path.dirname(current_dir):
        simplemodels_path = os.path.join(current_dir, "SimpleModels")
        if os.path.isdir(simplemodels_path):
            return simplemodels_path
        current_dir = os.path.dirname(current_dir)
    return None

def find_users_dir(start_path):
    current_dir = start_path
    while current_dir != os.path.dirname(current_dir):
        users_path = os.path.join(current_dir, "users")
        if os.path.isdir(users_path):
            return users_path
        current_dir = os.path.dirname(current_dir)
    return None

def normalize_path(path):
    path_mapping = load_model_paths()

    path_parts = path.split('/')
    if len(path_parts) < 2:
        return os.path.abspath(path)

    path_type = path_parts[0].lower()
    filename = '/'.join(path_parts[1:])

    sorted_dirs = sorted(
        path_mapping.get(path_type, []),
        key=lambda x: (
            0 if "SimpleModels" in x else
            1 if any(part == "models" for part in x.split(os.sep)) else
            2,
            x
        )
    )

    for base_dir in sorted_dirs:
        full_path = os.path.join(base_dir, filename)
        return os.path.abspath(full_path)

def typewriter_effect(text, delay=0.01):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
        
    print()
def print_instructions():
    print()
    print(f"{Fore.GREEN}★★★★★{Style.RESET_ALL}安装视频教程{Fore.YELLOW}https://www.bilibili.com/video/BV1ddkdYcEWg/{Style.RESET_ALL}{Fore.GREEN}★★★★★{Style.RESET_ALL}{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print()
    print(f"{Fore.GREEN}★{Style.RESET_ALL}攻略地址飞书文档:{Fore.YELLOW}https://acnmokx5gwds.feishu.cn/wiki/QK3LwOp2oiRRaTkFRhYcO4LonGe{Style.RESET_ALL}文章无权限即为未编辑完毕。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}稳速生图指南:Nvidia显卡驱动选择最新版驱动,驱动类型最好为Studio。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}在遇到生图速度断崖式下降或者爆显存OutOfMemory时,提高{Fore.GREEN}预留显存功能{Style.RESET_ALL}的数值至（1~2）{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}打开默认浏览器设置，关闭GPU加速、或图形加速的选项。{Fore.GREEN}★{Style.RESET_ALL}大内存(64+)与固态硬盘存放模型有助于减少模型加载时间。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}疑难杂症进QQ群求助：938075852{Fore.GREEN}★{Style.RESET_ALL}脚本：✿   冰華 |版本:25.03.02{Fore.GREEN}★{Style.RESET_ALL}")
    print()
    time.sleep(0.1)
    
def get_unique_filename(file_path, extension=".corrupted"):
    base = file_path + extension
    counter = 1
    while os.path.exists(base):
        base = f"{file_path}{extension}_{counter}"
        counter += 1
    return base

def validate_files(packages):
    cleanup()
    path_mapping = load_model_paths()
    print_colored(f">>>>>>默认模型根目录为：{simplemodels_root}<<<<<<", Fore.YELLOW)
    print()
    download_files = {}
    missing_package_names = []
    package_percentages = {}
    package_sizes = {}
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for package_key, package_info in packages.items():
        package_name = package_info["name"]
        package_note = package_info.get("note", "")
        files_and_sizes = package_info["files"]
        download_links = package_info["download_links"]


        total_size = sum([size for _, size in files_and_sizes])
        total_size_gb = total_size / (1024 ** 3)
        non_missing_size = 0

        print(f"－－－－－－－", end='')
        time.sleep(0.1)
        print(f"校验{package_name}文件－－－－{package_note}")

        missing_files = []
        size_mismatch_files = []
        case_mismatch_files = []

        for expected_path, expected_size in files_and_sizes:
            expected_filename = os.path.basename(expected_path) 
            path_parts = expected_path.split('/')
            path_type = path_parts[0].lower() if len(path_parts) > 0 else ''
            sub_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ''

            search_dirs = sorted(
                path_mapping.get(path_type, []),
                key=lambda x: (
                    0 if "SimpleModels" in x else
                    1 if any(part == "models" for part in x.split(os.sep)) else
                    2,
                    x
                )
            )
            if not search_dirs:
                simplemodels_default = os.path.join(root, "SimpleModels")
                search_dirs = [os.path.join(simplemodels_default, path_type)]
                print(simplemodels_default)

            found = False
            actual_dir = None

            for base_dir in search_dirs:
                full_path = os.path.join(base_dir, sub_path) if sub_path else os.path.join(base_dir, os.path.basename(expected_path))
                if os.path.exists(full_path):
                    actual_dir = os.path.dirname(full_path)
                    found = True
                    break

            if not found:
                missing_files.append((expected_path, expected_size))
                continue

            try:
                directory_listing = os.listdir(actual_dir)
            except Exception as e:
                print(f"{Fore.RED}目录访问错误: {actual_dir} - {str(e)}{Style.RESET_ALL}")
                missing_files.append((expected_path, expected_size))
                continue

            expected_filename = os.path.basename(expected_path)
            actual_filename = next((f for f in directory_listing if f.lower() == expected_filename.lower()), None)
            directory_listing = os.listdir(actual_dir)
            actual_filename = next((f for f in directory_listing if f.lower() == expected_filename.lower()), None)

            if actual_filename is None:
                missing_files.append((expected_path, expected_size))
            elif actual_filename != expected_filename:
                case_mismatch_files.append((os.path.join(actual_dir, actual_filename), expected_filename))
            else:
                actual_size = os.path.getsize(os.path.join(actual_dir, actual_filename))
                if actual_size != expected_size:
                    size_mismatch_files.append((os.path.join(actual_dir, actual_filename), actual_size, expected_size))
                else:
                    non_missing_size += expected_size
        obsolete_files = []
        MODEL_PATHS_TO_SCAN = [
        os.path.join(simplemodels_root, "checkpoints"),
        os.path.join(simplemodels_root, "loras"),
        os.path.join(simplemodels_root, "controlnet")]
        for model_root in MODEL_PATHS_TO_SCAN:
            if not os.path.exists(model_root):
                continue
            for root, _, files in os.walk(model_root):
                for file in files:
                    # 使用文件名全小写匹配
                    if file.lower() in [x.lower() for x in OBSOLETE_MODELS]:
                        full_path = os.path.join(root, file)
                        obsolete_files.append(full_path)


        if total_size > 0:
            non_missing_percentage = (non_missing_size / total_size) * 100
            package_percentages[package_name] = non_missing_percentage
            package_sizes[package_name] = total_size_gb

        if case_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件名大小写不匹配，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_filename in case_mismatch_files:
                print(f"文件: {normalize_path(file)}")
                time.sleep(0.1)
                print(f"正确文件名: {expected_filename}")
                
                corrected_file_path = os.path.join(os.path.dirname(file), expected_filename)
                os.rename(file, corrected_file_path)
                print(f"{Fore.GREEN}文件名已更正为: {expected_filename}{Style.RESET_ALL}")

        if size_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。{Style.RESET_ALL}")
            for file, actual_size, expected_size in size_mismatch_files:
                normalized_path = normalize_path(file)
                print(f"{normalized_path} 当前大小={actual_size}, 预期大小={expected_size}")
                time.sleep(0.1)
                
                corrupted_file_path = get_unique_filename(file)
                os.rename(file, corrupted_file_path)
                print(f"{Fore.YELLOW}文件已重命名为: {normalize_path(corrupted_file_path)}（大小不匹配）{Style.RESET_ALL}")
                
                relative_path = os.path.relpath(file, root).replace(os.sep, '/')
                download_files[relative_path] = expected_size
                if package_name not in missing_package_names:
                    missing_package_names.append(package_name)

        if missing_files:
            print(f"{Fore.RED}×{package_name}有文件缺失，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_size in missing_files:
                print(normalize_path(file))
                download_files[file] = expected_size
            if package_name not in missing_package_names:
                missing_package_names.append(package_name)

            if package_info["download_links"]:
                print(f"{Fore.YELLOW}下载链接(若为压缩包，则参考安装视频流程安装):{Style.RESET_ALL}")
                for link in package_info["download_links"]:
                    print(f"{Fore.YELLOW}{link}{Style.RESET_ALL}")
        if not missing_files and not size_mismatch_files and not case_mismatch_files:
            print(f"{Fore.GREEN}√{package_name}文件全部验证通过{Style.RESET_ALL}")


    if missing_package_names:
        print(f"{Fore.RED}△以下包体缺失文件，请检查并重新下载：{Style.RESET_ALL}")
        for package_name in missing_package_names:
            percentage = package_percentages.get(package_name, 0)
            total_size_gb = package_sizes.get(package_name, 0)

            missing_size_gb = total_size_gb * (1 - (percentage / 100))
            print(f"- {package_name} - 总大小：{total_size_gb:.2f}GB，完整度：{percentage:.2f}%，尚需下载：{missing_size_gb:.2f}GB")
    if obsolete_files:
        # 新增空间计算
        total_obsolete_size = 0
        for file in obsolete_files:
            try:
                total_obsolete_size += os.path.getsize(file)
            except:
                pass
        print(f"\n{Fore.YELLOW}△发现以下可删除的废弃模型：{Style.RESET_ALL}")
        for file in obsolete_files:
            print(f"  {file}")
        # 新增空间显示
        print(f"{Fore.CYAN}※这些模型已被新版替代，可节省空间: {total_obsolete_size/1024/1024/1024:.2f}GB (按0+回车清理时选择删除){Style.RESET_ALL}")
    # 新增基础包自动下载逻辑
    sorted_download_files = sorted(download_files.items(), key=lambda x: x[1])

    if sorted_download_files:
        with open("downloadlist.txt", "w") as f1, open("缺失模型下载链接.txt", "w") as f2:
            for file, size in sorted_download_files:
                if file == "inpaint/GroundingDINO_SwinT_OGC.cfg.py":
                    link = "https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py"
                else:
                    link = f"https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/{file.split('SimpleModels/')[-1]}"

                
                f1.write(f"{link},{size}\n")
                
                f2.write(f"{link}\n")
        print(f"{Fore.YELLOW}>>>问题文件的文件下载链接已保存到 '缺失模型下载链接.txt'。<<<<<<<<<<<<<<<<<<<<<{Style.RESET_ALL}")
    if "[1]基础模型包" in missing_package_names:
        package_id = 1
        selected_package = None
        for package_name, package_info in packages.items():
            if package_info["id"] == package_id:
                selected_package = package_info
                break

        if selected_package:
            get_download_links_for_package({package_name: selected_package}, "downloadlist.txt")

        print(f"\n{Fore.CYAN}△检测到基础包不完整，自动触发下载流程...{Style.RESET_ALL}")
        auto_download_missing_files_with_retry(max_threads=5)

def delete_partial_files():
    global OBSOLETE_MODELS
    try:
        path_mapping = load_model_paths()
    except Exception as e:
        print(f"{Fore.RED}△路径配置加载失败: {str(e)}{Style.RESET_ALL}")
        return

    scan_categories = [
        'checkpoints', 'loras', 'controlnet', 'embeddings',
        'vae_approx', 'vae', 'upscale_models', 'inpaint',
        'clip', 'clip_vision', 'llms', 'unet', 'diffusers'
    ]

    scan_dirs = []
    for category in scan_categories:
        scan_dirs.extend(path_mapping.get(category, []))
    
    default_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "SimpleModels"
    ))
    if default_dir not in scan_dirs:
        scan_dirs.append(default_dir)

    total_size = 0
    files_found = False
    files_to_delete = []
    obsolete_files_found = []  # 新增废弃文件存储

    for model_dir in scan_dirs:
        if not os.path.exists(model_dir):
            continue

        print(f"{Fore.CYAN}△扫描目录: {model_dir}{Style.RESET_ALL}")

        for root, _, files in os.walk(model_dir):
            for file in files:
                if ".partial" in file or ".corrupted" in file:
                    file_path = os.path.join(root, file)
                    files_found = True
                    files_to_delete.append(file_path)
                    try:
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
                if file in OBSOLETE_MODELS:  # 精确文件名匹配
                    file_path = os.path.join(root, file)
                    obsolete_files_found.append(file_path)
                    files_found = True
    if files_found:
        print(f"{Fore.YELLOW}△以下未下载完或损坏的文件将被删除：{Style.RESET_ALL}")
        for file_path in files_to_delete:
            print(f"- {file_path}")

        obsolete_total = sum(os.path.getsize(f) for f in obsolete_files_found if os.path.exists(f))

        if obsolete_files_found:
            print(f"\n{Fore.YELLOW}△以下废弃模型文件将被删除：{Style.RESET_ALL}")
            for file in obsolete_files_found:
                print(f"  {file}")
        all_files_to_delete = files_to_delete + obsolete_files_found  # 新增合并逻辑

        print(f"{Fore.CYAN}△可清理的磁盘空间: {(total_size + obsolete_total) / (1024 * 1024):.2f} MB{Style.RESET_ALL}")
        confirm = input(f"{Fore.GREEN}△是否确认删除这些文件？(y/n): {Style.RESET_ALL}")
        if confirm.lower() == 'y':
            success_count = 0
            for file_path in all_files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"{Fore.GREEN}√已删除: {file_path}{Style.RESET_ALL}")
                    success_count += 1
                except Exception as e:
                    print(f"{Fore.RED}×删除失败[{file_path}]: {str(e)}{Style.RESET_ALL}")
            print(f"操作完成！成功删除 {success_count}/{len(all_files_to_delete)} 个文件")
        else:
            print(f"{Fore.RED}△删除操作已取消{Style.RESET_ALL}")
    else:
        print(f">>>未找到需要删除的临时/损坏文件<<<")

def delete_specific_image_files():
    """
    从相对路径查找并删除所有 .png、.webp 和 .jpg/jpeg 文件，排除 welcome.png。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    users_dir = find_users_dir(script_dir)
    target_dir = os.path.join(users_dir, "guest_user", "comfyd_inputs")
    if not os.path.exists(target_dir):
        print(f"{Fore.RED}△未找到指定目录: {target_dir}{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}△正在清理目录 '{target_dir}' 中的临时图片缓存...{Style.RESET_ALL}")

    total_size = 0
    files_found = False
    files_to_delete = []

    for root, _, files in os.walk(target_dir):
        for file in files:
            if (file.endswith(".png") or file.endswith(".webp") or file.endswith(".jpg") or file.endswith(".jpeg")) and file != "welcome.png":
                files_found = True
                file_path = os.path.join(root, file)
                files_to_delete.append(file_path)
                total_size += os.path.getsize(file_path)
    if files_found:
        print(f"{Fore.YELLOW}△以下临时图片缓存文件将被删除：{Style.RESET_ALL}")
        for file_path in files_to_delete:
            print(f"- {file_path}")
        print(f"{Fore.CYAN}△可清理的磁盘空间: {total_size / (1024 * 1024):.2f} MB{Style.RESET_ALL}")

        confirm = input(f"{Fore.GREEN}△是否确认删除这些文件？(y/n): {Style.RESET_ALL}")
        if confirm.lower() == 'y':
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"{Fore.GREEN}√已删除文件: {file_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}△删除文件时出错: {file_path}, 错误原因: {e}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}△删除操作已取消。{Style.RESET_ALL}")
    else:
        print(f">>>未找到需要删除的临时图片缓存<<<")
        print()

def delete_log_files():
    """
    删除与脚本所在位置一致的 logs 目录下的所有 .logs 文件
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")

    if not os.path.exists(logs_dir):
        print(f"{Fore.RED}△未找到指定日志目录: {logs_dir}{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}△正在清理目录 '{logs_dir}' 中的日志文件...{Style.RESET_ALL}")

    total_size = 0
    files_found = False
    files_to_delete = []

    for root, _, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".log"):
                files_found = True
                file_path = os.path.join(root, file)
                files_to_delete.append(file_path)
                total_size += os.path.getsize(file_path)

    if files_found:
        print(f"{Fore.YELLOW}△以下日志文件将被删除：{Style.RESET_ALL}")
        for file_path in files_to_delete:
            print(f"- {file_path}")

        print(f"{Fore.CYAN}△可清理的磁盘空间: {total_size / (1024 * 1024):.2f} MB{Style.RESET_ALL}")

        confirm = input(f"{Fore.GREEN}△是否确认删除这些日志文件？(y/n): {Style.RESET_ALL}")
        if confirm.lower() == 'y':
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"{Fore.GREEN}√已删除日志文件: {file_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}△删除文件时出错: {file_path}, 错误原因: {e}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}△删除操作已取消。{Style.RESET_ALL}")
    else:
        print(f">>>未找到需要删除的日志文件<<<")
        print()

def download_file_with_resume(link, file_path, position, result_queue, max_retries=5, lock=None):
    partial_file_path = file_path + ".partial"
    retries = 0
    while retries < max_retries:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.exists(partial_file_path):
                resume_size = os.path.getsize(partial_file_path)
                headers = {'Range': f"bytes={resume_size}-"}
            else:
                resume_size = 0
                headers = {}
            response = requests.get(link, stream=True, headers=headers)
            total_size = int(response.headers.get('content-length', 0)) + resume_size
            block_size = 8192

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(partial_file_path, 'ab') as file, tqdm(
                    desc=os.path.basename(file_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    position=position,
                    initial=resume_size
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress_bar.update(len(data))

            final_file_path = os.path.normpath(file_path)
            partial_file_path = os.path.normpath(partial_file_path)
            os.rename(partial_file_path, final_file_path)
            print(f"{Fore.GREEN}√下载完成：{final_file_path}{Style.RESET_ALL}")

            if lock:
                with lock:
                    remove_link_from_downloadlist(link)

            result_queue.put(True)
            return

        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}△下载失败，正在重试... 错误：{e}{Style.RESET_ALL}")
            retries += 1
            time.sleep(5)
        except Exception as e:
            print(f"{Fore.RED}发生错误：{e}{Style.RESET_ALL}")
            result_queue.put(False)
            return

    print(f"△下载链接失败：{link}")
    result_queue.put(False)

def remove_link_from_downloadlist(link):
    """
    删除下载列表中已成功下载的条目
    :param link: 下载链接
    :return: None
    """
    with open("downloadlist.txt", "r") as f:
        lines = f.readlines()

    with open("downloadlist.txt", "w") as f:
        for line in lines:
            if link.strip() not in line.strip():
                f.write(line)
def trigger_manual_download():
    """手动触发指定文件下载"""
    path_mapping = load_model_paths()

    for link in MANUAL_DOWNLOAD_LIST:
        if "SimpleModels/" in link:
            path_part = link.split("SimpleModels/", 1)[1]
            path_parts = path_part.split('/')
            path_type = path_parts[0].lower()
            rel_path = '/'.join(path_parts[1:])
        else:
            continue

        sorted_base_dir = sorted(
            path_mapping.get(path_type, []),
            key=lambda x: (
                0 if "SimpleModels" in x else
                1 if any(part == "models" for part in x.split(os.sep)) else 2,
                x
            )
        )

        target_base_dir = None
        for base_dir in sorted_base_dir:
            if os.path.exists(base_dir):
                target_base_dir = base_dir
                break
        if not target_base_dir:
            continue

        file_name = os.path.basename(link)
        save_path = os.path.join(target_base_dir, rel_path)

        if os.path.exists(save_path):
            print(f"{Fore.GREEN}△文件已存在，跳过下载: {save_path}{Style.RESET_ALL}")
            continue

        print(f"{Fore.CYAN}△开始下载: {file_name}{Style.RESET_ALL}")
        result_queue = queue.Queue()
        download_file_with_resume(link, save_path, 0, result_queue)

def auto_download_missing_files_with_retry(max_threads=5):
    if not os.path.exists("downloadlist.txt"):
        print("未找到 'downloadlist.txt' 文件。")
        return

    with open("downloadlist.txt", "r") as f:
        links = f.readlines()

    if not links:
        print("没有缺失文件需要下载！")
        return

    path_mapping = load_model_paths()
    result_queue = queue.Queue()
    lock = threading.Lock()

    task_queue = queue.Queue()
    for position, line in enumerate(links):
        task_queue.put((position, line.strip()))

    def worker():
        while not task_queue.empty():
            try:
                position, line = task_queue.get_nowait()
                link, size = line.split(',')
                size_mb = int(size) / (1024 * 1024)
                print(f"{Fore.CYAN}▶ 正在下载: {link} ({size_mb:.1f}MB){Style.RESET_ALL}")
                original_repo = "https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/"
                if link.startswith(original_repo):
                    relative_path = link.replace(original_repo, "", 1).strip()
                    relative_path_without_prefix = relative_path.replace("SimpleModels/", "", 1)
                    path_type = relative_path_without_prefix.split('/')[0].lower()
                else:
                    relative_path = link.replace("https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/", "", 1).strip()
                    relative_path_without_prefix = link.split("https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/", 1)[-1].strip()
                    path_type = "inpaint"

                sorted_base_dir = sorted(
                    path_mapping.get(path_type, []),
                    key=lambda x: (
                        0 if "SimpleModels" in x else
                        1 if any(part == "models" for part in x.split(os.sep)) else
                        2,
                        x
                    )
                )

                target_base_dir = None
                for base_dir in sorted_base_dir:
                    if os.path.exists(base_dir):
                        target_base_dir = base_dir
                        break
                if not target_base_dir:
                    print(f"{Fore.RED}×无法找到路径类型 '{path_type}' 的配置，跳过下载: {relative_path}{Style.RESET_ALL}")
                    continue

                file_name = os.path.basename(relative_path)
                file_sub_dir = os.path.dirname(relative_path_without_prefix).replace(path_type, "").strip('/')
                save_dir = os.path.join(target_base_dir, file_sub_dir)
                file_path = os.path.join(save_dir, file_name)

                thread = threading.Thread(
                    target=download_file_with_resume,
                    args=(link, file_path, position, result_queue, 5, lock)
                )
                thread.start()
                thread.join()

                task_queue.task_done()
            except queue.Empty:
                break

    threads = []
    for _ in range(max_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    task_queue.join()

    success_count = 0
    fail_count = 0

    while not result_queue.empty():
        success = result_queue.get()
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"√下载成功：{success_count}个")
    print(f"×下载失败：{fail_count}个")

    if fail_count == 0 and success_count > 0:
        if os.path.exists("downloadlist.txt"):
            os.remove("downloadlist.txt")
            print("√下载完成，已删除 'downloadlist.txt' 文件。输入【R】重新检测")
    else:
        print(f"△有{fail_count}个文件下载失败，请检查网络连接或手动下载文件。")

def get_download_links_for_package(packages, download_list_path):
    """
    根据 packages 中的 files 列表生成路径，并与 downloadlist.txt 中的需求进行比对，
    更新 downloadlist.txt 中需要下载的文件，只保留 files 中有的文件链接。
    """
    if not os.path.exists(download_list_path):
        print(f"{Fore.RED}>>>downloadlist.txt不存在，输入【R】重新检测<<<{Style.RESET_ALL}")
        return []

    with open(download_list_path, "r") as f:
        existing_links = [line.strip().split(",")[0] for line in f.readlines()]

    valid_files = []
    with open(download_list_path, "r") as f:
        existing_lines = [line.strip() for line in f.readlines()]

    for line in existing_lines:
        existing_link = line.split(",")[0]
        for package_name, package_info in packages.items():
            for file_path, file_size in package_info["files"]:
                if file_path == "inpaint/GroundingDINO_SwinT_OGC.cfg.py":
                    generated_link = "https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py"
                else:
                    generated_link = f"https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/{file_path}"

                if generated_link == existing_link:
                    valid_files.append((generated_link, file_size))
                    break

    valid_files = sorted(valid_files, key=lambda x: x[1])

    with open(download_list_path, "w") as f:
        for link, size in valid_files:
            f.write(f"{link},{size}\n")

    print(f"{Fore.YELLOW}>>>下载列表已更新，开始下载（关闭窗口可中断）。<<<{Style.RESET_ALL}")

    return valid_files

def delete_package(package_name, packages):
    """删除指定包体文件（基于config路径配置）"""
    try:
        path_mapping = load_model_paths()
    except Exception as e:
        print(f"{Fore.RED}× 路径配置加载失败: {str(e)}{Style.RESET_ALL}")
        return

    if package_name not in packages:
        print(f"{Fore.RED}× 无效的包体名称！{Style.RESET_ALL}")
        return

    package = packages[package_name]
    print(f"\n{Fore.CYAN}△ 开始处理包体：{package['name']}{Style.RESET_ALL}")

    file_refs = defaultdict(list)
    for pkg_name, pkg_info in packages.items():
        for file_entry in pkg_info["files"]:
            path_parts = file_entry[0].split('/')
            if len(path_parts) < 1: continue

            file_type = path_parts[0].lower()
            rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""

            for base_dir in path_mapping.get(file_type, []):
                full_path = os.path.join(base_dir, rel_path)
                if os.path.exists(full_path):
                    file_refs[full_path].append(pkg_name)

    delete_candidates = []
    shared_files = []

    for file_entry in package["files"]:
        path_parts = file_entry[0].split('/')
        if len(path_parts) < 1: continue

        file_type = path_parts[0].lower()
        rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""
        found = False

        for base_dir in path_mapping.get(file_type, []):
            full_path = os.path.join(base_dir, rel_path)
            if os.path.exists(full_path):

                if len(file_refs[full_path]) == 1 and file_refs[full_path][0] == package_name:
                    delete_candidates.append(full_path)
                else:
                    shared_files.append(full_path)
                found = True
                break

        if not found:
            default_path = os.path.join(simplemodels_root, file_type, rel_path)
            if os.path.exists(default_path):
                if len(file_refs[default_path]) == 1 and file_refs[default_path][0] == package_name:
                    delete_candidates.append(default_path)
                else:
                    shared_files.append(default_path)

    if shared_files:
        print(f"\n{Fore.YELLOW}△ 以下文件被其他包体共享：{Style.RESET_ALL}")
        for path in shared_files:
            print(f"  {path}")

    if delete_candidates:
        print(f"\n{Fore.YELLOW}△ 以下孤立文件将被删除：{Style.RESET_ALL}")
        total_size = 0
        for path in delete_candidates:
            try:
                size = os.path.getsize(path)
                print(f"  {path} ({size/1024/1024:.1f}MB)")
                total_size += size
            except:
                print(f"  {path} (大小未知)")

        print(f"{Fore.CYAN}△ 总计释放空间: {total_size/1024/1024/1024:.2f}GB{Style.RESET_ALL}")
        
        confirm = input(f"\n{Fore.GREEN}是否确认删除？(y/n): {Style.RESET_ALL}")
        if confirm.lower() == 'y':
            success = 0
            for path in delete_candidates:
                try:
                    os.remove(path)
                    print(f"{Fore.GREEN}✓ 已删除: {path}{Style.RESET_ALL}")
                    success += 1
                except Exception as e:
                    print(f"{Fore.RED}× 删除失败: {path} ({str(e)}){Style.RESET_ALL}")
            
            print(f"\n{Fore.GREEN}✓ 包体{selected_package['name']}孤立文件已清除{Style.RESET_ALL}")
        else:
            print(f"{Fore.BLUE}× 操作已取消{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}△ 未找到可安全删除的文件{Style.RESET_ALL}")


packages = {
    "base_package": {
        "id": 1,
        "name": "[1]基础模型包",
        "note": "SDXL全功能|显存需求：★★ 速度：★★★☆",
        "files": [
            ("checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("checkpoints/realisticVisionV60B1_v51VAE.safetensors", 2132625894),
            ("clip_vision/clip_vision_vit_h.safetensors", 1972298538),
            ("clip_vision/model_base_caption_capfilt_large.pth", 896081425),
            ("clip_vision/wd-v1-4-moat-tagger-v2.onnx", 326197340),
            ("clip_vision/wd-v1-4-moat-tagger-v2.csv", 253906),
            ("clip_vision/clip-vit-large-patch14/merges.txt", 524619),
            ("clip_vision/clip-vit-large-patch14/special_tokens_map.json", 389),
            ("clip_vision/clip-vit-large-patch14/tokenizer_config.json", 905),
            ("clip_vision/clip-vit-large-patch14/vocab.json", 961143),
            ("configs/anything_v3.yaml", 1933),
            ("configs/v1-inference.yaml", 1873),
            ("configs/v1-inference_clip_skip_2.yaml", 1933),
            ("configs/v1-inference_clip_skip_2_fp16.yaml", 1956),
            ("configs/v1-inference_fp16.yaml", 1896),
            ("configs/v1-inpainting-inference.yaml", 1992),
            ("configs/v2-inference-v.yaml", 1815),
            ("configs/v2-inference-v_fp32.yaml", 1816),
            ("configs/v2-inference.yaml", 1789),
            ("configs/v2-inference_fp32.yaml", 1790),
            ("configs/v2-inpainting-inference.yaml", 4450),
            ("controlnet/detection_Resnet50_Final.pth", 109497761),
            ("controlnet/fooocus_ip_negative.safetensors", 65616),
            ("controlnet/ip-adapter-plus-face_sdxl_vit-h.bin", 1013454761),
            ("controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427),
            ("controlnet/parsing_parsenet.pth", 85331193),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("controlnet/lllyasviel/Annotators/body_pose_model.pth", 209267595),
            ("controlnet/lllyasviel/Annotators/facenet.pth", 153718792),
            ("controlnet/lllyasviel/Annotators/hand_pose_model.pth", 147341049),
            ("inpaint/fooocus_inpaint_head.pth", 52602),
            ("inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("inpaint/inpaint_v26.fooocus.patch", 1323362033),
            ("inpaint/isnet-anime.onnx", 176069933),
            ("inpaint/isnet-general-use.onnx", 178648008),
            ("inpaint/sam_vit_b_01ec64.pth", 375042383),
            ("inpaint/silueta.onnx", 44173029),
            ("inpaint/u2net.onnx", 175997641),
            ("inpaint/u2netp.onnx", 4574861),
            ("inpaint/u2net_cloth_seg.onnx", 176194565),
            ("inpaint/u2net_human_seg.onnx", 175997641),
            ("layer_model/layer_xl_fg2ble.safetensors", 701981624),
            ("layer_model/layer_xl_transparent_conv.safetensors", 3619745776),
            ("layer_model/vae_transparent_decoder.safetensors", 208266320),
            ("llms/bert-base-uncased/config.json", 570),
            ("llms/bert-base-uncased/model.safetensors", 440449768),
            ("llms/bert-base-uncased/tokenizer.json", 466062),
            ("llms/bert-base-uncased/tokenizer_config.json", 28),
            ("llms/bert-base-uncased/vocab.txt", 231508),
            ("llms/Helsinki-NLP/opus-mt-zh-en/config.json", 1394),
            ("llms/Helsinki-NLP/opus-mt-zh-en/generation_config.json", 293),
            ("llms/Helsinki-NLP/opus-mt-zh-en/metadata.json", 1477),
            ("llms/Helsinki-NLP/opus-mt-zh-en/pytorch_model.bin", 312087009),
            ("llms/Helsinki-NLP/opus-mt-zh-en/source.spm", 804677),
            ("llms/Helsinki-NLP/opus-mt-zh-en/target.spm", 806530),
            ("llms/Helsinki-NLP/opus-mt-zh-en/tokenizer_config.json", 44),
            ("llms/Helsinki-NLP/opus-mt-zh-en/vocab.json", 1617902),
            ("llms/superprompt-v1/config.json", 1512),
            ("llms/superprompt-v1/generation_config.json", 142),
            ("llms/superprompt-v1/model.safetensors", 307867048),
            ("llms/superprompt-v1/README.md", 3661),
            ("llms/superprompt-v1/spiece.model", 791656),
            ("llms/superprompt-v1/tokenizer.json", 2424064),
            ("llms/superprompt-v1/tokenizer_config.json", 2539),
            ("loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors", 371842896),
            ("loras/sdxl_lightning_4step_lora.safetensors", 393854592),
            ("loras/sd_xl_offset_example-lora_1.0.safetensors", 49553604),
            ("prompt_expansion/fooocus_expansion/config.json", 937),
            ("prompt_expansion/fooocus_expansion/merges.txt", 456356),
            ("prompt_expansion/fooocus_expansion/positive.txt", 5655),
            ("prompt_expansion/fooocus_expansion/pytorch_model.bin", 351283802),
            ("prompt_expansion/fooocus_expansion/special_tokens_map.json", 99),
            ("prompt_expansion/fooocus_expansion/tokenizer.json", 2107625),
            ("prompt_expansion/fooocus_expansion/tokenizer_config.json", 255),
            ("prompt_expansion/fooocus_expansion/vocab.json", 798156),
            ("rembg/RMBG-1.4.pth", 176718373),
            ("unet/iclight_sd15_fc_unet_ldm.safetensors", 1719144856),
            ("upscale_models/fooocus_upscaler_s409985e5.bin", 33636613),
            ("vae_approx/vaeapp_sd15.pth", 213777),
            ("vae_approx/xl-to-v1_interposer-v4.0.safetensors", 5667280),
            ("vae_approx/xlvaeapp.pth", 213777),
            ("clip/clip_l.safetensors", 246144152),
            ("vae/ponyDiffusionV6XL_vae.safetensors", 334641162),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
        ],
        "download_links": [
        "【必要】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_base_simpleai_1214.zip"
        ]
    },
    "extension_package": {
        "id": 2,
        "name": "[2]增强模型包",
        "note": "功能性补充|显存需求：★★ 速度：★★★☆",
        "files": [
            ("embeddings/unaestheticXLhk1.safetensors", 33296),
            ("embeddings/unaestheticXLv31.safetensors", 33296),
            ("inpaint/inpaint_v25.fooocus.patch", 2580722369),
            ("inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("inpaint/sam_vit_l_0b3195.pth", 1249524607),
            ("layer_model/layer_xl_bg2ble.safetensors", 701981624),
            ("layer_model/layer_xl_transparent_attn.safetensors", 743352688),
            ("llms/nllb-200-distilled-600M/pytorch_model.bin", 2460457927),
            ("llms/nllb-200-distilled-600M/sentencepiece.bpe.model", 4852054),
            ("llms/nllb-200-distilled-600M/tokenizer.json", 17331176),
            ("loras/FilmVelvia3.safetensors", 151108832),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("loras/SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 912593164),
            ("safety_checker/stable-diffusion-safety-checker.bin", 1216067303),
            ("unet/iclight_sd15_fbc_unet_ldm.safetensors", 1719167896),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("vae/ponyDiffusionV6XL_vae.safetensors", 334641162),
            ("vae/sdxl_fp16.vae.safetensors", 167335342),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_enhance_simpleai_0908.zip"
        ]
    },
        "kolors_package": {
        "id": 3,
        "name": "[3]可图扩展包",
        "note": "可图文生图|显存需求：★★ 速度：★★★☆",
        "files": [
            ("diffusers/Kolors/model_index.json", 427),
            ("diffusers/Kolors/MODEL_LICENSE", 14920),
            ("diffusers/Kolors/README.md", 4707),
            ("diffusers/Kolors/scheduler/scheduler_config.json", 606),
            ("diffusers/Kolors/text_encoder/config.json", 1323),
            ("diffusers/Kolors/text_encoder/configuration_chatglm.py", 2332),
            ("diffusers/Kolors/text_encoder/modeling_chatglm.py", 55722),
            ("diffusers/Kolors/text_encoder/pytorch_model-00001-of-00007.bin", 1827781090),
            ("diffusers/Kolors/text_encoder/pytorch_model-00002-of-00007.bin", 1968299480),
            ("diffusers/Kolors/text_encoder/pytorch_model-00003-of-00007.bin", 1927415036),
            ("diffusers/Kolors/text_encoder/pytorch_model-00004-of-00007.bin", 1815225998),
            ("diffusers/Kolors/text_encoder/pytorch_model-00005-of-00007.bin", 1968299544),
            ("diffusers/Kolors/text_encoder/pytorch_model-00006-of-00007.bin", 1927415036),
            ("diffusers/Kolors/text_encoder/pytorch_model-00007-of-00007.bin", 1052808542),
            ("diffusers/Kolors/text_encoder/pytorch_model.bin.index.json", 20437),
            ("diffusers/Kolors/text_encoder/quantization.py", 14692),
            ("diffusers/Kolors/text_encoder/tokenization_chatglm.py", 12223),
            ("diffusers/Kolors/text_encoder/tokenizer.model", 1018370),
            ("diffusers/Kolors/text_encoder/tokenizer_config.json", 249),
            ("diffusers/Kolors/text_encoder/vocab.txt", 1018370),
            ("diffusers/Kolors/tokenizer/tokenization_chatglm.py", 12223),
            ("diffusers/Kolors/tokenizer/tokenizer.model", 1018370),
            ("diffusers/Kolors/tokenizer/tokenizer_config.json", 249),
            ("diffusers/Kolors/tokenizer/vocab.txt", 1018370),
            ("diffusers/Kolors/unet/config.json", 1785),
            ("diffusers/Kolors/unet/diffusion_pytorch_model.fp16.safetensors", 0),
            ("diffusers/Kolors/vae/diffusion_pytorch_model.fp16.safetensors", 0),
            ("diffusers/Kolors/vae/config.json", 611),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("checkpoints/kolors_unet_fp16.safetensors", 5159140240),
            ("vae/sdxl_fp16.vae.safetensors", 167335342),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_kolors_fp16_simpleai_0909.zip"
        ]
    },
        "additional_package": {
        "id": 4,
        "name": "[4]额外模型包",
        "note": "动漫/混元/PG/小马/写实/SD3|显存需求：★★ 速度：★★★☆",
        "files": [
            ("checkpoints/animaPencilXL_v500.safetensors", 6938041144),
            ("checkpoints/hunyuan_dit_1.2.safetensors", 8240228270),
            ("checkpoints/playground-v2.5-1024px.safetensors", 6938040576),
            ("checkpoints/ponyDiffusionV6XL.safetensors", 6938041050),
            ("checkpoints/realisticStockPhoto_v20.safetensors", 6938054242),
            ("checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors", 10867168284),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_ckpt_SD3_HY_PonyV6_PGv25_aPencilXL_rsPhoto_simpleai_0909.zip"
        ]
    },
        "Flux_package": {
        "id": 5,
        "name": "[5]Flux全量包",
        "note": "Flux官方全量|显存需求：★★★★★ 速度：★★",
        "files": [
            ("checkpoints/flux1-dev.safetensors", 23802932552),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp16.safetensors", 9787841024),
            ("vae/ae.safetensors", 335304388),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_flux1_fp16_simpleai_0909.zip"
        ]
    },
        "Flux_aio_package": {
        "id": 6,
        "name": "[6]Flux_AIO扩展包",
        "note": "Flux全功能[Q5模型]|显存需求：★★★☆ 速度：★★",
        "files": [
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("clip/t5xxl_fp16.safetensors", 9787841024),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("controlnet/flux.1-dev_controlnet_union_pro_2.0.safetensors", 4281779224),
            ("controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("controlnet/parsing_bisenet.pth", 53289463),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("loras/flux1-canny-dev-lora.safetensors", 1244443944),
            ("loras/flux1-depth-dev-lora.safetensors", 1244440512),
            ("checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("vae/ae.safetensors", 335304388),
            ("style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_flux_aio_simpleai_1214.zip"
        ]
    },
        "SD15_aio_package": {
        "id": 7,
        "name": "[7]SD1.5_AIO扩展包",
        "note": "SD1.5全功能|显存需求：★ 速度：★★★★",
        "files": [
            ("checkpoints/realisticVisionV60B1_v51VAE.safetensors", 2132625894),
            ("loras/sd_xl_offset_example-lora_1.0.safetensors", 49553604),
            ("clip/sd15_clip_model.fp16.safetensors", 246144864),
            ("controlnet/control_v11f1e_sd15_tile_fp16.safetensors", 722601104),
            ("controlnet/control_v11f1p_sd15_depth_fp16.safetensors", 722601100),
            ("controlnet/control_v11p_sd15_canny_fp16.safetensors", 722601100),
            ("controlnet/control_v11p_sd15_openpose_fp16.safetensors", 722601100),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("inpaint/sd15_powerpaint_brushnet_clip_v2_1.bin", 492401329),
            ("inpaint/sd15_powerpaint_brushnet_v2_1.safetensors", 3544366408),
            ("insightface/models/buffalo_l/1k3d68.onnx", 143607619),
            ("insightface/models/buffalo_l/2d106det.onnx", 5030888),
            ("insightface/models/buffalo_l/det_10g.onnx", 16923827),
            ("insightface/models/buffalo_l/genderage.onnx", 1322532),
            ("insightface/models/buffalo_l/w600k_r50.onnx", 174383860),
            ("ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("ipadapter/ip-adapter-faceid-plusv2_sd15.bin", 156558509),
            ("ipadapter/ip-adapter_sd15.safetensors", 44642768),
            ("loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors", 51059544),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604)
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_sd15_aio_simpleai_1214.zip"
        ]
    },
        "Kolors_aio_package": {
        "id": 8,
        "name": "[8]Kolors_AIO扩展包",
        "note": "可图全功能|显存需求：★★★ 速度：★★★",
        "files": [
            ("checkpoints/kolors_unet_fp16.safetensors", 5159140240),
            ("clip_vision/kolors_clip_ipa_plus_vit_large_patch14_336.bin", 1711974081),
            ("controlnet/kolors_controlnet_pose.safetensors", 2526129624),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("diffusers/Kolors/model_index.json", 427),
            ("diffusers/Kolors/MODEL_LICENSE", 14920),
            ("diffusers/Kolors/README.md", 4707),
            ("diffusers/Kolors/scheduler/scheduler_config.json", 606),
            ("diffusers/Kolors/text_encoder/config.json", 1323),
            ("diffusers/Kolors/text_encoder/configuration_chatglm.py", 2332),
            ("diffusers/Kolors/text_encoder/modeling_chatglm.py", 55722),
            ("diffusers/Kolors/text_encoder/pytorch_model-00001-of-00007.bin", 1827781090),
            ("diffusers/Kolors/text_encoder/pytorch_model-00002-of-00007.bin", 1968299480),
            ("diffusers/Kolors/text_encoder/pytorch_model-00003-of-00007.bin", 1927415036),
            ("diffusers/Kolors/text_encoder/pytorch_model-00004-of-00007.bin", 1815225998),
            ("diffusers/Kolors/text_encoder/pytorch_model-00005-of-00007.bin", 1968299544),
            ("diffusers/Kolors/text_encoder/pytorch_model-00006-of-00007.bin", 1927415036),
            ("diffusers/Kolors/text_encoder/pytorch_model-00007-of-00007.bin", 1052808542),
            ("diffusers/Kolors/text_encoder/pytorch_model.bin.index.json", 20437),
            ("diffusers/Kolors/text_encoder/quantization.py", 14692),
            ("diffusers/Kolors/text_encoder/tokenization_chatglm.py", 12223),
            ("diffusers/Kolors/text_encoder/tokenizer.model", 1018370),
            ("diffusers/Kolors/text_encoder/tokenizer_config.json", 249),
            ("diffusers/Kolors/text_encoder/vocab.txt", 1018370),
            ("diffusers/Kolors/tokenizer/tokenization_chatglm.py", 12223),
            ("diffusers/Kolors/tokenizer/tokenizer.model", 1018370),
            ("diffusers/Kolors/tokenizer/tokenizer_config.json", 249),
            ("diffusers/Kolors/tokenizer/vocab.txt", 1018370),
            ("diffusers/Kolors/unet/config.json", 1785),
            ("diffusers/Kolors/vae/config.json", 611),
            ("diffusers/Kolors/unet/diffusion_pytorch_model.fp16.safetensors", 0),
            ("diffusers/Kolors/vae/diffusion_pytorch_model.fp16.safetensors", 0),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("ipadapter/kolors_ipa_faceid_plus.bin", 2385842603),
            ("ipadapter/kolors_ip_adapter_plus_general.bin", 1013163359),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("loras/sd_xl_offset_example-lora_1.0.safetensors", 49553604),
            ("unet/kolors_inpainting.safetensors", 5159169040),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("vae/sdxl_fp16.vae.safetensors", 167335342)
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_kolors_aio_simpleai_1214.zip"
        ]
    },
        "SD3x_medium_package": {
        "id": 9,
        "name": "[9]SD3.5_medium扩展包",
        "note": "SD3.5中号文生图|显存需求：★★ 速度：★★★",
        "files": [
            ("checkpoints/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors", 11638004202),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("vae/sd3x_fp16.vae.safetensors", 167666654),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"
        ]
    },
        "SD3x_large_package": {
        "id": 10,
        "name": "[10]SD3.5_Large 扩展包",
        "note": "SD3.5大号文生图|显存需求：★★★★★ 速度：★★",
        "files": [
            ("checkpoints/sd3.5_large.safetensors", 16460379262),
            ("clip/clip_g.safetensors", 1389382176),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp16.safetensors", 9787841024),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("vae/sd3x_fp16.vae.safetensors", 167666654),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_sd35_large_clips_simpleai_1214.zip"
        ]
    },
        "MiniCPM_package": {
        "id": 11,
        "name": "[11]MiniCPMv26反推扩展包",
        "note": "本地多模态大语言模型|显存需求：★★ 速度：★★",
        "files": [
            ("llms/MiniCPMv2_6-prompt-generator/.gitattributes", 1657),
            ("llms/MiniCPMv2_6-prompt-generator/.mdl", 49),
            ("llms/MiniCPMv2_6-prompt-generator/.msc", 1655),
            ("llms/MiniCPMv2_6-prompt-generator/.mv", 36),
            ("llms/MiniCPMv2_6-prompt-generator/added_tokens.json", 629),
            ("llms/MiniCPMv2_6-prompt-generator/config.json", 1951),
            ("llms/MiniCPMv2_6-prompt-generator/configuration.json", 27),
            ("llms/MiniCPMv2_6-prompt-generator/configuration_minicpm.py", 3280),
            ("llms/MiniCPMv2_6-prompt-generator/generation_config.json", 121),
            ("llms/MiniCPMv2_6-prompt-generator/image_processing_minicpmv.py", 16579),
            ("llms/MiniCPMv2_6-prompt-generator/merges.txt", 1671853),
            ("llms/MiniCPMv2_6-prompt-generator/modeling_minicpmv.py", 15738),
            ("llms/MiniCPMv2_6-prompt-generator/modeling_navit_siglip.py", 41835),
            ("llms/MiniCPMv2_6-prompt-generator/preprocessor_config.json", 714),
            ("llms/MiniCPMv2_6-prompt-generator/processing_minicpmv.py", 9962),
            ("llms/MiniCPMv2_6-prompt-generator/pytorch_model-00001-of-00002.bin", 4454731094),
            ("llms/MiniCPMv2_6-prompt-generator/pytorch_model-00002-of-00002.bin", 1503635286),
            ("llms/MiniCPMv2_6-prompt-generator/pytorch_model.bin.index.json", 233389),
            ("llms/MiniCPMv2_6-prompt-generator/README.md", 2124),
            ("llms/MiniCPMv2_6-prompt-generator/resampler.py", 34699),
            ("llms/MiniCPMv2_6-prompt-generator/special_tokens_map.json", 1041),
            ("llms/MiniCPMv2_6-prompt-generator/test.py", 1162),
            ("llms/MiniCPMv2_6-prompt-generator/tokenization_minicpmv_fast.py", 1659),
            ("llms/MiniCPMv2_6-prompt-generator/tokenizer.json", 7032006),
            ("llms/MiniCPMv2_6-prompt-generator/tokenizer_config.json", 5663),
            ("llms/MiniCPMv2_6-prompt-generator/vocab.json", 2776833),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_minicpm_v2.6_prompt_simpleai_1224.zip"
        ]
    },
        "happy_package": {
        "id": 12,
        "name": "[12]贺年卡",
        "note": "贺卡预设|显存需求：★★★ 速度：★★",
        "files": [
            ("loras/flux_graffiti_v1.safetensors", 612893792),
            ("loras/kolors_crayonsketch_e10.safetensors", 170566628),
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("vae/ae.safetensors", 335304388),
            ("checkpoints/kolors_unet_fp16.safetensors", 5159140240),
            ("clip_vision/kolors_clip_ipa_plus_vit_large_patch14_336.bin", 1711974081),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("diffusers/Kolors/model_index.json", 427),
            ("diffusers/Kolors/MODEL_LICENSE", 14920),
            ("diffusers/Kolors/README.md", 4707),
            ("diffusers/Kolors/scheduler/scheduler_config.json", 606),
            ("diffusers/Kolors/text_encoder/config.json", 1323),
            ("diffusers/Kolors/text_encoder/configuration_chatglm.py", 2332),
            ("diffusers/Kolors/text_encoder/modeling_chatglm.py", 55722),
            ("diffusers/Kolors/text_encoder/pytorch_model-00001-of-00007.bin", 1827781090),
            ("diffusers/Kolors/text_encoder/pytorch_model-00002-of-00007.bin", 1968299480),
            ("diffusers/Kolors/text_encoder/pytorch_model-00003-of-00007.bin", 1927415036),
            ("diffusers/Kolors/text_encoder/pytorch_model-00004-of-00007.bin", 1815225998),
            ("diffusers/Kolors/text_encoder/pytorch_model-00005-of-00007.bin", 1968299544),
            ("diffusers/Kolors/text_encoder/pytorch_model-00006-of-00007.bin", 1927415036),
            ("diffusers/Kolors/text_encoder/pytorch_model-00007-of-00007.bin", 1052808542),
            ("diffusers/Kolors/text_encoder/pytorch_model.bin.index.json", 20437),
            ("diffusers/Kolors/text_encoder/quantization.py", 14692),
            ("diffusers/Kolors/text_encoder/tokenization_chatglm.py", 12223),
            ("diffusers/Kolors/text_encoder/tokenizer.model", 1018370),
            ("diffusers/Kolors/text_encoder/tokenizer_config.json", 249),
            ("diffusers/Kolors/text_encoder/vocab.txt", 1018370),
            ("diffusers/Kolors/tokenizer/tokenization_chatglm.py", 12223),
            ("diffusers/Kolors/tokenizer/tokenizer.model", 1018370),
            ("diffusers/Kolors/tokenizer/tokenizer_config.json", 249),
            ("diffusers/Kolors/tokenizer/vocab.txt", 1018370),
            ("diffusers/Kolors/unet/config.json", 1785),
            ("diffusers/Kolors/vae/config.json", 611),
            ("ipadapter/kolors_ipa_faceid_plus.bin", 2385842603),
            ("ipadapter/kolors_ip_adapter_plus_general.bin", 1013163359),
            ("vae/sdxl_fp16.vae.safetensors", 167335342),
        ],
        "download_links": [
        "【选配】贺年卡基于FluxAIO、可图AIO扩展，请检查所需包体。Lora点击生成会自动下载。"
        ]
    },
        "clothing_package": {
        "id": 13,
        "name": "[13]换装包",
        "note": "万物迁移[Q4模型]|显存需求：★★★ 速度：★★",
        "files": [
            ("inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("inpaint/GroundingDINO_SwinT_OGC.cfg.py", 1006),
            ("checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("clip/clip_l.safetensors", 246144152), 
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("vae/ae.safetensors", 335304388),
            ("inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("style_models/flux1-redux-dev.safetensors", 129063232),
            ("rembg/General.safetensors", 884878856),
            ("loras/comfyui_subject_lora16.safetensors", 153268392)
        ],
        "download_links": [
        "【选配】换装基于增强包，FluxAIO组件扩展，请检查所需包体。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "3DPurikura_package": {
        "id": 14,
        "name": "[14]3D大头贴",
        "note": "3D个性头像|显存需求：★★ 速度：★★",
        "files": [
            ("checkpoints/SDXL_Yamers_Cartoon_Arcadia.safetensors", 6938040714),
            ("upscale_models/RealESRGAN_x4plus_anime_6B.pth", 17938799),
            ("rembg/Portrait.safetensors", 884878856),
            ("ipadapter/ip-adapter-faceid-plusv2_sdxl.bin", 1487555181),
            ("ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("insightface/models/buffalo_l/1k3d68.onnx", 143607619),
            ("insightface/models/buffalo_l/2d106det.onnx", 5030888),
            ("insightface/models/buffalo_l/det_10g.onnx", 16923827),
            ("insightface/models/buffalo_l/genderage.onnx", 1322532),
            ("insightface/models/buffalo_l/w600k_r50.onnx", 174383860),
            ("loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors", 371842896),
            ("loras/StickersRedmond.safetensors", 170540036)
        ],
        "download_links": [
        "【选配】浏览器进入模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "x1-okremovebg_package": {
        "id": 15,
        "name": "[15]一键抠图",
        "note": "抠图去背景神器|显存需求：★ 速度：★★★★★",
        "files": [
            ("rembg/ckpt_base.pth", 367520613),
            ("rembg/RMBG-1.4.pth", 176718373),
            ("rembg/General.safetensors", 884878856),
            ("rembg/Portrait.safetensors", 884878856)
        ],
        "download_links": [
        "【选配】浏览器进入模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "x2-okimagerepair_package": {
        "id": 16,
        "name": "[16]一键修复",
        "note": "上色、修复模糊、旧照片|显存需求：★★★ 速度：★☆",
        "files": [
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("checkpoints/LEOSAM_HelloWorldXL_70.safetensors", 6938040682),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("vae/ae.safetensors", 335304388),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("controlnet/detection_Resnet50_Final.pth", 109497761),
            ("controlnet/facerestore_models/codeformer-v0.1.0.pth", 376637898),
            ("controlnet/facerestore_models/GFPGANv1.4.pth", 348632874),
            ("ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427),
            ("upscale_models/4xNomos8kSCHAT-L.pth", 331564661)
        ],
        "download_links": [
        "【选配】浏览器进入模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "x3-swapface_package": {
        "id": 17,
        "name": "[17]一键换脸",
        "note": "高精度换脸|显存需求：★★★ 速度：★★",
        "files": [
            ("checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("vae/ae.safetensors", 335304388),
            ("loras/flux1-turbo.safetensors", 694082424),
            ("inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("inpaint/GroundingDINO_SwinT_OGC.cfg.py", 1006),
            ("inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("style_models/flux1-redux-dev.safetensors", 129063232),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("loras/comfyui_portrait_lora64.safetensors",612742344)
        ],
        "download_links": [
        "【选配】浏览器进入模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "Flux_aio_plus_package": {
        "id": 18,
        "name": "[18]Flux_AIO_plus扩展包",
        "note": "Flux全功能[fp8模型]|显存需求：★★★★ 速度：★★☆",
        "files": [
            ("checkpoints/flux1-dev-fp8.safetensors", 11901525888),
            ("checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("clip/t5xxl_fp16.safetensors", 9787841024),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("controlnet/flux.1-dev_controlnet_union_pro_2.0.safetensors", 4281779224),
            ("controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("controlnet/parsing_bisenet.pth", 53289463),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("loras/flux1-canny-dev-lora.safetensors", 1244443944),
            ("loras/flux1-depth-dev-lora.safetensors", 1244440512),
            ("checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("vae/ae.safetensors", 335304388),
            ("style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": [
        "【选配】基于FluxAIO扩展包扩展",
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/flux1-dev-fp8.safetensors"
        ]
    },
        "clothing_plus_package": {
        "id": 19,
        "name": "[19]换装plus包",
        "note": "万物迁移[fp8模型]|显存需求：★★★☆ 速度：★★★",
        "files": [
            ("inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("inpaint/GroundingDINO_SwinT_OGC.cfg.py", 1006),
            ("checkpoints/flux1-fill-dev_fp8.safetensors", 11902532704),
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("vae/ae.safetensors", 335304388),
            ("inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("style_models/flux1-redux-dev.safetensors", 129063232),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("rembg/General.safetensors", 884878856),
            ("loras/comfyui_subject_lora16.safetensors", 153268392)
        ],
        "download_links": [
        "【选配】换装基于增强包，FluxAIO组件扩展，请检查所需包体。部分文件、Lora点击生成会自动下载。",
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/flux1-fill-dev_fp8.safetensors"
        ]
    },
        "eraser-a_package": {
        "id": 20,
        "name": "[20]一键消除",
        "note": "一键消除|显存需求：★★ 速度：★★☆",
        "files": [
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("vae/ae.safetensors", 335304388),
            ("loras/fill_remove.safetensors",104667792),
            ("style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": [
        "【选配】一键消除基于FluxAIO组件扩展，请检查所需包体。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "Illustrious_package": {
        "id": 21,
        "name": "[21]光辉模型包",
        "note": "支持NoobAI/光辉模型文生图|显存需求：★★ 速度：★★★☆",
        "files": [
            ("checkpoints/miaomiaoHarem_v15b.safetensors", 6938043202)
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/miaomiaoHarem_v15b.safetensors"
        ]
    },
        "Illustrious_aio_package": {
        "id": 22,
        "name": "[22]光辉AIO扩展包",
        "note": "NoobAI/光辉模型全功能图生图|显存需求：★★★ 速度：★★★",
        "files": [
            ("checkpoints/miaomiaoHarem_v15b.safetensors", 6938043202),
            ("ipadapter/noob_ip_adapter.bin", 1396798350),
            ("upscale_models/RealESRGAN_x4plus_anime_6B.pth", 17938799),
            ("ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("controlnet/noob_sdxl_controlnet_inpainting.safetensors", 5004167832),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408)
        ],
        "download_links": [
        "【选配】浏览器进入模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。",
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/miaomiaoHarem_v15b.safetensors"
        ]
    },
        "StyleTransfer_package": {
        "id": 23,
        "name": "[23]风格转绘扩展包",
        "note": "多种图像风格转绘|显存需求：★★★ 速度：★★★",
        "files": [
            ("checkpoints/LEOSAM_HelloWorldXL_70.safetensors", 6938040682),
            ("checkpoints/miaomiaoHarem_v15b.safetensors", 6938043202),
            ("checkpoints/SDXL_Yamers_Cartoon_Arcadia.safetensors", 6938040714),
            ("loras/SDXL_claymate.safetensors", 912561180),
            ("loras/SDXL_crayon.safetensors", 340776492),
            ("loras/SDXL_cute.safetensors", 681244276),
            ("loras/SDXL_ghibli.safetensors", 681268820),
            ("loras/SDXL_inkpainting.safetensors", 228466036),
            ("loras/SDXL_oilpainting.safetensors", 202694420),
            ("loras/SDXL_papercut.safetensors", 456489140),
            ("loras/SDXL_watercolor.safetensors", 228458788),
            ("loras/Illustrious_pixelart.safetensors", 228504612),
            ("loras/noob_pvc.safetensors", 607394012),
            ("loras/SDXL_lineart.safetensors", 170540028),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("controlnet/lllyasviel/Annotators/sk_model.pth", 17173511),
            ("controlnet/lllyasviel/Annotators/sk_model2.pth", 17173511),
            ("controlnet/lllyasviel/Annotators/ControlNetHED.pth", 29444406),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("ipadapter/ip-adapter-faceid-plusv2_sdxl.bin", 1487555181),
            ("ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("ipadapter/noob_ip_adapter.bin", 1396798350),
            ("controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427),
            ("insightface/models/buffalo_l/1k3d68.onnx", 143607619),
            ("insightface/models/buffalo_l/2d106det.onnx", 5030888),
            ("insightface/models/buffalo_l/det_10g.onnx", 16923827),
            ("insightface/models/buffalo_l/genderage.onnx", 1322532),
            ("insightface/models/buffalo_l/w600k_r50.onnx", 174383860)
        ],
        "download_links": [
        "【选配】浏览器进入模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "okdepthstatue_package": {
        "id": 24,
        "name": "[24]深度图、雕像扩展包",
        "note": "深度图、白瓷雕像风格扩展|显存需求：★★ 速度：★★★★★",
        "files": [
            ("checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("controlnet/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth", 1341395338),
            ("controlnet/lllyasviel/Annotators/sk_model.pth", 17173511),
            ("controlnet/lllyasviel/Annotators/sk_model2.pth", 17173511),
            ("ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427)
        ],
        "download_links": [
        "【选配】浏览器进入模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "Framepack_package": {
        "id": 25,
        "name": "[25]Framepack视频扩展包",
        "note": "图像转视频功能支持|显存需求：★★★ 速度：★",
        "files": [
            ("checkpoints/FramePackI2V_HY_fp8_e4m3fn.safetensors", 16331849976),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/llava_llama3_fp8_scaled.safetensors", 9091392483),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("vae/hunyuan_video_vae_bf16.safetensors", 492984198)
        ],
        "download_links": [
            "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/FramePackI2V_HY_fp8_e4m3fn.safetensors",
            "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/clip/llava_llama3_fp8_scaled.safetensors",
            "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/vae/hunyuan_video_vae_bf16.safetensors"
        ]
    }
}

MANUAL_DOWNLOAD_MAP = {
    "checkpoints": [
        "animaPencilXL_v500.jpg",
        "flux1-dev.jpg",
        "flux1-dev-fp8.jpg",
        "flux1-fill-dev_fp8.jpg",
        "flux1-fill-dev-hyp8-Q4_K_S.jpg",
        "flux-hyp8-Q5_K_M.jpg",
        "hunyuan_dit_1.2.jpg",
        "juggernautXL_juggXIByRundiffusion.jpg",
        "kolors_unet_fp16.jpg",
        "LEOSAM_HelloWorldXL_70.jpg",
        "miaomiaoHarem_v15b.jpg",
        "playground-v2.5-1024px.jpg",
        "ponyDiffusionV6XL.jpg",
        "realisticStockPhoto_v20.jpg",
        "realisticVisionV60B1_v51VAE.jpg",
        "sd3.5_large.jpg",
        "sd3.5_medium_incl_clips_t5xxlfp8scaled.jpg",
        "sd3_medium_incl_clips_t5xxlfp8.jpg",
        "SDXL_Yamers_Cartoon_Arcadia.jpg"
    ],
    "loras": [
        "comfyui_portrait_lora64.jpg",
        "comfyui_subject_lora16.jpg",
        "fill_remove.jpg",
        "FilmVelvia3.jpg",
        "flux_graffiti_v1.jpg",
        "flux1-canny-dev-lora.jpg",
        "flux1-depth-dev-lora.jpg",
        "flux1-turbo.jpg",
        "Hyper-SDXL-8steps-lora.jpg",
        "ip-adapter-faceid-plusv2_sd15_lora.jpg",
        "ip-adapter-faceid-plusv2_sdxl_lora.jpg",
        "kolors_crayonsketch_e10.jpg",
        "sd_xl_offset_example-lora_1.0.jpg",
        "SDXL_FILM_PHOTOGRAPHY_STYLE_V1.jpg",
        "sdxl_hyper_sd_4step_lora.jpg",
        "sdxl_lightning_4step_lora.jpg",
        "StickersRedmond.jpg",
        "Illustrious_pixelart.jpg",
        "noob_pvc.jpg",
        "SDXL_claymate.jpg", 
        "SDXL_crayon.jpg",
        "SDXL_cute.jpg",
        "SDXL_ghibli.jpg",
        "SDXL_inkpainting.jpg",
        "SDXL_lineart.jpg",
        "SDXL_oilpainting.jpg",
        "SDXL_papercut.jpg",
        "SDXL_watercolor.jpg"
    ]
}

MANUAL_DOWNLOAD_LIST = [
    f"https://hf-mirror.com/windecay/SimpleSDXL2/resolve/main/SimpleModels/{category}/{filename}"
    for category, files in MANUAL_DOWNLOAD_MAP.items() 
    for filename in files
]

OBSOLETE_MODELS = [
    "flux1-dev-bnb-nf4.safetensors",
    "flux1-schnell-bnb-nf4.safetensors",
    "sdxl_hyper_sd_4step_lora.safetensors",
    "xinsir_cn_openpose_sdxl_1.0.safetensors",
    "fooocus_xl_cpds_128.safetensors",
    "control-lora-canny-rank128.safetensors",
    "flux.1-dev_controlnet_union_pro.safetensors",
    "illustriousXL_controlnet_tile_v2.5.safetensors",
    "kolors_controlnet_canny.safetensors",
    "kolors_controlnet_depth.safetensors",
    "noob_sdxl_controlnet_canny.fp16.safetensors",
    "noob_sdxl_controlnet_depth.fp16.safetensors",
    "noob_sdxl_controlnet_pose.fp16.safetensors",
    "NoobAI-XL-v1.1.safetensors"
]
def main():
    print()
    print_colored("★★★★★★★★★★★★★★★★★★欢迎使用SimpleAI模型检测器★★★★★★★★★★★★★★★★★★", Fore.CYAN)
    time.sleep(0.1)
    print()
    check_python_embedded()
    time.sleep(0.1)
    check_script_file()
    time.sleep(0.1)
    total_virtual = get_total_virtual_memory()
    time.sleep(0.1)
    check_virtual_memory(total_virtual)
    time.sleep(0.1)
    print_instructions()
    time.sleep(0.1)
    validate_files(packages)
    print()
    print_colored("★★★★★★★★★★★★★★★★★★检测已结束执行自动下载模块★★★★★★★★★★★★★★★★★★", Fore.CYAN)

if __name__ == "__main__":
    main()
    print()
    while True:
        print(f">>>按下【{Fore.YELLOW}Enter回车{Style.RESET_ALL}】----------------启动全部文件下载<<<     备注：支持断点续传，顺序从小文件开始。")
        print(f">>>输入【{Fore.YELLOW}包体编号{Style.RESET_ALL}】+【{Fore.YELLOW}回车{Style.RESET_ALL}】----------启动预置包补全<<<     备注：若速度太慢直接拿链接用P2P软件下载")
        print(f">>>数字【{Fore.YELLOW}0{Style.RESET_ALL}】+【{Fore.YELLOW}回车{Style.RESET_ALL}】-清理日志/下载/图片缓存与坏文件<<<     备注：△谨慎执行。慎防误删私有模型")
        print(f">>>输入【{Fore.YELLOW}DEL{Style.RESET_ALL}】【{Fore.YELLOW}包体编号{Style.RESET_ALL}】----------删除已有包体文件<<<     备注：△谨慎执行。自动避开关联文件")
        print(f">>>输入【{Fore.YELLOW}R{Style.RESET_ALL}】+【{Fore.YELLOW}回车{Style.RESET_ALL}】-----------------------重新检测<<<     备注：再玩一遍，玩不腻")
        print(f">>>输入【{Fore.YELLOW}S{Style.RESET_ALL}】+【{Fore.YELLOW}回车{Style.RESET_ALL}】-----------------下载模型预览图<<<     备注：只下载checkpoints和lora预览图")

        user_input = input("请选择操作(不需要括号):")

        if user_input == "":
            print("※启动自动下载模块,支持断点续传，关闭窗口可中断。")
            auto_download_missing_files_with_retry(max_threads=5)
        elif user_input.isdigit():
            package_id = int(user_input)
            selected_package = None
            for package_name, package_info in packages.items():
                if package_info["id"] == package_id:
                    selected_package = package_info
                    break

            if selected_package:
                get_download_links_for_package({package_name: selected_package}, "downloadlist.txt")
                auto_download_missing_files_with_retry(max_threads=5)
            elif package_id == 0:
                delete_partial_files()
                delete_specific_image_files()
                delete_log_files()
            else:
                print(f"{Fore.RED}△包体编号{package_id} 无效，请输入正确的包体ID。{Style.RESET_ALL}")

        elif user_input.lower().startswith("del"):
            try:
                path_mapping = load_model_paths()
                package_id_str = user_input[3:].strip()
                
                if not package_id_str.isdigit():
                    print(f"{Fore.RED}△输入格式错误，请输入类似 del1 来删除对应包体。{Style.RESET_ALL}")
                else:
                    package_id = int(package_id_str)
                    selected_package = None
                    
                    for pkg_name, pkg_info in packages.items():
                        if pkg_info["id"] == package_id:
                            selected_package = pkg_info
                            break
                    
                    if selected_package:
                        print(f"{Fore.YELLOW}△即将删除包体：[{selected_package['name']}]{Style.RESET_ALL}")
                        delete_package(pkg_name, packages)
                    else:
                        print(f"{Fore.RED}△无效的包体编号！{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}△删除过程中发生错误：{str(e)}{Style.RESET_ALL}")
        elif user_input.lower() == "r":
            print("重新检测文件...")
            validate_files(packages)
        elif user_input.lower() == "s":
            print("下载预览图...")
            trigger_manual_download()
        else:
            print(f"{Fore.RED}△无效的输入，请输入回车或有效的包体编号（不需要括号）。{Style.RESET_ALL}")