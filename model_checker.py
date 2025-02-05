import os
import sys
import time
import requests
from tqdm import tqdm
from colorama import init, Fore, Style
import threading
import atexit

def cleanup():
    if os.path.exists("downloadlist.txt"):
        os.remove("downloadlist.txt")
        print("已删除 'downloadlist.txt' 文件。")

# 在程序开始时注册退出时执行清理操作
atexit.register(cleanup)

# 初始化 colorama
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
    # 获取主程序路径
    script_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SimpleSDXL", "entry_with_update.py")

    # 检查文件是否存在
    if os.path.exists(script_file):
        print_colored("√找到主程序目录", Fore.GREEN)
    else:
        print_colored("×未找到主程序目录，请检查脚本位置", Fore.RED)
        input("按任意键继续。")
        sys.exit(1)

    # 验证主程序目录的层级是否超过 2
    base_dir = os.path.dirname(os.path.dirname(script_file))  # 获取 "SimpleSDXL" 的上级目录
    directory_level = len(base_dir.split(os.sep))  # 按系统分隔符分割路径，计算层级

    if directory_level <= 2:  # 如果层级小于等于 2，则提示错误
        print_colored("×主程序目录层级不足，可能会导致脚本结果有误。请按照安装视频指引先建立SimpleAI主文件夹", Fore.RED)
    else: 
        print_colored("√主程序目录层级验证通过", Fore.GREEN)

def get_total_virtual_memory():
    import psutil
    try:
        virtual_mem = psutil.virtual_memory().total  # 物理内存
        swap_mem = psutil.swap_memory().total        # 交换分区
        total_virtual_memory = virtual_mem + swap_mem
        return total_virtual_memory
    except ImportError:
        print_colored("无法导入 psutil 模块，跳过内存检查", Fore.YELLOW)
        return None

def check_virtual_memory(total_virtual):
    if total_virtual is None:
        return
    total_gb = total_virtual / (1024 ** 3)
    if total_gb < 40:
        print_colored("警告：系统虚拟内存小于40GB，会禁用部分预置包，请参考安装视频教程设置系统虚拟内存。", Fore.YELLOW)
    else:
        print_colored("√系统虚拟内存充足", Fore.GREEN)
    print(f"系统总虚拟内存: {total_gb:.2f} GB")

def find_simplemodels_dir(start_path):
    """
    从当前路径开始，逐级向上查找 SimpleModels 目录
    """
    current_dir = start_path
    while current_dir != os.path.dirname(current_dir):  # 防止进入根目录
        simplemodels_path = os.path.join(current_dir, "SimpleModels")
        if os.path.isdir(simplemodels_path):
            return simplemodels_path
        current_dir = os.path.dirname(current_dir)
    return None

def normalize_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simplemodels_dir = find_simplemodels_dir(script_dir)
    
    if simplemodels_dir:
        if path.startswith("SimpleModels"):
            normalized_path = os.path.join(simplemodels_dir, path[len("SimpleModels/"):])
        else:
            normalized_path = os.path.join(simplemodels_dir, path)
        return os.path.abspath(normalized_path)
    else:
        return os.path.abspath(path)

def typewriter_effect(text, delay=0.01):
    for char in text:
        print(char, end='', flush=True)  # 确保即时输出字符
        time.sleep(delay)
        
    print()  # 打印换行符
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
    print(f"{Fore.GREEN}★{Style.RESET_ALL}疑难杂症进QQ群求助：938075852{Fore.GREEN}★{Style.RESET_ALL}脚本：✿   冰華 |版本:25.01.27{Fore.GREEN}★{Style.RESET_ALL}")
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
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # 使用字典来存储下载信息，确保路径唯一
    download_files = {}
    missing_package_names = []  # 用于存储缺失文件的包体名称
    package_percentages = {}  # 存储包体百分比信息
    package_sizes = {}  # 存储包体总大小（GB）

    for package_key, package_info in packages.items():
        package_name = package_info["name"]
        package_note = package_info.get("note", "")  # 获取备注信息，如果没有则为空字符串
        files_and_sizes = package_info["files"]
        download_links = package_info["download_links"]

        # 计算包体总大小和非缺失文件的总大小
        total_size = sum([size for _, size in files_and_sizes])  # 计算包体中所有文件的总大小
        total_size_gb = total_size / (1024 ** 3)  # 转换为GB
        non_missing_size = 0  # 非缺失文件的总大小

        print(f"－－－－－－－", end='')
        time.sleep(0.1)
        print(f"校验{package_name}文件－－－－{package_note}")

        missing_files = []
        size_mismatch_files = []
        case_mismatch_files = []

        for expected_path, expected_size in files_and_sizes:
            expected_dir = os.path.join(root, os.path.dirname(expected_path))
            expected_filename = os.path.basename(expected_path)

            if not os.path.exists(expected_dir):
                missing_files.append((expected_path, expected_size))
                continue

            directory_listing = os.listdir(expected_dir)
            actual_filename = next((f for f in directory_listing if f.lower() == expected_filename.lower()), None)

            if actual_filename is None:
                missing_files.append((expected_path, expected_size))
            elif actual_filename != expected_filename:
                case_mismatch_files.append((os.path.join(expected_dir, actual_filename), expected_filename))
            else:
                actual_size = os.path.getsize(os.path.join(expected_dir, actual_filename))
                if actual_size != expected_size:
                    size_mismatch_files.append((os.path.join(expected_dir, actual_filename), actual_size, expected_size))
                else:
                    # 如果文件没有缺失且大小匹配，则累加到非缺失文件的总大小
                    non_missing_size += expected_size

        # 计算非缺失文件的百分比
        if total_size > 0:
            non_missing_percentage = (non_missing_size / total_size) * 100
            package_percentages[package_name] = non_missing_percentage
            package_sizes[package_name] = total_size_gb

        # 处理文件名大小写不匹配
        if case_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件名大小写不匹配，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_filename in case_mismatch_files:
                print(f"文件: {normalize_path(file)}")
                time.sleep(0.1)
                print(f"正确文件名: {expected_filename}")
                
                corrected_file_path = os.path.join(os.path.dirname(file), expected_filename)
                os.rename(file, corrected_file_path)
                print(f"{Fore.GREEN}文件名已更正为: {expected_filename}{Style.RESET_ALL}")

        # 处理文件大小不匹配
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

        # 输出文件验证结果
        if missing_files:
            print(f"{Fore.RED}×{package_name}有文件缺失，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_size in missing_files:
                print(normalize_path(file))
                download_files[file] = expected_size
            # 将缺失包体名称保存到列表中，确保没有重复添加
            if package_name not in missing_package_names:
                missing_package_names.append(package_name)
            # 统一在最后打印下载链接
            if package_info["download_links"]:
                print(f"{Fore.YELLOW}下载链接(若为压缩包，则参考安装视频流程安装):{Style.RESET_ALL}")
                for link in package_info["download_links"]:
                    print(f"{Fore.YELLOW}{link}{Style.RESET_ALL}")
        if not missing_files and not size_mismatch_files and not case_mismatch_files:
            print(f"{Fore.GREEN}√{package_name}文件全部验证通过{Style.RESET_ALL}")

    # 将缺失的包体名称打印出来，同时显示百分比（如果有缺失文件）
    if missing_package_names:
        print(f"{Fore.RED}以下包体缺失文件，请检查并重新下载：{Style.RESET_ALL}")
        for package_name in missing_package_names:
            percentage = package_percentages.get(package_name, 0)
            total_size_gb = package_sizes.get(package_name, 0)
            # 计算尚需下载的 GB
            missing_size_gb = total_size_gb * (1 - (percentage / 100))
            print(f"- {package_name} - 总大小：{total_size_gb:.2f}GB，完整度：{percentage:.2f}%，尚需下载：{missing_size_gb:.2f}GB")

    # 将字典转换为列表并按文件大小排序
    sorted_download_files = sorted(download_files.items(), key=lambda x: x[1])

    # 保存有问题的文件的下载路径到两个不同的txt文件中
    if sorted_download_files:
        with open("downloadlist.txt", "w") as f1, open("缺失模型下载链接.txt", "w") as f2:
            for file, size in sorted_download_files:
                if file == "SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py":
                    link = "https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py"
                else:
                    link = f"https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/{file}"
                
                # 写入带有大小信息的文件
                f1.write(f"{link},{size}\n")
                
                # 写入仅包含下载路径的文件
                f2.write(f"{link}\n")
        print(f"{Fore.YELLOW}>>>所有有问题的文件下载路径已保存到 '缺失模型下载链接.txt'。<<<{Style.RESET_ALL}")

def delete_partial_files():
    """
    从当前脚本位置开始，查找 SimpleModels 目录，并删除其中所有 .partial 和 .corrupted 文件
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 查找 SimpleModels 目录
    simplemodels_dir = find_simplemodels_dir(script_dir)

    if not simplemodels_dir:
        print(f"{Fore.RED}未找到 SimpleModels 目录，请检查目录结构。{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}正在清理目录 '{simplemodels_dir}' 中下载的临时文件与损坏文件...{Style.RESET_ALL}")
    
    files_found = False

    for root, _, files in os.walk(simplemodels_dir):
        for file in files:
            # 删除文件名包含 .partial 或 .corrupted（包括 .corrupted_1 等）的文件
            if ".partial" in file or ".corrupted" in file:
                files_found = True
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)  # 删除文件
                    print(f"{Fore.GREEN}已删除临时文件: {file_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}删除文件时出错: {file_path}, 错误原因: {e}{Style.RESET_ALL}")

    if not files_found:
        print(f">>>未找到需要删除的临时文件<<<")
        print()

    
def download_file_with_resume(link, file_path, position, max_retries=5):
    """
    支持断点续传和重连的下载方法
    :param link: 下载链接
    :param file_path: 文件保存路径
    :param position: 下载位置（用于显示进度条）
    :param max_retries: 最大重试次数
    :return: None
    """
    partial_file_path = file_path + ".partial"
    
    retries = 0
    while retries < max_retries:
        try:
            if os.path.exists(partial_file_path):
                # 获取已下载部分的文件大小
                resume_size = os.path.getsize(partial_file_path)
                headers = {'Range': f"bytes={resume_size}-"}  # 从已下载的位置继续下载
            else:
                resume_size = 0
                headers = {}  # 如果没有文件，就从头开始下载

            response = requests.get(link, stream=True, headers=headers)
            total_size = int(response.headers.get('content-length', 0)) + resume_size
            block_size = 8192  # 8 KB

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 使用.partial文件进行下载
            with open(partial_file_path, 'ab') as file, tqdm(
                    desc=os.path.basename(file_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    position=position,
                    initial=resume_size  # 设置下载的起始位置
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress_bar.update(len(data))

            # 下载完成后重命名临时文件为最终文件名
            os.rename(partial_file_path, file_path)
            print(f"{Fore.GREEN}下载完成：{file_path}{Style.RESET_ALL}")
            return  # 下载成功，跳出循环

        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}下载失败，正在重试... 错误：{e}{Style.RESET_ALL}")
            retries += 1
            time.sleep(5)  # 重试间隔 5 秒
        except Exception as e:
            print(f"{Fore.RED}发生错误：{e}{Style.RESET_ALL}")
            break  # 出现其他错误，跳出循环

    print(f"{Fore.RED}多次尝试后下载失败，请检查网络连接或手动下载文件。{Style.RESET_ALL}")


def auto_download_missing_files_with_retry(max_threads=5):
    """
    启动多线程下载文件，支持断点续传和重试机制
    :param max_threads: 最大线程数
    """
    if not os.path.exists("downloadlist.txt"):
        print("未找到 'downloadlist.txt' 文件。")
        return

    with open("downloadlist.txt", "r") as f:
        links = f.readlines()

    if not links:
        print("没有缺失文件需要下载！")
        return

    threads = []
    active_threads = 0

    for position, line in enumerate(links):
        link, size = line.strip().split(',')
        
        # 处理路径
        if "ShilongLiu/GroundingDINO" in link:
            relative_path = "SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py"
        else:
            relative_path = link.replace("https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/", "")
        
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_path = os.path.join(root, relative_path)
        
        # 启动下载线程
        thread = threading.Thread(target=download_file_with_resume, args=(link, file_path, position))
        threads.append(thread)
        
        active_threads += 1
        thread.start()
        
        # 控制最大线程数
        if active_threads >= max_threads:
            for t in threads:
                t.join()
            threads = []
            active_threads = 0

    for thread in threads:
        thread.join()
        
    # 下载完成后删除downloadlist.txt
    if os.path.exists("downloadlist.txt"):
        os.remove("downloadlist.txt")
        print("下载完成，已删除 'downloadlist.txt' 文件。")


def get_download_links_for_package(packages, download_list_path):
    """
    根据 packages 中的 files 列表生成路径，并与 downloadlist.txt 中的需求进行比对，
    更新 downloadlist.txt 中需要下载的文件，只保留 files 中有的文件链接。
    """
    # 检查 downloadlist.txt 是否存在
    if not os.path.exists(download_list_path):
        print(f"{Fore.RED}>>>downloadlist.txt不存在<<<{Style.RESET_ALL}")
        return []

    # 读取 downloadlist.txt 中的所有文件链接
    with open(download_list_path, "r") as f:
        existing_links = [line.strip().split(",")[0] for line in f.readlines()]

    # 创建一个列表存储需要保留的文件链接
    valid_files = []

    # 遍历 packages 中的 files 列表，生成需要保留的文件链接
    for package_name, package_info in packages.items():
        for file_path, file_size in package_info["files"]:
            # 拼接完整下载链接
            if file_path == "SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py":
                link = "https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py"
            else:
                link = f"https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/{file_path}"

            # 如果该链接在现有的下载列表中，添加到需要保留的文件列表
            if link in existing_links:
                valid_files.append((link, file_size))

    # 更新 downloadlist.txt，只保留需要保留的文件链接
    with open(download_list_path, "w") as f:
        for link, size in valid_files:
            f.write(f"{link},{size}\n")

    print(f"{Fore.YELLOW}>>>downloadlist.txt 已更新<<<{Style.RESET_ALL}")

    return valid_files

def delete_package(package_name, packages):
    """
    删除指定包体的文件
    :param package_name: 包体名称
    :param packages: 包体字典
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # 确保包体名称在 packages 字典中
    if package_name not in packages:
        print(f"{Fore.RED}无效的包体名称！{Style.RESET_ALL}")
        return

    package = packages[package_name]
    print(f"开始删除包体：{package['name']}")

    # 先构建一个文件引用计数
    file_references = {}
    for pkg_name, pkg_info in packages.items():
        for file_info in pkg_info["files"]:
            file_path = file_info[0]
            if file_path not in file_references:
                file_references[file_path] = 0
            file_references[file_path] += 1  # 增加引用计数

    # 存储将要删除的有效文件
    files_to_delete = []
    for file_info in package["files"]:
        file_path = file_info[0]
        expected_dir = os.path.join(root, os.path.dirname(file_path))  # 路径转换
        expected_filename = os.path.basename(file_path)

        # 构造完整的文件路径
        full_file_path = os.path.join(expected_dir, expected_filename)

        # 只删除那些在当前包体中独有的文件，并检查文件是否存在
        if file_references[file_path] == 1:  # 该文件仅属于当前包体
            if os.path.exists(full_file_path):
                files_to_delete.append(full_file_path)
                print(f"文件: {full_file_path}")
            else:
                print(f"{Fore.RED}文件不存在: {full_file_path}{Style.RESET_ALL}")
    
    # 如果有文件准备删除，询问用户确认
    if files_to_delete:
        confirm = input(f"{Fore.GREEN}是否确认删除此包体及其文件？(y/n): {Style.RESET_ALL}")
        
        if confirm.lower() == 'y':
            # 执行删除操作
            for file in files_to_delete:
                try:
                    os.remove(file)
                    print(f"{Fore.GREEN}已删除文件: {file}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}删除文件失败: {file}，错误信息: {e}{Style.RESET_ALL}")
            
            # 删除包体条目
            del packages[package_name]
            print(f"{Fore.GREEN}包体 [{package_name}] 已从列表中删除。{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}删除操作已取消。{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}没有可删除的文件，文件已被删除，或被其他包体引用。{Style.RESET_ALL}")

packages = {
    "base_package": {
        "id": 1,
        "name": "[1]基础模型包",
        "note": "SDXL全功能|显存需求：★★ 速度：★★★☆",
        "files": [
            ("SimpleModels/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("SimpleModels/checkpoints/realisticVisionV60B1_v51VAE.safetensors", 2132625894),
            ("SimpleModels/clip_vision/clip_vision_vit_h.safetensors", 1972298538),
            ("SimpleModels/clip_vision/model_base_caption_capfilt_large.pth", 896081425),
            ("SimpleModels/clip_vision/wd-v1-4-moat-tagger-v2.onnx", 326197340),
            ("SimpleModels/clip_vision/clip-vit-large-patch14/merges.txt", 524619),
            ("SimpleModels/clip_vision/clip-vit-large-patch14/special_tokens_map.json", 389),
            ("SimpleModels/clip_vision/clip-vit-large-patch14/tokenizer_config.json", 905),
            ("SimpleModels/clip_vision/clip-vit-large-patch14/vocab.json", 961143),
            ("SimpleModels/configs/anything_v3.yaml", 1933),
            ("SimpleModels/configs/v1-inference.yaml", 1873),
            ("SimpleModels/configs/v1-inference_clip_skip_2.yaml", 1933),
            ("SimpleModels/configs/v1-inference_clip_skip_2_fp16.yaml", 1956),
            ("SimpleModels/configs/v1-inference_fp16.yaml", 1896),
            ("SimpleModels/configs/v1-inpainting-inference.yaml", 1992),
            ("SimpleModels/configs/v2-inference-v.yaml", 1815),
            ("SimpleModels/configs/v2-inference-v_fp32.yaml", 1816),
            ("SimpleModels/configs/v2-inference.yaml", 1789),
            ("SimpleModels/configs/v2-inference_fp32.yaml", 1790),
            ("SimpleModels/configs/v2-inpainting-inference.yaml", 4450),
            ("SimpleModels/controlnet/control-lora-canny-rank128.safetensors", 395733680),
            ("SimpleModels/controlnet/detection_Resnet50_Final.pth", 109497761),
            ("SimpleModels/controlnet/fooocus_ip_negative.safetensors", 65616),
            ("SimpleModels/controlnet/fooocus_xl_cpds_128.safetensors", 395706528),
            ("SimpleModels/controlnet/ip-adapter-plus-face_sdxl_vit-h.bin", 1013454761),
            ("SimpleModels/controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427),
            ("SimpleModels/controlnet/parsing_parsenet.pth", 85331193),
            ("SimpleModels/controlnet/xinsir_cn_openpose_sdxl_1.0.safetensors", 2502139104),
            ("SimpleModels/controlnet/lllyasviel/Annotators/body_pose_model.pth", 209267595),
            ("SimpleModels/controlnet/lllyasviel/Annotators/facenet.pth", 153718792),
            ("SimpleModels/controlnet/lllyasviel/Annotators/hand_pose_model.pth", 147341049),
            ("SimpleModels/inpaint/fooocus_inpaint_head.pth", 52602),
            ("SimpleModels/inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("SimpleModels/inpaint/inpaint_v26.fooocus.patch", 1323362033),
            ("SimpleModels/inpaint/isnet-anime.onnx", 176069933),
            ("SimpleModels/inpaint/isnet-general-use.onnx", 178648008),
            ("SimpleModels/inpaint/sam_vit_b_01ec64.pth", 375042383),
            ("SimpleModels/inpaint/silueta.onnx", 44173029),
            ("SimpleModels/inpaint/u2net.onnx", 175997641),
            ("SimpleModels/inpaint/u2netp.onnx", 4574861),
            ("SimpleModels/inpaint/u2net_cloth_seg.onnx", 176194565),
            ("SimpleModels/inpaint/u2net_human_seg.onnx", 175997641),
            ("SimpleModels/layer_model/layer_xl_fg2ble.safetensors", 701981624),
            ("SimpleModels/layer_model/layer_xl_transparent_conv.safetensors", 3619745776),
            ("SimpleModels/layer_model/vae_transparent_decoder.safetensors", 208266320),
            ("SimpleModels/llms/bert-base-uncased/config.json", 570),
            ("SimpleModels/llms/bert-base-uncased/model.safetensors", 440449768),
            ("SimpleModels/llms/bert-base-uncased/tokenizer.json", 466062),
            ("SimpleModels/llms/bert-base-uncased/tokenizer_config.json", 28),
            ("SimpleModels/llms/bert-base-uncased/vocab.txt", 231508),
            ("SimpleModels/llms/Helsinki-NLP/opus-mt-zh-en/config.json", 1394),
            ("SimpleModels/llms/Helsinki-NLP/opus-mt-zh-en/generation_config.json", 293),
            ("SimpleModels/llms/Helsinki-NLP/opus-mt-zh-en/metadata.json", 1477),
            ("SimpleModels/llms/Helsinki-NLP/opus-mt-zh-en/pytorch_model.bin", 312087009),
            ("SimpleModels/llms/Helsinki-NLP/opus-mt-zh-en/source.spm", 804677),
            ("SimpleModels/llms/Helsinki-NLP/opus-mt-zh-en/target.spm", 806530),
            ("SimpleModels/llms/Helsinki-NLP/opus-mt-zh-en/tokenizer_config.json", 44),
            ("SimpleModels/llms/Helsinki-NLP/opus-mt-zh-en/vocab.json", 1617902),
            ("SimpleModels/llms/superprompt-v1/config.json", 1512),
            ("SimpleModels/llms/superprompt-v1/generation_config.json", 142),
            ("SimpleModels/llms/superprompt-v1/model.safetensors", 307867048),
            ("SimpleModels/llms/superprompt-v1/README.md", 3661),
            ("SimpleModels/llms/superprompt-v1/spiece.model", 791656),
            ("SimpleModels/llms/superprompt-v1/tokenizer.json", 2424064),
            ("SimpleModels/llms/superprompt-v1/tokenizer_config.json", 2539),
            ("SimpleModels/loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors", 371842896),
            ("SimpleModels/loras/sdxl_hyper_sd_4step_lora.safetensors", 787359648),
            ("SimpleModels/loras/sdxl_lightning_4step_lora.safetensors", 393854592),
            ("SimpleModels/loras/sd_xl_offset_example-lora_1.0.safetensors", 49553604),
            ("SimpleModels/prompt_expansion/fooocus_expansion/config.json", 937),
            ("SimpleModels/prompt_expansion/fooocus_expansion/merges.txt", 456356),
            ("SimpleModels/prompt_expansion/fooocus_expansion/positive.txt", 5655),
            ("SimpleModels/prompt_expansion/fooocus_expansion/pytorch_model.bin", 351283802),
            ("SimpleModels/prompt_expansion/fooocus_expansion/special_tokens_map.json", 99),
            ("SimpleModels/prompt_expansion/fooocus_expansion/tokenizer.json", 2107625),
            ("SimpleModels/prompt_expansion/fooocus_expansion/tokenizer_config.json", 255),
            ("SimpleModels/prompt_expansion/fooocus_expansion/vocab.json", 798156),
            ("SimpleModels/rembg/RMBG-1.4.pth", 176718373),
            ("SimpleModels/unet/iclight_sd15_fc_unet_ldm.safetensors", 1719144856),
            ("SimpleModels/upscale_models/fooocus_upscaler_s409985e5.bin", 33636613),
            ("SimpleModels/vae_approx/vaeapp_sd15.pth", 213777),
            ("SimpleModels/vae_approx/xl-to-v1_interposer-v4.0.safetensors", 5667280),
            ("SimpleModels/vae_approx/xlvaeapp.pth", 213777),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/vae/ponyDiffusionV6XL_vae.safetensors", 334641162),
            ("SimpleModels/loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
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
            ("SimpleModels/embeddings/unaestheticXLhk1.safetensors", 33296),
            ("SimpleModels/embeddings/unaestheticXLv31.safetensors", 33296),
            ("SimpleModels/inpaint/inpaint_v25.fooocus.patch", 2580722369),
            ("SimpleModels/inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("SimpleModels/inpaint/sam_vit_l_0b3195.pth", 1249524607),
            ("SimpleModels/layer_model/layer_xl_bg2ble.safetensors", 701981624),
            ("SimpleModels/layer_model/layer_xl_transparent_attn.safetensors", 743352688),
            ("SimpleModels/llms/nllb-200-distilled-600M/pytorch_model.bin", 2460457927),
            ("SimpleModels/llms/nllb-200-distilled-600M/sentencepiece.bpe.model", 4852054),
            ("SimpleModels/llms/nllb-200-distilled-600M/tokenizer.json", 17331176),
            ("SimpleModels/loras/FilmVelvia3.safetensors", 151108832),
            ("SimpleModels/loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("SimpleModels/loras/SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 912593164),
            ("SimpleModels/safety_checker/stable-diffusion-safety-checker.bin", 1216067303),
            ("SimpleModels/unet/iclight_sd15_fbc_unet_ldm.safetensors", 1719167896),
            ("SimpleModels/upscale_models/4x-UltraSharp.pth", 66961958),
            ("SimpleModels/vae/ponyDiffusionV6XL_vae.safetensors", 334641162),
            ("SimpleModels/vae/sdxl_fp16.vae.safetensors", 167335342),
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
            ("SimpleModels/diffusers/Kolors/model_index.json", 427),
            ("SimpleModels/diffusers/Kolors/MODEL_LICENSE", 14920),
            ("SimpleModels/diffusers/Kolors/README.md", 4707),
            ("SimpleModels/diffusers/Kolors/scheduler/scheduler_config.json", 606),
            ("SimpleModels/diffusers/Kolors/text_encoder/config.json", 1323),
            ("SimpleModels/diffusers/Kolors/text_encoder/configuration_chatglm.py", 2332),
            ("SimpleModels/diffusers/Kolors/text_encoder/modeling_chatglm.py", 55722),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00001-of-00007.bin", 1827781090),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00002-of-00007.bin", 1968299480),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00003-of-00007.bin", 1927415036),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00004-of-00007.bin", 1815225998),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00005-of-00007.bin", 1968299544),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00006-of-00007.bin", 1927415036),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00007-of-00007.bin", 1052808542),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model.bin.index.json", 20437),
            ("SimpleModels/diffusers/Kolors/text_encoder/quantization.py", 14692),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenization_chatglm.py", 12223),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenizer.model", 1018370),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenizer_config.json", 249),
            ("SimpleModels/diffusers/Kolors/text_encoder/vocab.txt", 1018370),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenization_chatglm.py", 12223),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenizer.model", 1018370),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenizer_config.json", 249),
            ("SimpleModels/diffusers/Kolors/tokenizer/vocab.txt", 1018370),
            ("SimpleModels/diffusers/Kolors/unet/config.json", 1785),
            ("SimpleModels/diffusers/Kolors/unet/diffusion_pytorch_model.fp16.safetensors", 0),
            ("SimpleModels/diffusers/Kolors/vae/diffusion_pytorch_model.fp16.safetensors", 0),
            ("SimpleModels/diffusers/Kolors/vae/config.json", 611),
            ("SimpleModels/loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("SimpleModels/checkpoints/kolors_unet_fp16.safetensors", 5159140240),
            ("SimpleModels/vae/sdxl_fp16.vae.safetensors", 167335342),
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
            ("SimpleModels/checkpoints/animaPencilXL_v500.safetensors", 6938041144),
            ("SimpleModels/checkpoints/hunyuan_dit_1.2.safetensors", 8240228270),
            ("SimpleModels/checkpoints/playground-v2.5-1024px.safetensors", 6938040576),
            ("SimpleModels/checkpoints/ponyDiffusionV6XL.safetensors", 6938041050),
            ("SimpleModels/checkpoints/realisticStockPhoto_v20.safetensors", 6938054242),
            ("SimpleModels/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors", 10867168284),
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
            ("SimpleModels/checkpoints/flux1-dev.safetensors", 23802932552),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/t5xxl_fp16.safetensors", 9787841024),
            ("SimpleModels/vae/ae.safetensors", 335304388),
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
            ("SimpleModels/checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("SimpleModels/checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("SimpleModels/clip/t5xxl_fp16.safetensors", 9787841024),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("SimpleModels/controlnet/flux.1-dev_controlnet_union_pro.safetensors", 6603953920),
            ("SimpleModels/controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("SimpleModels/controlnet/parsing_bisenet.pth", 53289463),
            ("SimpleModels/controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("SimpleModels/insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("SimpleModels/insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("SimpleModels/insightface/models/antelopev2/genderage.onnx", 1322532),
            ("SimpleModels/insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("SimpleModels/insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("SimpleModels/loras/flux1-canny-dev-lora.safetensors", 1244443944),
            ("SimpleModels/loras/flux1-depth-dev-lora.safetensors", 1244440512),
            ("SimpleModels/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("SimpleModels/pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("SimpleModels/upscale_models/4x-UltraSharp.pth", 66961958),
            ("SimpleModels/upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_Flux_AIO_simpleai_1214.zip"
        ]
    },
        "SD15_aio_package": {
        "id": 7,
        "name": "[7]SD1.5_AIO扩展包",
        "note": "SD1.5全功能|显存需求：★ 速度：★★★★",
        "files": [
            ("SimpleModels/checkpoints/realisticVisionV60B1_v51VAE.safetensors", 2132625894),
            ("SimpleModels/loras/sd_xl_offset_example-lora_1.0.safetensors", 49553604),
            ("SimpleModels/clip/sd15_clip_model.fp16.safetensors", 246144864),
            ("SimpleModels/controlnet/control_v11f1e_sd15_tile_fp16.safetensors", 722601104),
            ("SimpleModels/controlnet/control_v11f1p_sd15_depth_fp16.safetensors", 722601100),
            ("SimpleModels/controlnet/control_v11p_sd15_canny_fp16.safetensors", 722601100),
            ("SimpleModels/controlnet/control_v11p_sd15_openpose_fp16.safetensors", 722601100),
            ("SimpleModels/controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("SimpleModels/inpaint/sd15_powerpaint_brushnet_clip_v2_1.bin", 492401329),
            ("SimpleModels/inpaint/sd15_powerpaint_brushnet_v2_1.safetensors", 3544366408),
            ("SimpleModels/insightface/models/buffalo_l/1k3d68.onnx", 143607619),
            ("SimpleModels/insightface/models/buffalo_l/2d106det.onnx", 5030888),
            ("SimpleModels/insightface/models/buffalo_l/det_10g.onnx", 16923827),
            ("SimpleModels/insightface/models/buffalo_l/genderage.onnx", 1322532),
            ("SimpleModels/insightface/models/buffalo_l/w600k_r50.onnx", 174383860),
            ("SimpleModels/ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("SimpleModels/ipadapter/ip-adapter-faceid-plusv2_sd15.bin", 156558509),
            ("SimpleModels/ipadapter/ip-adapter_sd15.safetensors", 44642768),
            ("SimpleModels/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors", 51059544),
            ("SimpleModels/upscale_models/4x-UltraSharp.pth", 66961958),
            ("SimpleModels/upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604)
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
            ("SimpleModels/checkpoints/kolors_unet_fp16.safetensors", 5159140240),
            ("SimpleModels/clip_vision/kolors_clip_ipa_plus_vit_large_patch14_336.bin", 1711974081),
            ("SimpleModels/controlnet/kolors_controlnet_canny.safetensors", 2526129624),
            ("SimpleModels/controlnet/kolors_controlnet_depth.safetensors", 2526129624),
            ("SimpleModels/controlnet/kolors_controlnet_pose.safetensors", 2526129624),
            ("SimpleModels/controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("SimpleModels/diffusers/Kolors/model_index.json", 427),
            ("SimpleModels/diffusers/Kolors/MODEL_LICENSE", 14920),
            ("SimpleModels/diffusers/Kolors/README.md", 4707),
            ("SimpleModels/diffusers/Kolors/scheduler/scheduler_config.json", 606),
            ("SimpleModels/diffusers/Kolors/text_encoder/config.json", 1323),
            ("SimpleModels/diffusers/Kolors/text_encoder/configuration_chatglm.py", 2332),
            ("SimpleModels/diffusers/Kolors/text_encoder/modeling_chatglm.py", 55722),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00001-of-00007.bin", 1827781090),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00002-of-00007.bin", 1968299480),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00003-of-00007.bin", 1927415036),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00004-of-00007.bin", 1815225998),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00005-of-00007.bin", 1968299544),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00006-of-00007.bin", 1927415036),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00007-of-00007.bin", 1052808542),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model.bin.index.json", 20437),
            ("SimpleModels/diffusers/Kolors/text_encoder/quantization.py", 14692),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenization_chatglm.py", 12223),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenizer.model", 1018370),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenizer_config.json", 249),
            ("SimpleModels/diffusers/Kolors/text_encoder/vocab.txt", 1018370),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenization_chatglm.py", 12223),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenizer.model", 1018370),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenizer_config.json", 249),
            ("SimpleModels/diffusers/Kolors/tokenizer/vocab.txt", 1018370),
            ("SimpleModels/diffusers/Kolors/unet/config.json", 1785),
            ("SimpleModels/diffusers/Kolors/vae/config.json", 611),
            ("SimpleModels/diffusers/Kolors/unet/diffusion_pytorch_model.fp16.safetensors", 0),
            ("SimpleModels/diffusers/Kolors/vae/diffusion_pytorch_model.fp16.safetensors", 0),
            ("SimpleModels/insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("SimpleModels/insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("SimpleModels/insightface/models/antelopev2/genderage.onnx", 1322532),
            ("SimpleModels/insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("SimpleModels/insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("SimpleModels/ipadapter/kolors_ipa_faceid_plus.bin", 2385842603),
            ("SimpleModels/ipadapter/kolors_ip_adapter_plus_general.bin", 1013163359),
            ("SimpleModels/loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("SimpleModels/loras/sd_xl_offset_example-lora_1.0.safetensors", 49553604),
            ("SimpleModels/unet/kolors_inpainting.safetensors", 5159169040),
            ("SimpleModels/upscale_models/4x-UltraSharp.pth", 66961958),
            ("SimpleModels/upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("SimpleModels/vae/sdxl_fp16.vae.safetensors", 167335342)
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_Kolors_AIO_simpleai_1214.zip"
        ]
    },
        "SD3x_medium_package": {
        "id": 9,
        "name": "[9]SD3.5_medium扩展包",
        "note": "SD3.5中号文生图|显存需求：★★ 速度：★★★",
        "files": [
            ("SimpleModels/checkpoints/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors", 11638004202),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/vae/sd3x_fp16.vae.safetensors", 167666654),
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
            ("SimpleModels/checkpoints/sd3.5_large.safetensors", 16460379262),
            ("SimpleModels/clip/clip_g.safetensors", 1389382176),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/t5xxl_fp16.safetensors", 9787841024),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/vae/sd3x_fp16.vae.safetensors", 167666654),
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
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/.gitattributes", 1657),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/.mdl", 49),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/.msc", 1655),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/.mv", 36),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/added_tokens.json", 629),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/config.json", 1951),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/configuration.json", 27),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/configuration_minicpm.py", 3280),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/generation_config.json", 121),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/image_processing_minicpmv.py", 16579),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/merges.txt", 1671853),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/modeling_minicpmv.py", 15738),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/modeling_navit_siglip.py", 41835),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/preprocessor_config.json", 714),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/processing_minicpmv.py", 9962),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/pytorch_model-00001-of-00002.bin", 4454731094),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/pytorch_model-00002-of-00002.bin", 1503635286),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/pytorch_model.bin.index.json", 233389),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/README.md", 2124),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/resampler.py", 34699),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/special_tokens_map.json", 1041),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/test.py", 1162),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/tokenization_minicpmv_fast.py", 1659),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/tokenizer.json", 7032006),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/tokenizer_config.json", 5663),
            ("SimpleModels/llms/MiniCPMv2_6-prompt-generator/vocab.json", 2776833),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/blob/main/models_minicpm_v2.6_prompt_simpleai_1224.zip"
        ]
    },
        "happy_package": {
        "id": 12,
        "name": "[12]贺年卡",
        "note": "贺卡预设|显存需求：★★★ 速度：★★",
        "files": [
            ("SimpleModels/loras/flux_graffiti_v1.safetensors", 612893792),
            ("SimpleModels/loras/kolors_crayonsketch_e10.safetensors", 170566628),
            ("SimpleModels/checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("SimpleModels/clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/checkpoints/kolors_unet_fp16.safetensors", 5159140240),
            ("SimpleModels/clip_vision/kolors_clip_ipa_plus_vit_large_patch14_336.bin", 1711974081),
            ("SimpleModels/controlnet/kolors_controlnet_canny.safetensors", 2526129624),
            ("SimpleModels/diffusers/Kolors/model_index.json", 427),
            ("SimpleModels/diffusers/Kolors/MODEL_LICENSE", 14920),
            ("SimpleModels/diffusers/Kolors/README.md", 4707),
            ("SimpleModels/diffusers/Kolors/scheduler/scheduler_config.json", 606),
            ("SimpleModels/diffusers/Kolors/text_encoder/config.json", 1323),
            ("SimpleModels/diffusers/Kolors/text_encoder/configuration_chatglm.py", 2332),
            ("SimpleModels/diffusers/Kolors/text_encoder/modeling_chatglm.py", 55722),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00001-of-00007.bin", 1827781090),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00002-of-00007.bin", 1968299480),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00003-of-00007.bin", 1927415036),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00004-of-00007.bin", 1815225998),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00005-of-00007.bin", 1968299544),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00006-of-00007.bin", 1927415036),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model-00007-of-00007.bin", 1052808542),
            ("SimpleModels/diffusers/Kolors/text_encoder/pytorch_model.bin.index.json", 20437),
            ("SimpleModels/diffusers/Kolors/text_encoder/quantization.py", 14692),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenization_chatglm.py", 12223),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenizer.model", 1018370),
            ("SimpleModels/diffusers/Kolors/text_encoder/tokenizer_config.json", 249),
            ("SimpleModels/diffusers/Kolors/text_encoder/vocab.txt", 1018370),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenization_chatglm.py", 12223),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenizer.model", 1018370),
            ("SimpleModels/diffusers/Kolors/tokenizer/tokenizer_config.json", 249),
            ("SimpleModels/diffusers/Kolors/tokenizer/vocab.txt", 1018370),
            ("SimpleModels/diffusers/Kolors/unet/config.json", 1785),
            ("SimpleModels/diffusers/Kolors/vae/config.json", 611),
            ("SimpleModels/ipadapter/kolors_ipa_faceid_plus.bin", 2385842603),
            ("SimpleModels/ipadapter/kolors_ip_adapter_plus_general.bin", 1013163359),
            ("SimpleModels/vae/sdxl_fp16.vae.safetensors", 167335342),
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
            ("SimpleModels/inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py", 1006),
            ("SimpleModels/checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("SimpleModels/clip/clip_l.safetensors", 246144152), 
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("SimpleModels/style_models/flux1-redux-dev.safetensors", 129063232),
            ("SimpleModels/rembg/General.safetensors", 884878856)
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
            ("SimpleModels/checkpoints/SDXL_Yamers_Cartoon_Arcadia.safetensors", 6938040714),
            ("SimpleModels/upscale_models/RealESRGAN_x4plus_anime_6B.pth", 17938799),
            ("SimpleModels/rembg/Portrait.safetensors", 884878856),
            ("SimpleModels/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin", 1487555181),
            ("SimpleModels/ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("SimpleModels/insightface/models/buffalo_l/1k3d68.onnx", 143607619),
            ("SimpleModels/insightface/models/buffalo_l/2d106det.onnx", 5030888),
            ("SimpleModels/insightface/models/buffalo_l/det_10g.onnx", 16923827),
            ("SimpleModels/insightface/models/buffalo_l/genderage.onnx", 1322532),
            ("SimpleModels/insightface/models/buffalo_l/w600k_r50.onnx", 174383860),
            ("SimpleModels/loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors", 371842896),
            ("SimpleModels/loras/StickersRedmond.safetensors", 170540036)
        ],
        "download_links": [
        "【选配】模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "x1-okremovebg_package": {
        "id": 15,
        "name": "[15]一键抠图",
        "note": "抠图去背景神器|显存需求：★ 速度：★★★★★",
        "files": [
            ("SimpleModels/rembg/ckpt_base.pth", 367520613),
            ("SimpleModels/rembg/RMBG-1.4.pth", 176718373)
        ],
        "download_links": [
        "【选配】模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "x2-okimagerepair_package": {
        "id": 16,
        "name": "[16]一键修复",
        "note": "上色、修复模糊、旧照片|显存需求：★★★ 速度：★☆",
        "files": [
            ("SimpleModels/checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("SimpleModels/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("SimpleModels/checkpoints/LEOSAM_HelloWorldXL_70.safetensors", 6938040682),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("SimpleModels/controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("SimpleModels/controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("SimpleModels/upscale_models/4xNomos8kSCHAT-L.pth", 331564661)
        ],
        "download_links": [
        "【选配】模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "x3-swapface_package": {
        "id": 17,
        "name": "[17]一键换脸",
        "note": "高精度换脸|显存需求：★★★ 速度：★☆",
        "files": [
            ("SimpleModels/checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("SimpleModels/pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/loras/flux1-turbo.safetensors", 694082424),
            ("SimpleModels/inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py", 1006),
            ("SimpleModels/inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("SimpleModels/style_models/flux1-redux-dev.safetensors", 129063232),
            ("SimpleModels/insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("SimpleModels/insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("SimpleModels/insightface/models/antelopev2/genderage.onnx", 1322532),
            ("SimpleModels/insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("SimpleModels/insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("SimpleModels/clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210)
        ],
        "download_links": [
        "【选配】模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "Flux_aio_plus_package": {
        "id": 18,
        "name": "[18]Flux_AIO_plus扩展包",
        "note": "Flux全功能[fp8模型]|显存需求：★★★★ 速度：★★☆",
        "files": [
            ("SimpleModels/checkpoints/flux1-dev-fp8.safetensors", 11901525888),
            ("SimpleModels/checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("SimpleModels/clip/t5xxl_fp16.safetensors", 9787841024),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("SimpleModels/controlnet/flux.1-dev_controlnet_union_pro.safetensors", 6603953920),
            ("SimpleModels/controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("SimpleModels/controlnet/parsing_bisenet.pth", 53289463),
            ("SimpleModels/controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("SimpleModels/insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("SimpleModels/insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("SimpleModels/insightface/models/antelopev2/genderage.onnx", 1322532),
            ("SimpleModels/insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("SimpleModels/insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("SimpleModels/loras/flux1-canny-dev-lora.safetensors", 1244443944),
            ("SimpleModels/loras/flux1-depth-dev-lora.safetensors", 1244440512),
            ("SimpleModels/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("SimpleModels/pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("SimpleModels/upscale_models/4x-UltraSharp.pth", 66961958),
            ("SimpleModels/upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": [
        "【选配】基于FluxAIO扩展包扩展",
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/flux-dev-fp8.safetensors"
        ]
    },
        "clothing_plus_package": {
        "id": 19,
        "name": "[19]换装plus包",
        "note": "万物迁移[fp8模型]|显存需求：★★★☆ 速度：★★★",
        "files": [
            ("SimpleModels/inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py", 1006),
            ("SimpleModels/checkpoints/flux1-fill-dev_fp8.safetensors", 11902532704),
            ("SimpleModels/checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("SimpleModels/style_models/flux1-redux-dev.safetensors", 129063232),
            ("SimpleModels/rembg/General.safetensors", 884878856)
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
            ("SimpleModels/checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("SimpleModels/checkpoints/flux1-fill-dev-hyp8-Q4_K_S.gguf", 6809920800),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/loras/fill_remove.safetensors",104667792),
            ("SimpleModels/style_models/flux1-redux-dev.safetensors", 129063232)
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
            ("SimpleModels/checkpoints/NoobAI-XL-v1.1.safetensors", 7105349958)
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/NoobAI-XL-v1.1.safetensors"
        ]
    },
        "Illustrious_aio_package": {
        "id": 22,
        "name": "[22]光辉AIO扩展包",
        "note": "NoobAI/光辉模型全功能图生图|显存需求：★★★ 速度：★★★",
        "files": [
            ("SimpleModels/checkpoints/NoobAI-XL-v1.1.safetensors", 7105349958),
            ("SimpleModels/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("SimpleModels/loras/sd_xl_offset_example-lora_1.0.safetensors", 49553604),
            ("SimpleModels/controlnet/noob_sdxl_controlnet_canny.fp16.safetensors", 2502139104),
            ("SimpleModels/controlnet/noob_sdxl_controlnet_depth.fp16.safetensors", 2502139136),
            ("SimpleModels/controlnet/noob_sdxl_controlnet_pose.fp16.safetensors", 2502140008),
            ("SimpleModels/ipadapter/noob_ip_adapter.bin", 1396798350),
            ("SimpleModels/inpaint/inpaint_v25.fooocus.patch", 2580722369),
            ("SimpleModels/inpaint/fooocus_inpaint_head.pth", 52602),
            ("SimpleModels/loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("SimpleModels/upscale_models/RealESRGAN_x4plus_anime_6B.pth", 17938799),
            ("SimpleModels/ipadapter/clip-vit-h-14-laion2B-s32B-b79K.safetensors", 3944517836),
            ("SimpleModels/controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099)
        ],
        "download_links": [
        "【选配】模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。",
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/NoobAI-XL-v1.1.safetensors"
        ]
    }
}
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
    main()  # 执行初始化
    print()
    while True:
        print(f">>>按【{Fore.YELLOW}Enter回车{Style.RESET_ALL}】启动全部文件下载。支持断点续传，顺序从小文件开始")
        print(f">>>输入【{Fore.YELLOW}包体编号{Style.RESET_ALL}】+【{Fore.YELLOW}回车{Style.RESET_ALL}】启动预置包补全，支持断点续传，顺序从小文件开始")
        print(f">>>输入【{Fore.YELLOW}0{Style.RESET_ALL}】+【{Fore.YELLOW}回车{Style.RESET_ALL}】清理下载缓存与损坏文件")
        print(f">>>输入【{Fore.YELLOW}del+包体编号{Style.RESET_ALL}】删除已有包体文件（避开关联文件）")
        print(f">>>输入【{Fore.YELLOW}r{Style.RESET_ALL}】+【{Fore.YELLOW}回车{Style.RESET_ALL}】重新检测")
        
        user_input = input("请选择操作：")

        if user_input == "":
            # 启动下载所有文件
            print("启动自动下载模块,支持断点续传，关闭窗口则中断。")
            auto_download_missing_files_with_retry(max_threads=5)  # 启动下载所有文件

        elif user_input.isdigit():
            # 如果用户输入了包体 ID 数字，执行下载指定包体文件
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
                # 如果用户输入0，则清除所有下载缓存
                delete_partial_files()
            else:
                print(f"{Fore.RED}包体编号{package_id} 无效，请输入正确的包体ID。{Style.RESET_ALL}")

        elif user_input.lower().startswith("del"):
            root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            # 删除包体文件
            try:
                # 去除 'del' 前缀并去除多余的空格
                package_id_str = user_input[3:].strip()
                
                # 如果输入为空或不合法，提醒用户
                if not package_id_str.isdigit():
                    print(f"{Fore.RED}输入格式错误，请输入类似 del1 来删除对应包体。{Style.RESET_ALL}")
                else:
                    package_id = int(package_id_str)  # 转换为整数
                    selected_package = None
                    
                    # 查找对应包体
                    for package_name, package_info in packages.items():
                        if package_info["id"] == package_id:
                            selected_package = package_info
                            break
                    
                    # 判断包体是否找到并打印信息
                    if selected_package:
                        # 打印包体信息
                        print(f"{Fore.YELLOW}即将删除以下包体文件：{Style.RESET_ALL}")
                        print(f"包体编号: {package_id}")
                        print(f"包体名称: {selected_package['name']}")
                        
                        # 先构建一个文件引用计数
                        file_references = {}
                        for pkg_name, pkg_info in packages.items():
                            for file_info in pkg_info["files"]:
                                file_path = file_info[0]
                                if file_path not in file_references:
                                    file_references[file_path] = 0
                                file_references[file_path] += 1  # 增加引用计数

                        # 存储将要删除的有效文件
                        files_to_delete = []
                        total_size_to_free = 0  # 用来存储即将删除的文件的总大小
                        
                        for file_info in selected_package["files"]:
                            file_path = file_info[0]
                            # 只删除那些在当前包体中独有的文件，并检查文件是否存在
                            if file_references[file_path] == 1:  # 该文件仅属于当前包体
                                # 确保路径转换一致
                                expected_dir = os.path.join(root, os.path.dirname(file_path))
                                expected_filename = os.path.basename(file_path)
                                full_file_path = os.path.join(expected_dir, expected_filename)

                                # 规范化路径
                                full_file_path = normalize_path(full_file_path)

                                # 检查文件是否存在
                                if os.path.exists(full_file_path):
                                    files_to_delete.append(full_file_path)
                                    # 获取文件大小并累加
                                    total_size_to_free += os.path.getsize(full_file_path)
                                    print(f"文件: {full_file_path} 大小: {os.path.getsize(full_file_path) / (1024 * 1024):.2f} MB")
                                
                        # 打印将要释放的空间
                        if files_to_delete:
                            print(f"{Fore.GREEN}总共可以释放空间: {total_size_to_free / (1024 * 1024 * 1024):.2f} GB{Style.RESET_ALL}")
                            
                            confirm = input(f"{Fore.GREEN}是否确认删除此包体内文件？【y/n】+【回车】: {Style.RESET_ALL}")
                            
                            if confirm.lower() == 'y':
                                # 先删除文件
                                for file_path in files_to_delete:
                                    try:
                                        os.remove(file_path)
                                        print(f"{Fore.GREEN}已删除文件: {file_path}{Style.RESET_ALL}")
                                    except Exception as e:
                                        print(f"{Fore.RED}删除文件失败: {file_path}，错误信息: {e}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.RED}删除操作已取消。{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}没有可删除的文件，因为它们被多个包体引用或文件不存在。{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}无效的包体编号！无法找到对应的包体。{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}输入格式错误，请输入类似 del1 来删除对应包体。{Style.RESET_ALL}")

        elif user_input.lower() == "r":
            # 重新执行文件检测
            print("重新检测文件...")
            validate_files(packages)

        else:
            print(f"{Fore.RED}无效的输入，请输入回车或有效的包体编号（不需要括号）。{Style.RESET_ALL}")