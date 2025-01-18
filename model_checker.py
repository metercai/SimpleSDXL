import os
import sys
import time
from colorama import init, Fore, Style

# 初始化 colorama
init(autoreset=True)

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

import os

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
    time.sleep(0.2)
    print()
    print(f"{Fore.GREEN}★{Style.RESET_ALL}攻略地址飞书文档:{Fore.YELLOW}https://acnmokx5gwds.feishu.cn/wiki/QK3LwOp2oiRRaTkFRhYcO4LonGe{Style.RESET_ALL}文章无权限即为未编辑完毕。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.2)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}稳速生图指南:Nvidia显卡驱动选择最新版驱动,驱动类型最好为Studio。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.2)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}在遇到生图速度断崖式下降或者爆显存OutOfMemory时,提高{Fore.GREEN}预留显存功能{Style.RESET_ALL}的数值至（1~2）{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.2)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}打开默认浏览器设置，关闭GPU加速、或图形加速的选项。{Fore.GREEN}★{Style.RESET_ALL}大内存(64+)与固态硬盘存放模型有助于减少模型加载时间。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.2)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}疑难杂症进QQ群求助：938075852{Fore.GREEN}★{Style.RESET_ALL}脚本：✿   冰華 {Fore.GREEN}★{Style.RESET_ALL}")
    print()
    time.sleep(0.2)

def validate_files(packages):
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # 使用 set 来确保路径唯一
    download_paths = set()

    for package_key, package_info in packages.items():
        package_name = package_info["name"]
        files_and_sizes = package_info["files"]
        download_links = package_info["download_links"]
        print(f"－－－－－－－", end='')  # 不换行
        time.sleep(0.1)
        print(f"校验{package_name}文件－－－－－－－")  # 打字动态效果
        time.sleep(0.1)
        missing_files = []
        size_mismatch_files = []
        case_mismatch_files = []

        for expected_path, expected_size in files_and_sizes:
            expected_dir = os.path.join(root, os.path.dirname(expected_path))
            expected_filename = os.path.basename(expected_path)

            if not os.path.exists(expected_dir):
                missing_files.append(expected_path)
                continue

            directory_listing = os.listdir(expected_dir)
            actual_filename = next((f for f in directory_listing if f.lower() == expected_filename.lower()), None)

            if actual_filename is None:
                missing_files.append(expected_path)
            elif actual_filename != expected_filename:
                case_mismatch_files.append((os.path.join(expected_dir, actual_filename), expected_filename))
            else:
                actual_size = os.path.getsize(os.path.join(expected_dir, actual_filename))
                if actual_size != expected_size:
                    size_mismatch_files.append((os.path.join(expected_dir, actual_filename), actual_size, expected_size))

        # 输出结果并拼接有问题的文件的下载链接
        if missing_files:
            print(f"{Fore.RED}×{package_name}有文件缺失，请检查以下文件:{Style.RESET_ALL}")
            for file in missing_files:
                print(normalize_path(file))  # 调用 normalize_path 规范化路径
                time.sleep(0.01)
                # 拼接下载路径并添加到 download_paths 列表中
                if file == "SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py":
                    download_paths.add("https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py")
                else:
                    download_paths.add(f"https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/{file}")
            print("请使用以下链接下载所需的文件：")
            for link in download_links:
                print(Fore.YELLOW + link + Fore.RESET)  # 直接打印字符串，避免输出单引号
            print("下载后，请按照安装视频使用脚本安装模型")

        if size_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。{Style.RESET_ALL}")
            for file, actual_size, expected_size in size_mismatch_files:
                print(f"{normalize_path(file)} 当前大小={actual_size}, 预期大小={expected_size}")  # 调用 normalize_path
                time.sleep(0.1)
                # 拼接下载路径并添加到 download_paths 列表中
                if file == "SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py":
                    download_paths.add("https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py")
                else:
                    download_paths.add(f"https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/{file}")
            print(f"请前往模型总仓{Fore.YELLOW}https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels{Style.RESET_ALL}收集替换")

        if case_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件名大小写不匹配，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_filename in case_mismatch_files:
                print(f"文件: {normalize_path(file)}")  # 调用 normalize_path
                time.sleep(0.1)
                print(f"正确文件名: {expected_filename}")
                # 拼接下载路径并添加到 download_paths 列表中
                if file == "SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py":
                    download_paths.add("https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py")
                else:
                    download_paths.add(f"https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/{file}")

        if not missing_files and not size_mismatch_files and not case_mismatch_files:
            print(f"{Fore.GREEN}√{package_name}文件全部验证通过{Style.RESET_ALL}")
        time.sleep(0.1)
        print()
    # 保存有问题的文件的下载路径到一个txt文件中
    if download_paths:
        with open("缺失模型下载链接.txt", "w") as f:
            for path in download_paths:
                f.write(path + "\n")
        print(f"{Fore.YELLOW}>>>所有有问题的文件下载路径已保存到 '缺失模型下载链接.txt'。<<<{Style.RESET_ALL}")

packages = {
    "base_package": {
        "name": "基础模型包",
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
        "name": "增强模型包",
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
        "name": "可图扩展包",
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
        "name": "额外模型包",
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
        "name": "Flux全量包",
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
        "name": "Flux_AIO扩展包",
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
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_Flux_AIO_simpleai_1214.zip"
        ]
    },
        "SD15_aio_package": {
        "name": "SD1.5_AIO扩展包",
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
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_sd15_aio_simpleai_1214.zip"
        ]
    },
        "Kolors_aio_package": {
        "name": "Kolors_AIO扩展包",
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
            ("SimpleModels/insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("SimpleModels/insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("SimpleModels/insightface/models/antelopev2/genderage.onnx", 1322532),
            ("SimpleModels/insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("SimpleModels/insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("SimpleModels/ipadapter/kolors_ipa_faceid_plus.bin", 2385842603),
            ("SimpleModels/ipadapter/kolors_ip_adapter_plus_general.bin", 1013163359),
            ("SimpleModels/loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("SimpleModels/unet/kolors_inpainting.safetensors", 5159169040),
            ("SimpleModels/upscale_models/4x-UltraSharp.pth", 66961958),
            ("SimpleModels/vae/sdxl_fp16.vae.safetensors", 167335342),
        ],
        "download_links": [
        "【选配】https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_Kolors_AIO_simpleai_1214.zip"
        ]
    },
        "SD3x_medium_package": {
        "name": "SD3.5_medium扩展包",
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
        "name": "SD3.5_Large 扩展包",
        "files": [
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
        "name": "MiniCPMv26反推扩展包",
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
        "name": "贺年卡",
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
        "name": "换装包",
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
        "name": "3D大头贴",
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
        "name": "一键抠图",
        "files": [
            ("SimpleModels/rembg/ckpt_base.pth", 367520613),
            ("SimpleModels/rembg/RMBG-1.4.pth", 176718373)
        ],
        "download_links": [
        "【选配】模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "x2-okimagerepair_package": {
        "name": "一键修复",
        "files": [
            ("SimpleModels/checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("SimpleModels/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("SimpleModels/checkpoints/LEOSAM_HelloWorldXL_70.safetensors", 6938040682),
            ("SimpleModels/clip/clip_l.safetensors", 246144152),
            ("SimpleModels/clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("SimpleModels/vae/ae.safetensors", 335304388),
            ("SimpleModels/loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("SimpleModels/controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("SimpleModels/upscale_models/4xNomos8kSCHAT-L.pth", 331564661)
        ],
        "download_links": [
        "【选配】模型仓库https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels。部分文件、Lora点击生成会自动下载。"
        ]
    },
        "x3-swapface_package": {
        "name": "一键换脸",
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
    print_colored("★★★★★★★★★★★★★★★★★★检测已结束执行自动解压脚本★★★★★★★★★★★★★★★★★★", Fore.CYAN)
if __name__ == "__main__":
    main()
