## SimpleSDXL - Fooocus中文孪生版
<div align=center><img src="https://github.com/metercai/SimpleSDXL/assets/5652458/fb4b0f05-44b3-451d-95e0-e3e6de7cd929"></div>
在Fooocus基础上增强功能，可无缝升级，两版本同时可用。<br> Enhanced features base on Fooocus, seamless upgrading and dual versions available simultaneously.

- **中英文混编提示词**: 离线在线自主选择，支持翻译后再编辑，更适于Tag表述。<br>
  **Chinese English mixed prompts**:  Offline and online autonomous selection, support editing after translation, more suitable for Tags.
- **已生成图片集浏览**: 原生版本仅限浏览当次生成的图片集。 <br>
  **Finished image sets browsing**:  Fooocus only can browse the current generated image set.
- **查看生成参数和提取重生**: 可即时查看历史图片的生成参数，也可提取参数回填界面，二次生成。 <br>
  **View parameters and extract regeneration**:  Can view the generation parameters of historical images, and can also extract parameters to backfill for secondary generation.
- **预置包在线切换和生成**: 原生版切换需要修改启动参数重启，没有提供生成工具。 <br>
  **Switch preset and theme online**:  Fooocus requires modification of startup parameters to restart, no generation tool provided.

## 🔔 更新 / Update
- [2023.12.30] <b> 提示词翻译器升级：支持中英文混合编辑，支持翻译后再编辑，可自主选择本地模型翻译还是第三方接口翻译，提升提示词翻译器的总体可用性和可控性，翻译质量也能得到保障，应该是目前文生图提示词翻译器的最佳实践了。同步主线v2.1.856。</b>
- [2023.12.28] 重要更新：底层代码结构重构，相册图集、顶部导航、右侧工具箱解耦。预置包生成流程和加载流程优化，启用MUID模型文件标识，文件名不同自动转换，本地无模型文件自动下载，自动识别embeddings模型，模型MUID标识同步改手动触发。顶部导航视觉优化，新增日志消息提示层，提取和展示Fooocus和SimpleSDXL的最新更新日志，增强互动。本次更新为由预置包和嵌参图片组成的开放系统打下基础，以实现多场景适配的定制能力。同步主线v2.1.855。
- [2023.12.20] 下载优化：启动流程针对国内网络进行优化，包括启动器瘦身，以及PyPI包索引库、项目代码库和模型库等全都选用国内源进行更新和下载。一键启动器(75M): [SimpleSDXL_win64_in](https://gitee.com/metercai/SimpleSDXL/releases/download/win64/SimpleSDXL_win64_in.exe)，提供从零开始的安装过程引导。SimpleSDXL升级包(40M): [SimpleSDXL_win64_upgrade_in](https://gitee.com/metercai/SimpleSDXL/releases/download/win64/SimpleSDXL_win64_upgrade_in.exe)，提供在Fooocus已有环境上的升级，保持两个版本同时可用。同步主线版本v2.1.853，并适配了新的log.html格式。
- [2023.12.12] 新增特性：将当前环境的配置和参数生成自己的预置包，自动上到顶部导航。对接统一模型标识信息库，为预置包和带参图片分享的可用性提供基础支撑。同步主线版本v2.1.830。
- [2023.12.01] 新增特性：提取历史图片的生成参数，自动回填输入界面，可编辑后二次做生成。
- [2023.11.30] 重要更新：新增图片参数工具箱，实现图片浏览过程中对生成参数的及时查看，数据源来自于图片目录的 `log.html` ，可以与主线版本无缝衔接。
- [2023.11.26] 重要更新：新增顶部工具条，实现了场景预置包的切换和背景切换。优化界面布局，将“出图数量”选项提高到首屏，默认中文界面、夜黑背景和高级选项打开，常用操作尽可能快捷。同步主线最新版本v2.1.824。
- [2023.11.20] 完善历史图片索引功能，新增一天内的图片分页，避免组内图片数量过大；修复其他已知bug。
- [2023.11.18] 将大部分汉化翻译移到 `language/cn.json` ；界面上新增历史图片浏览功能，可以按照日期分组快速浏览历史生成的图片，每次进入会默认加载最新一组的最新一张。同步最新版本v2.1.821。
- [2023.11.14] 风格名称汉化，里面有太多西方专属词汇，中文用户比较少用，待改进。同步版本到v2.1.805，看到测试中的FaceSwap了 。
- [2023.11.12] 同步Fooocus最新版到v2.1.789，修订界面汉化文字，新增离线多语言翻译器：**nllb-200**，Prompt支持中英文混编，自动识别中文并统一翻译到英文。此功能需提前下载翻译模型到 `models/translator/` 目录，源地址：https://huggingface.co/facebook/nllb-200-distilled-600M 。注：需要整目录下载，保留目录名。
- [2023.10.16] 初始版本，界面文字汉化，新增 `--webroot` 参数，设定云端URL访问的根路径。如云端访问地址为：http://hostname/sdxl/ ，启动参数中需追加 `--webroot /sdxl` 。

## 什么是SimpleSDXL？/ What's SimpleSDXL?
- **化繁为简** AI的本质应该是化繁为简，让操作更简洁，让想法更易达成。SDXL的出图质量很出色，Fooocus的易用性非常棒，站在巨人的肩膀上有了SimpleSDXL，更简洁更易用。
- **中文适配** 中文环境与英语环境有很多差异。不仅仅在语言文字上，包括思维习惯和网络环境都有很多不同。让中文用户使用更简单，用的更爽，也是SimpleSDXL的初衷之一。
- **场景定制** 文生图和图生图有非常多的使用场景，需要出色的裁剪定制能力，进一步简化流程与操作，以接入更多使用场景，发挥SDXL的强大能力。

## 增强特性 / Enhanced Features
已实现及计划实现的功能特性：
- [x] **中文界面** SimpleSDXL的汉化翻译更贴合中文用户思维习惯。针对个别场景做了定制翻译，比如"Advanced"，在不同位置出现会有不同的文字翻译。 
- [x] **中英混编提示词** SDXL模型是以英文标签词（Tag）为主的提示词系统，很多词汇比较生僻，不利于中文用户使用。SimpleSDXL使用Meta(Facebook)最新SOTA的多语种翻译模型 nllb-200 ，实现本地化的提示词中英文混编。方便中文用户利用已有英文提示词进行改编创作。社区中的在线翻译器是调用第三方Api接口，会有访问限制，也无法适应中英混编的Tag表述模式。
- [x] **历史图片索引** Fooocus无法浏览前一次生成的图片，难以进行多次出图间的比对。SimpleSDXL新增了历史图片浏览功能，可以按照日期分组进行快速浏览比对。
- [x] **生成参数管理** 对出图后的提示词等参数进行管理，可以查看和自动回填，方便对历史图片生成参数的借鉴使用。
- [x] **场景预置包切换** 在主界面上可以对场景预置包进行切换，自动下载预置包相关资源，确保可用性，体验不同场景预置包的出图效果。
- [ ] **图片预置包嵌入** 精选优质的已生成图片，嵌入生成参数和预置包信息，方便学习和交流。
- [x] **访问根路径可设置** Fooocus的主场景在本机部署。当在云端部署配置前置转发后，会引起URL路径系统混乱。SimpleSDXL新增 `--webroot` 参数，可以设定访问URL的根路径，方便云端部署。
- [ ] **前后端分离，算力云化** 实现操控端本机部署，模型端云化部署。让无GPU卡设备用户也可使用上SDXL。
- [x] **模型打包，内外双源** 中文用户天然要面对复杂网络环境的处理。SimpleSDXL将使用到的模型文件打包，提供内外两个源供下载。
- [x] **主线功能及时同步** SimpleSDXL新增代码会保持良好的结构，与Fooocus主线版本具备兼容性，可以及时同步主线新增能力和Bug修复。

## 安装使用 / Install & Usage
### Windows :
分两种情况:<br>
一，在现有Fooocus环境下安装，两个版本可同时使用:
    点击下载**一键升级包**(40M,可执行压缩包): [SimpleSDXL_win64_upgrade_in](https://gitee.com/metercai/SimpleSDXL/releases/download/win64/SimpleSDXL_win64_upgrade_in.exe)。下载后点击运行将包内文件解压到现有Fooocus目录的父目录，即保证SimpleSDXL目录与Fooocus目录同级。然后启动命令行窗口`cmd`，进入到这个同级目录，用命令行运行`run_SimpleSDXL.bat`。注意必须用cmd的命令行运行。在第一次启动后，需要足够权限创建软连接，确保与Fooocus共享同一模型库和图片目录，从而让两个版本可同时使用。

二，全新安装SimpleSDXL:<br>
1, 点击下载**一键启动包**(75M,可执行压缩包): [SimpleSDXL_win64_in](https://gitee.com/metercai/SimpleSDXL/releases/download/win64/SimpleSDXL_win64_in.exe)。<br>
2, 解压缩到工作目录后，点击运行：`run.bat` 。第一次运行会下载项目代码，然后自动安装所需的各种PyPI模块.程序启动后，还会下载基础大模型文件。虽然下载源已全部更新为国内源，但模型尺寸比较大，总体时间较长，需耐心等待。<br>
3, 启动成功后，会自动打开浏览器，进入主界面。

### Linux :
一，在现有Fooocus环境下安装，两个版本可同时使用:<br>
1, 进入与现有Fooocus目录的父目录，即保证新建的SimpleSDXL目录与Fooocus目录同级。

    git clone https://github.com/metercai/SimpleSDXL.git
    #国内用户可换用gitee源: https://gitee.com/metercai/SimpleSDXL.git
    cd SimpleSDXL
    rm -rf models
    ln -s ../Fooocus/models models
    ln -s ../Fooocus/outputs outputs
    ln -s ../Fooocus/input input
    python entry_with_update.py 
    # 云端部署可以配置： ip, port, webroot 等参数
    # python entry_with_update.py --listen 0.0.0.0 --port 8889 --webroot /sdxl --preset realistic
    # Enter English UI : --language en
    # python entry_with_update.py --language en

二，全新安装SimpleSDXL:<br>
1, 安装 Anaconda 

    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
    bash Miniconda3-latest-Linux-x86_64.sh
2, 安装应用环境

    git clone https://github.com/metercai/SimpleSDXL.git
    #国内用户可换用gitee源: https://gitee.com/metercai/SimpleSDXL.git
    cd SimpleSDXL
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt
3, 启动服务

    python entry_with_update.py
    # 云端部署可以配置： ip, port, webroot 等参数
    # python entry_with_update.py --listen 0.0.0.0 --port 8889 --webroot /sdxl --preset realistic
    # Enter English UI : --language en
    # python entry_with_update.py --language en

模型库文件同步: 条件允许的可以直接同步模型库，不用一个个下载了。

    git lfs install
    git clone https://huggingface.co/metercai/SimpleSDXL models
    # 国内用户可以换魔搭社区的源
    # git clone https://www.modelscope.cn/metercai/SimpleSDXL-models.git models

---
<div align=center>
<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/483fb86d-c9a2-4c20-997c-46dafc124f25">

**Non-cherry-picked** random batch by just typing two words "forest elf", 

without any parameter tweaking, without any strange prompt tags. 

See also **non-cherry-picked** generalization and diversity tests [here](https://github.com/lllyasviel/Fooocus/discussions/808) and [here](https://github.com/lllyasviel/Fooocus/discussions/679) and [here](https://github.com/lllyasviel/Fooocus/discussions/679#realistic).

In the entire open source community, only Fooocus can achieve this level of **non-cherry-picked** quality.

</div>


# Fooocus

Fooocus is an image generating software (based on [Gradio](https://www.gradio.app/)).

Fooocus is a rethinking of Stable Diffusion and Midjourney’s designs:

* Learned from Stable Diffusion, the software is offline, open source, and free.

* Learned from Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images.

Fooocus has included and automated [lots of inner optimizations and quality improvements](#tech_list). Users can forget all those difficult technical parameters, and just enjoy the interaction between human and computer to "explore new mediums of thought and expanding the imaginative powers of the human species" `[1]`.

Fooocus has simplified the installation. Between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

`[1]` David Holz, 2019.

**Recently many fake websites exist on Google when you search “fooocus”. Do not trust those – here is the only official source of Fooocus.**

## [Installing Fooocus](#download)

# Moving from Midjourney to Fooocus

Using Fooocus is as easy as (probably easier than) Midjourney – but this does not mean we lack functionality. Below are the details.

| Midjourney | Fooocus |
| - | - |
| High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Unknown method) | High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Fooocus has an offline GPT-2 based prompt processing engine and lots of sampling improvements so that results are always beautiful, no matter if your prompt is as short as “house in garden” or as long as 1000 words) |
| V1 V2 V3 V4 | Input Image -> Upscale or Variation -> Vary (Subtle) / Vary (Strong)|
| U1 U2 U3 U4 | Input Image -> Upscale or Variation -> Upscale (1.5x) / Upscale (2x) |
| Inpaint / Up / Down / Left / Right (Pan) | Input Image -> Inpaint or Outpaint -> Inpaint / Up / Down / Left / Right <br> (Fooocus uses its own inpaint algorithm and inpaint models so that results are more satisfying than all other software that uses standard SDXL inpaint method/model) |
| Image Prompt | Input Image -> Image Prompt <br> (Fooocus uses its own image prompt algorithm so that result quality and prompt understanding are more satisfying than all other software that uses standard SDXL methods like standard IP-Adapters or Revisions) |
| --style | Advanced -> Style |
| --stylize | Advanced -> Advanced -> Guidance |
| --niji | [Multiple launchers: "run.bat", "run_anime.bat", and "run_realistic.bat".](https://github.com/lllyasviel/Fooocus/discussions/679) <br> Fooocus support SDXL models on Civitai <br> (You can google search “Civitai” if you do not know about it) |
| --quality | Advanced -> Quality |
| --repeat | Advanced -> Image Number |
| Multi Prompts (::) | Just use multiple lines of prompts |
| Prompt Weights | You can use " I am (happy:1.5)". <br> Fooocus uses A1111's reweighting algorithm so that results are better than ComfyUI if users directly copy prompts from Civitai. (Because if prompts are written in ComfyUI's reweighting, users are less likely to copy prompt texts as they prefer dragging files) <br> To use embedding, you can use "(embedding:file_name:1.1)" |
| --no | Advanced -> Negative Prompt |
| --ar | Advanced -> Aspect Ratios |
| InsightFace | Input Image -> Image Prompt -> Advanced -> FaceSwap |
| Describe | Input Image -> Describe |

We also have a few things borrowed from the best parts of LeonardoAI:

| LeonardoAI | Fooocus |
| - | - |
| Prompt Magic | Advanced -> Style -> Fooocus V2 |
| Advanced Sampler Parameters (like Contrast/Sharpness/etc) | Advanced -> Advanced -> Sampling Sharpness / etc |
| User-friendly ControlNets | Input Image -> Image Prompt -> Advanced |

Fooocus also developed many "fooocus-only" features for advanced users to get perfect results. [Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

# Download

### Windows

You can directly download Fooocus with:

**[>>> Click here to download <<<](https://github.com/lllyasviel/Fooocus/releases/download/release/Fooocus_win64_2-1-831.7z)**

After you download the file, please uncompress it and then run the "run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

The first time you launch the software, it will automatically download models:

1. It will download [default models](#models) to the folder "Fooocus\models\checkpoints" given different presets. You can download them in advance if you do not want automatic download.
2. Note that if you use inpaint, at the first time you inpaint an image, it will download [Fooocus's own inpaint control model from here](https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch) as the file "Fooocus\models\inpaint\inpaint_v26.fooocus.patch" (the size of this file is 1.28GB).

After Fooocus 2.1.60, you will also have `run_anime.bat` and `run_realistic.bat`. They are different model presets (and require different models, but they will be automatically downloaded). [Check here for more details](https://github.com/lllyasviel/Fooocus/discussions/679).

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

If you already have these files, you can copy them to the above locations to speed up installation.

Note that if you see **"MetadataIncompleteBuffer" or "PytorchStreamReader"**, then your model files are corrupted. Please download models again.

Below is a test on a relatively low-end laptop with **16GB System RAM** and **6GB VRAM** (Nvidia 3060 laptop). The speed on this machine is about 1.35 seconds per iteration. Pretty impressive – nowadays laptops with 3060 are usually at very acceptable price.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

Besides, recently many other software report that Nvidia driver above 532 is sometimes 10x slower than Nvidia driver 531. If your generation time is very long, consider download [Nvidia Driver 531 Laptop](https://www.nvidia.com/download/driverResults.aspx/199991/en-us/) or [Nvidia Driver 531 Desktop](https://www.nvidia.com/download/driverResults.aspx/199990/en-us/).

Note that the minimal requirement is **4GB Nvidia GPU memory (4GB VRAM)** and **8GB system memory (8GB RAM)**. This requires using Microsoft’s Virtual Swap technique, which is automatically enabled by your Windows installation in most cases, so you often do not need to do anything about it. However, if you are not sure, or if you manually turned it off (would anyone really do that?), or **if you see any "RuntimeError: CPUAllocator"**, you can enable it here:

<details>
<summary>Click here to see the image instructions. </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

**And make sure that you have at least 40GB free space on each drive if you still see "RuntimeError: CPUAllocator" !**

</details>

Please open an issue if you use similar devices but still cannot achieve acceptable performances.

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

See also the common problems and troubleshoots [here](troubleshoot.md).

### Colab

(Last tested - 2023 Dec 12)

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb) | Fooocus Official

In Colab, you can modify the last line to `!python entry_with_update.py --share` or `!python entry_with_update.py --preset anime --share` or `!python entry_with_update.py --preset realistic --share` for Fooocus Default/Anime/Realistic Edition.

Note that this Colab will disable refiner by default because Colab free's resources are relatively limited (and some "big" features like image prompt may cause free-tier Colab to disconnect). We make sure that basic text-to-image is always working on free-tier Colab.

Thanks to [camenduru](https://github.com/camenduru)!

### Linux (Using Anaconda)

If you want to use Anaconda/Miniconda, you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

Then download the models: download [default models](#models) to the folder "Fooocus\models\checkpoints". **Or let Fooocus automatically download the models** using the launcher:

    conda activate fooocus
    python entry_with_update.py

Or, if you want to open a remote port, use

    conda activate fooocus
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (Using Python Venv)

Your Linux needs to have **Python 3.10** installed, and let's say your Python can be called with the command **python3** with your venv system working; you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    python3 -m venv fooocus_env
    source fooocus_env/bin/activate
    pip install -r requirements_versions.txt

See the above sections for model downloads. You can launch the software with:

    source fooocus_env/bin/activate
    python entry_with_update.py

Or, if you want to open a remote port, use

    source fooocus_env/bin/activate
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (Using native system Python)

If you know what you are doing, and your Linux already has **Python 3.10** installed, and your Python can be called with the command **python3** (and Pip with **pip3**), you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    pip3 install -r requirements_versions.txt

See the above sections for model downloads. You can launch the software with:

    python3 entry_with_update.py

Or, if you want to open a remote port, use

    python3 entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (AMD GPUs)

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Same with the above instructions. You need to change torch to the AMD version

    pip uninstall torch torchvision torchaudio torchtext functorch xformers 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

AMD is not intensively tested, however. The AMD support is in beta.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Windows(AMD GPUs)

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Same with Windows. Download the software and edit the content of `run.bat` as:

    .\python_embeded\python.exe -m pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
    .\python_embeded\python.exe -m pip install torch-directml
    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --directml
    pause

Then run the `run.bat`.

AMD is not intensively tested, however. The AMD support is in beta.

For AMD, use `python entry_with_update.py --directml --preset anime` or `python entry_with_update.py --directml --preset realistic` for Fooocus Anime/Realistic Edition.

### Mac

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Mac is not intensively tested. Below is an unofficial guideline for using Mac. You can discuss problems [here](https://github.com/lllyasviel/Fooocus/pull/129).

You can install Fooocus on Apple Mac silicon (M1 or M2) with macOS 'Catalina' or a newer version. Fooocus runs on Apple silicon computers via [PyTorch](https://pytorch.org/get-started/locally/) MPS device acceleration. Mac Silicon computers don't come with a dedicated graphics card, resulting in significantly longer image processing times compared to computers with dedicated graphics cards.

1. Install the conda package manager and pytorch nightly. Read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide for instructions. Make sure pytorch recognizes your MPS device.
1. Open the macOS Terminal app and clone this repository with `git clone https://github.com/lllyasviel/Fooocus.git`.
1. Change to the new Fooocus directory, `cd Fooocus`.
1. Create a new conda environment, `conda env create -f environment.yaml`.
1. Activate your new conda environment, `conda activate fooocus`.
1. Install the packages required by Fooocus, `pip install -r requirements_versions.txt`.
1. Launch Fooocus by running `python entry_with_update.py`. (Some Mac M2 users may need `python entry_with_update.py --disable-offload-from-vram` to speed up model loading/unloading.) The first time you run Fooocus, it will automatically download the Stable Diffusion SDXL models and will take a significant amount of time, depending on your internet connection.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Download Previous Version

See the guidelines [here](https://github.com/lllyasviel/Fooocus/discussions/1405).

## Minimal Requirement

Below is the minimal requirement for running Fooocus locally. If your device capability is lower than this spec, you may not be able to use Fooocus locally. (Please let us know, in any case, if your device capability is lower but Fooocus still works.)

| Operating System  | GPU                          | Minimal GPU Memory           | Minimal System Memory     | [System Swap](troubleshoot.md) | Note                                                                       |
|-------------------|------------------------------|------------------------------|---------------------------|--------------------------------|----------------------------------------------------------------------------|
| Windows/Linux     | Nvidia RTX 4XXX              | 4GB                          | 8GB                       | Required                       | fastest                                                                    |
| Windows/Linux     | Nvidia RTX 3XXX              | 4GB                          | 8GB                       | Required                       | usually faster than RTX 2XXX                                               |
| Windows/Linux     | Nvidia RTX 2XXX              | 4GB                          | 8GB                       | Required                       | usually faster than GTX 1XXX                                               |
| Windows/Linux     | Nvidia GTX 1XXX              | 8GB (&ast; 6GB uncertain)    | 8GB                       | Required                       | only marginally faster than CPU                                            |
| Windows/Linux     | Nvidia GTX 9XX               | 8GB                          | 8GB                       | Required                       | faster or slower than CPU                                                  |
| Windows/Linux     | Nvidia GTX < 9XX             | Not supported                | /                         | /                              | /                                                                          |
| Windows           | AMD GPU                      | 8GB    (updated 2023 Dec 30) | 8GB                       | Required                       | via DirectML (&ast; ROCm is on hold), about 3x slower than Nvidia RTX 3XXX |
| Linux             | AMD GPU                      | 8GB                          | 8GB                       | Required                       | via ROCm, about 1.5x slower than Nvidia RTX 3XXX                           |
| Mac               | M1/M2 MPS                    | Shared                       | Shared                    | Shared                         | about 9x slower than Nvidia RTX 3XXX                                       |
| Windows/Linux/Mac | only use CPU                 | 0GB                          | 32GB                      | Required                       | about 17x slower than Nvidia RTX 3XXX                                      |

&ast; AMD GPU ROCm (on hold): The AMD is still working on supporting ROCm on Windows.

&ast; Nvidia GTX 1XXX 6GB uncertain: Some people report 6GB success on GTX 10XX, but some other people report failure cases.

*Note that Fooocus is only for extremely high quality image generating. We will not support smaller models to reduce the requirement and sacrifice result quality.*

## Troubleshoot

See the common problems [here](troubleshoot.md).

## Default Models
<a name="models"></a>

Given different goals, the default models and configs of Fooocus are different:

| Task | Windows | Linux args | Main Model | Refiner | Config |
| --- | --- | --- | --- | --- | --- |
| General | run.bat |  | [juggernautXL v6_RunDiffusion](https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors) | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/modules/path.py) |
| Realistic | run_realistic.bat | --preset realistic | [realistic_stock_photo](https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticStockPhoto_v10.safetensors) | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/realistic.json) |
| Anime | run_anime.bat | --preset anime | [bluepencil_v50](https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/bluePencilXL_v050.safetensors) | [dreamsharper_v8](https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/DreamShaper_8_pruned.safetensors) (SD1.5) | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/anime.json) |

Note that the download is **automatic** - you do not need to do anything if the internet connection is okay. However, you can download them manually if you (or move them from somewhere else) have your own preparation.

## List of "Hidden" Tricks
<a name="tech_list"></a>

The below things are already inside the software, and **users do not need to do anything about these**.

1. GPT2-based [prompt expansion as a dynamic style "Fooocus V2".](https://github.com/lllyasviel/Fooocus/discussions/117#raw) (similar to Midjourney's hidden pre-processsing and "raw" mode, or the LeonardoAI's Prompt Magic).
2. Native refiner swap inside one single k-sampler. The advantage is that the refiner model can now reuse the base model's momentum (or ODE's history parameters) collected from k-sampling to achieve more coherent sampling. In Automatic1111's high-res fix and ComfyUI's node system, the base model and refiner use two independent k-samplers, which means the momentum is largely wasted, and the sampling continuity is broken. Fooocus uses its own advanced k-diffusion sampling that ensures seamless, native, and continuous swap in a refiner setup. (Update Aug 13: Actually, I discussed this with Automatic1111 several days ago, and it seems that the “native refiner swap inside one single k-sampler” is [merged]( https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371) into the dev branch of webui. Great!)
3. Negative ADM guidance. Because the highest resolution level of XL Base does not have cross attentions, the positive and negative signals for XL's highest resolution level cannot receive enough contrasts during the CFG sampling, causing the results to look a bit plastic or overly smooth in certain cases. Fortunately, since the XL's highest resolution level is still conditioned on image aspect ratios (ADM), we can modify the adm on the positive/negative side to compensate for the lack of CFG contrast in the highest resolution level. (Update Aug 16, the IOS App [Drawing Things](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820) will support Negative ADM Guidance. Great!)
4. We implemented a carefully tuned variation of Section 5.1 of ["Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"](https://arxiv.org/pdf/2210.00939.pdf). The weight is set to very low, but this is Fooocus's final guarantee to make sure that the XL will never yield an overly smooth or plastic appearance (examples [here](https://github.com/lllyasviel/Fooocus/discussions/117#sharpness)). This can almost eliminate all cases for which XL still occasionally produces overly smooth results, even with negative ADM guidance. (Update 2023 Aug 18, the Gaussian kernel of SAG is changed to an anisotropic kernel for better structure preservation and fewer artifacts.)
5. We modified the style templates a bit and added the "cinematic-default".
6. We tested the "sd_xl_offset_example-lora_1.0.safetensors" and it seems that when the lora weight is below 0.5, the results are always better than XL without lora.
7. The parameters of samplers are carefully tuned.
8. Because XL uses positional encoding for generation resolution, images generated by several fixed resolutions look a bit better than those from arbitrary resolutions (because the positional encoding is not very good at handling int numbers that are unseen during training). This suggests that the resolutions in UI may be hard coded for best results.
9. Separated prompts for two different text encoders seem unnecessary. Separated prompts for the base model and refiner may work, but the effects are random, and we refrain from implementing this.
10. The DPM family seems well-suited for XL since XL sometimes generates overly smooth texture, but the DPM family sometimes generates overly dense detail in texture. Their joint effect looks neutral and appealing to human perception.
11. A carefully designed system for balancing multiple styles as well as prompt expansion.
12. Using automatic1111's method to normalize prompt emphasizing. This significantly improves results when users directly copy prompts from civitai.
13. The joint swap system of the refiner now also supports img2img and upscale in a seamless way.
14. CFG Scale and TSNR correction (tuned for SDXL) when CFG is bigger than 10.

## Customization

After the first time you run Fooocus, a config file will be generated at `Fooocus\config.txt`. This file can be edited to change the model path or default parameters.

For example, an edited `Fooocus\config.txt` (this file will be generated after the first launch) may look like this:

```json
{
    "path_checkpoints": "D:\\Fooocus\\models\\checkpoints",
    "path_loras": "D:\\Fooocus\\models\\loras",
    "path_embeddings": "D:\\Fooocus\\models\\embeddings",
    "path_vae_approx": "D:\\Fooocus\\models\\vae_approx",
    "path_upscale_models": "D:\\Fooocus\\models\\upscale_models",
    "path_inpaint": "D:\\Fooocus\\models\\inpaint",
    "path_controlnet": "D:\\Fooocus\\models\\controlnet",
    "path_clip_vision": "D:\\Fooocus\\models\\clip_vision",
    "path_fooocus_expansion": "D:\\Fooocus\\models\\prompt_expansion\\fooocus_expansion",
    "path_outputs": "D:\\Fooocus\\outputs",
    "default_model": "realisticStockPhoto_v10.safetensors",
    "default_refiner": "",
    "default_loras": [["lora_filename_1.safetensors", 0.5], ["lora_filename_2.safetensors", 0.5]],
    "default_cfg_scale": 3.0,
    "default_sampler": "dpmpp_2m",
    "default_scheduler": "karras",
    "default_negative_prompt": "low quality",
    "default_positive_prompt": "",
    "default_styles": [
        "Fooocus V2",
        "Fooocus Photograph",
        "Fooocus Negative"
    ]
}
```

Many other keys, formats, and examples are in `Fooocus\config_modification_tutorial.txt` (this file will be generated after the first launch).

Consider twice before you really change the config. If you find yourself breaking things, just delete `Fooocus\config.txt`. Fooocus will go back to default.

A safer way is just to try "run_anime.bat" or "run_realistic.bat" - they should already be good enough for different tasks.

~Note that `user_path_config.txt` is deprecated and will be removed soon.~ (Edit: it is already removed.)

### All CMD Flags

```
entry_with_update.py  [-h] [--listen [IP]] [--port PORT]
                      [--disable-header-check [ORIGIN]]
                      [--web-upload-size WEB_UPLOAD_SIZE]
                      [--external-working-path PATH [PATH ...]]
                      [--output-path OUTPUT_PATH] [--temp-path TEMP_PATH]
                      [--cache-path CACHE_PATH] [--in-browser]
                      [--disable-in-browser] [--gpu-device-id DEVICE_ID]
                      [--async-cuda-allocation | --disable-async-cuda-allocation]
                      [--disable-attention-upcast] [--all-in-fp32 | --all-in-fp16]
                      [--unet-in-bf16 | --unet-in-fp16 | --unet-in-fp8-e4m3fn | --unet-in-fp8-e5m2]
                      [--vae-in-fp16 | --vae-in-fp32 | --vae-in-bf16]
                      [--clip-in-fp8-e4m3fn | --clip-in-fp8-e5m2 | --clip-in-fp16 | --clip-in-fp32]
                      [--directml [DIRECTML_DEVICE]] [--disable-ipex-hijack]
                      [--preview-option [none,auto,fast,taesd]]
                      [--attention-split | --attention-quad | --attention-pytorch]
                      [--disable-xformers]
                      [--always-gpu | --always-high-vram | --always-normal-vram | 
                       --always-low-vram | --always-no-vram | --always-cpu]
                      [--always-offload-from-vram] [--disable-server-log]
                      [--debug-mode] [--is-windows-embedded-python]
                      [--disable-server-info] [--share] [--preset PRESET]
                      [--language LANGUAGE] [--disable-offload-from-vram]
                      [--theme THEME] [--disable-image-log]
```

## Advanced Features

[Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

Fooocus also has many community forks, just like SD-WebUI's [vladmandic/automatic](https://github.com/vladmandic/automatic) and [anapnoe/stable-diffusion-webui-ux](https://github.com/anapnoe/stable-diffusion-webui-ux), for enthusiastic users who want to try!

| Fooocus' forks |
| - |
| [fenneishi/Fooocus-Control](https://github.com/fenneishi/Fooocus-Control) </br>[runew0lf/RuinedFooocus](https://github.com/runew0lf/RuinedFooocus) </br> [MoonRide303/Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE) </br> [metercai/SimpleSDXL](https://github.com/metercai/SimpleSDXL) </br> and so on ... |

See also [About Forking and Promotion of Forks](https://github.com/lllyasviel/Fooocus/discussions/699).

## Thanks

Special thanks to [twri](https://github.com/twri) and [3Diva](https://github.com/3Diva) and [Marc K3nt3L](https://github.com/K3nt3L) for creating additional SDXL styles available in Fooocus. Thanks [daswer123](https://github.com/daswer123) for contributing the Canvas Zoom!

## Update Log

The log is [here](update_log.md).

## Localization/Translation/I18N

**We need your help!** Please help translate Fooocus into international languages.

You can put json files in the `language` folder to translate the user interface.

For example, below is the content of `Fooocus/language/example.json`:

```json
{
  "Generate": "生成",
  "Input Image": "入力画像",
  "Advanced": "고급",
  "SAI 3D Model": "SAI 3D Modèle"
}
```

If you add `--language example` arg, Fooocus will read `Fooocus/language/example.json` to translate the UI.

For example, you can edit the ending line of Windows `run.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example

Or `run_anime.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset anime

Or `run_realistic.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset realistic

For practical translation, you may create your own file like `Fooocus/language/jp.json` or `Fooocus/language/cn.json` and then use flag `--language jp` or `--language cn`. Apparently, these files do not exist now. **We need your help to create these files!**

Note that if no `--language` is given and at the same time `Fooocus/language/default.json` exists, Fooocus will always load `Fooocus/language/default.json` for translation. By default, the file `Fooocus/language/default.json` does not exist.
