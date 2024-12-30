@echo off
setlocal enabledelayedexpansion
chcp 65001
rem ANSI escape codes for colors
set "GREEN=[32m"
set "RED=[31m"
set "RESET=[0m"
echo.
echo !RED!##################请将本脚本置于【常规启动】脚本旁运行以获得正确的结果##################!RESET!
echo.
REM 设置 Python 独立环境的路径
set PYTHON_EMBEDDED_DIR=%~dp0python_embeded

:: 设置初始路径
set PYTHON_EMBEDDED_DIR=%~dp0python_embeded

:: 尝试运行 python.exe 并捕获输出
set "PYTHON_EXE=%PYTHON_EMBEDDED_DIR%\python.exe"

:: 使用 where 命令查找 python.exe 的实际路径
for /f "delims=" %%i in ('where /r %PYTHON_EMBEDDED_DIR% python.exe 2^>nul') do (
    set "ACTUAL_PYTHON_PATH=%%i"
)

:: 检查是否找到了实际路径
if defined ACTUAL_PYTHON_PATH (
    echo Python.exe 实际指向的路径是: %ACTUAL_PYTHON_PATH%
) else (
    echo 未找到 python.exe 文件。
    echo.
    echo 按任意键继续。
    pause
)
REM 检查 SimpleSDXL\entry_with_update.py 文件是否存在
set "SCRIPT_FILE=%~dp0SimpleSDXL\entry_with_update.py"
echo 检查的脚本文件路径：%SCRIPT_FILE%

if exist "%SCRIPT_FILE%" (
    echo 找到 entry_with_update.py 文件。
) else (
    echo 未找到 entry_with_update.py 文件。
    echo.
    echo 按任意键继续。
    pause
)

REM 调用 PowerShell 并获取输出
for /f "delims=" %%a in ('powershell -NoProfile -Command "[math]::Round((Get-WmiObject Win32_OperatingSystem).TotalVirtualMemorySize / 1KB)"') do set TOTAL_VIRTUAL=%%a
REM 输出虚拟内存值
echo.
echo 总虚拟内存: !GREEN!%TOTAL_VIRTUAL% MB!RESET!
echo.
echo !GREEN!★!RESET!本脚本含有自动解压功能，若已经手动解压模型包并合并SimpleModels目录，使用!GREEN!常规启动!RESET!运行，不需要再执行安装模型启动。!GREEN!★!RESET!
if %TOTAL_VIRTUAL% LSS 40960 (
    echo 虚拟内存不足，总虚拟内存小于40G，建议开到60G以上，请增加虚拟内存再启动。操作指导https://www.bilibili.com/video/BV1ddkdYcEWg/
    echo 虚拟内存设置在固态硬盘分区，windows家庭版一般只能设置在系统分区，疑难杂症进QQ群求助:938075852
    pause
    echo.
    echo 按任意键继续。
)
echo.
echo !GREEN!★★★★★!RESET!安装视频教程https://www.bilibili.com/video/BV1ddkdYcEWg/ !GREEN!★★★★★!RESET!!GREEN!★!RESET!
echo.
echo !GREEN!★!RESET!攻略地址飞书文档:https://acnmokx5gwds.feishu.cn/wiki/QK3LwOp2oiRRaTkFRhYcO4LonGe 文章无权限即为未编辑完毕。!GREEN!★!RESET!
echo.
echo !GREEN!★!RESET!稳速生图指南:Nvidia显卡驱动选择最新版驱动,驱动类型最好为Studio。在遇到生图速度断崖式下降或者爆显存OutOfMemory时,提高!GREEN!预留显存功能!RESET!的数值至（1~2）!GREEN!★!RESET!
echo.
echo !GREEN!★!RESET!打开默认浏览器设置，关闭GPU加速、或图形加速的选项。!GREEN!★!RESET!大内存(64+)与固态硬盘存放模型有助于减少模型加载时间。!GREEN!★!RESET!
echo.
echo !GREEN!★!RESET!疑难杂症进QQ群求助：938075852!GREEN!★!RESET!脚本：✿   冰華 !GREEN!★!RESET!

set "current_dir=%cd%"
for %%I in ("%current_dir%\..") do set "root=%%~fI"
set root=%root%\

if not exist "%root%users" mkdir "%root%users"
if not exist "%root%SimpleModels" mkdir "%root%SimpleModels"
echo.
echo -----------------开始检测基础包（必要）-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\checkpoints\juggernautXL_juggXIByRundiffusion.safetensors,7105350536
echo SimpleModels\checkpoints\realisticVisionV60B1_v51VAE.safetensors,2132625894
echo SimpleModels\clip_vision\clip_vision_vit_h.safetensors,1972298538
echo SimpleModels\clip_vision\model_base_caption_capfilt_large.pth,896081425
echo SimpleModels\clip_vision\wd-v1-4-moat-tagger-v2.onnx,326197340
echo SimpleModels\clip_vision\clip-vit-large-patch14\merges.txt,524619
echo SimpleModels\clip_vision\clip-vit-large-patch14\special_tokens_map.json,389
echo SimpleModels\clip_vision\clip-vit-large-patch14\tokenizer_config.json,905
echo SimpleModels\clip_vision\clip-vit-large-patch14\vocab.json,961143
echo SimpleModels\configs\anything_v3.yaml,1933
echo SimpleModels\configs\v1-inference.yaml,1873
echo SimpleModels\configs\v1-inference_clip_skip_2.yaml,1933
echo SimpleModels\configs\v1-inference_clip_skip_2_fp16.yaml,1956
echo SimpleModels\configs\v1-inference_fp16.yaml,1896
echo SimpleModels\configs\v1-inpainting-inference.yaml,1992
echo SimpleModels\configs\v2-inference-v.yaml,1815
echo SimpleModels\configs\v2-inference-v_fp32.yaml,1816
echo SimpleModels\configs\v2-inference.yaml,1789
echo SimpleModels\configs\v2-inference_fp32.yaml,1790
echo SimpleModels\configs\v2-inpainting-inference.yaml,4450
echo SimpleModels\controlnet\control-lora-canny-rank128.safetensors,395733680
echo SimpleModels\controlnet\detection_Resnet50_Final.pth,109497761
echo SimpleModels\controlnet\fooocus_ip_negative.safetensors,65616
echo SimpleModels\controlnet\fooocus_xl_cpds_128.safetensors,395706528
echo SimpleModels\controlnet\ip-adapter-plus-face_sdxl_vit-h.bin,1013454761
echo SimpleModels\controlnet\ip-adapter-plus_sdxl_vit-h.bin,1013454427
echo SimpleModels\controlnet\parsing_parsenet.pth,85331193
echo SimpleModels\controlnet\xinsir_cn_openpose_sdxl_1.0.safetensors,2502139104
echo SimpleModels\controlnet\lllyasviel\Annotators\body_pose_model.pth,209267595
echo SimpleModels\controlnet\lllyasviel\Annotators\facenet.pth,153718792
echo SimpleModels\controlnet\lllyasviel\Annotators\hand_pose_model.pth,147341049
echo SimpleModels\inpaint\fooocus_inpaint_head.pth,52602
echo SimpleModels\inpaint\groundingdino_swint_ogc.pth,693997677
echo SimpleModels\inpaint\inpaint_v26.fooocus.patch,1323362033
echo SimpleModels\inpaint\isnet-anime.onnx,176069933
echo SimpleModels\inpaint\isnet-general-use.onnx,178648008
echo SimpleModels\inpaint\sam_vit_b_01ec64.pth,375042383
echo SimpleModels\inpaint\silueta.onnx,44173029
echo SimpleModels\inpaint\u2net.onnx,175997641
echo SimpleModels\inpaint\u2netp.onnx,4574861
echo SimpleModels\inpaint\u2net_cloth_seg.onnx,176194565
echo SimpleModels\inpaint\u2net_human_seg.onnx,175997641
echo SimpleModels\layer_model\layer_xl_fg2ble.safetensors,701981624
echo SimpleModels\layer_model\layer_xl_transparent_conv.safetensors,3619745776
echo SimpleModels\layer_model\vae_transparent_decoder.safetensors,208266320
echo SimpleModels\llms\bert-base-uncased\config.json,570
echo SimpleModels\llms\bert-base-uncased\model.safetensors,440449768
echo SimpleModels\llms\bert-base-uncased\tokenizer.json,466062
echo SimpleModels\llms\bert-base-uncased\tokenizer_config.json,28
echo SimpleModels\llms\bert-base-uncased\vocab.txt,231508
echo SimpleModels\llms\Helsinki-NLP\opus-mt-zh-en\config.json,1394
echo SimpleModels\llms\Helsinki-NLP\opus-mt-zh-en\generation_config.json,293
echo SimpleModels\llms\Helsinki-NLP\opus-mt-zh-en\metadata.json,1477
echo SimpleModels\llms\Helsinki-NLP\opus-mt-zh-en\pytorch_model.bin,312087009
echo SimpleModels\llms\Helsinki-NLP\opus-mt-zh-en\source.spm,804677
echo SimpleModels\llms\Helsinki-NLP\opus-mt-zh-en\target.spm,806530
echo SimpleModels\llms\Helsinki-NLP\opus-mt-zh-en\tokenizer_config.json,44
echo SimpleModels\llms\Helsinki-NLP\opus-mt-zh-en\vocab.json,1617902
echo SimpleModels\llms\superprompt-v1\config.json,1512
echo SimpleModels\llms\superprompt-v1\generation_config.json,142
echo SimpleModels\llms\superprompt-v1\model.safetensors,307867048
echo SimpleModels\llms\superprompt-v1\README.md,3661
echo SimpleModels\llms\superprompt-v1\spiece.model,791656
echo SimpleModels\llms\superprompt-v1\tokenizer.json,2424064
echo SimpleModels\llms\superprompt-v1\tokenizer_config.json,2539
echo SimpleModels\loras\ip-adapter-faceid-plusv2_sdxl_lora.safetensors,371842896
echo SimpleModels\loras\sdxl_hyper_sd_4step_lora.safetensors,787359648
echo SimpleModels\loras\sdxl_lightning_4step_lora.safetensors,393854592
echo SimpleModels\loras\sd_xl_offset_example-lora_1.0.safetensors,49553604
echo SimpleModels\prompt_expansion\fooocus_expansion\config.json,937
echo SimpleModels\prompt_expansion\fooocus_expansion\merges.txt,456356
echo SimpleModels\prompt_expansion\fooocus_expansion\positive.txt,5655
echo SimpleModels\prompt_expansion\fooocus_expansion\pytorch_model.bin,351283802
echo SimpleModels\prompt_expansion\fooocus_expansion\special_tokens_map.json,99
echo SimpleModels\prompt_expansion\fooocus_expansion\tokenizer.json,2107625
echo SimpleModels\prompt_expansion\fooocus_expansion\tokenizer_config.json,255
echo SimpleModels\prompt_expansion\fooocus_expansion\vocab.json,798156
echo SimpleModels\rembg\RMBG-1.4.pth,176718373
echo SimpleModels\unet\iclight_sd15_fc_unet_ldm.safetensors,1719144856
echo SimpleModels\upscale_models\fooocus_upscaler_s409985e5.bin,33636613
echo SimpleModels\vae_approx\vaeapp_sd15.pth,213777
echo SimpleModels\vae_approx\xl-to-v1_interposer-v4.0.safetensors,5667280
echo SimpleModels\vae_approx\xlvaeapp.pth,213777
echo SimpleModels\clip\clip_l.safetensors,246144152
echo SimpleModels\vae\ponyDiffusionV6XL_vae.safetensors,334641162
echo SimpleModels\loras\Hyper-SDXL-8steps-lora.safetensors,787359648
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有基础包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo 请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_base_simpleai_1214.zip补充基础模型包。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_base_simpleai_1214.zip下载基础包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√基础包全部验证通过，可正常开启SDXL模型生图、修图功能!RESET!
    echo.
)


echo -----------------开始检测增强扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\embeddings\unaestheticXLhk1.safetensors,33296
echo SimpleModels\embeddings\unaestheticXLv31.safetensors,33296
echo SimpleModels\inpaint\inpaint_v25.fooocus.patch,2580722369
echo SimpleModels\inpaint\sam_vit_h_4b8939.pth,2564550879
echo SimpleModels\inpaint\sam_vit_l_0b3195.pth,1249524607
echo SimpleModels\layer_model\layer_xl_bg2ble.safetensors,701981624
echo SimpleModels\layer_model\layer_xl_transparent_attn.safetensors,743352688
echo SimpleModels\llms\nllb-200-distilled-600M\pytorch_model.bin,2460457927
echo SimpleModels\llms\nllb-200-distilled-600M\sentencepiece.bpe.model,4852054
echo SimpleModels\llms\nllb-200-distilled-600M\tokenizer.json,17331176
echo SimpleModels\loras\FilmVelvia3.safetensors,151108832
echo SimpleModels\loras\Hyper-SDXL-8steps-lora.safetensors,787359648
echo SimpleModels\loras\SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors,912593164
echo SimpleModels\safety_checker\stable-diffusion-safety-checker.bin,1216067303
echo SimpleModels\unet\iclight_sd15_fbc_unet_ldm.safetensors,1719167896
echo SimpleModels\upscale_models\4x-UltraSharp.pth,66961958
echo SimpleModels\vae\ponyDiffusionV6XL_vae.safetensors,334641162
echo SimpleModels\vae\sdxl_fp16.vae.safetensors,167335342
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有增强包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo 请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_enhance_simpleai_0908.zip补充增强模型包。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充!RESET!
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_enhance_simpleai_0908.zip下载增强包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√增强包全部验证通过!RESET!
    echo.
)

echo -----------------开始检测可图扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\diffusers\Kolors\model_index.json,427
echo SimpleModels\diffusers\Kolors\MODEL_LICENSE,14920
echo SimpleModels\diffusers\Kolors\README.md,4707
echo SimpleModels\diffusers\Kolors\scheduler\scheduler_config.json,606
echo SimpleModels\diffusers\Kolors\text_encoder\config.json,1323
echo SimpleModels\diffusers\Kolors\text_encoder\configuration_chatglm.py,2332
echo SimpleModels\diffusers\Kolors\text_encoder\modeling_chatglm.py,55722
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00001-of-00007.bin,1827781090
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00002-of-00007.bin,1968299480
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00003-of-00007.bin,1927415036
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00004-of-00007.bin,1815225998
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00005-of-00007.bin,1968299544
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00006-of-00007.bin,1927415036
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00007-of-00007.bin,1052808542
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model.bin.index.json,20437
echo SimpleModels\diffusers\Kolors\text_encoder\quantization.py,14692
echo SimpleModels\diffusers\Kolors\text_encoder\tokenization_chatglm.py,12223
echo SimpleModels\diffusers\Kolors\text_encoder\tokenizer.model,1018370
echo SimpleModels\diffusers\Kolors\text_encoder\tokenizer_config.json,249
echo SimpleModels\diffusers\Kolors\text_encoder\vocab.txt,1018370
echo SimpleModels\diffusers\Kolors\tokenizer\tokenization_chatglm.py,12223
echo SimpleModels\diffusers\Kolors\tokenizer\tokenizer.model,1018370
echo SimpleModels\diffusers\Kolors\tokenizer\tokenizer_config.json,249
echo SimpleModels\diffusers\Kolors\tokenizer\vocab.txt,1018370
echo SimpleModels\diffusers\Kolors\unet\config.json,1785
echo SimpleModels\diffusers\Kolors\vae\config.json,611
echo SimpleModels\loras\Hyper-SDXL-8steps-lora.safetensors,787359648
echo SimpleModels\checkpoints\kolors_unet_fp16.safetensors,5159140240
echo SimpleModels\vae\sdxl_fp16.vae.safetensors,167335342
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有可图模型包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_kolors_fp16_simpleai_0909.zip补充可图模型包。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_kolors_fp16_simpleai_0909.zip下载可图基础模型包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√可图模型包全部验证通过,可正常开启可图预置包功能!RESET!
    echo.
)

echo -----------------开始检测额外模型扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\checkpoints\animaPencilXL_v500.safetensors,6938041144
echo SimpleModels\checkpoints\hunyuan_dit_1.2.safetensors,8240228270
echo SimpleModels\checkpoints\playground-v2.5-1024px.safetensors,6938040576
echo SimpleModels\checkpoints\ponyDiffusionV6XL.safetensors,6938041050
echo SimpleModels\checkpoints\realisticStockPhoto_v20.safetensors,6938054242
echo SimpleModels\checkpoints\sd3_medium_incl_clips_t5xxlfp8.safetensors,10867168284
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有扩展模型包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_ckpt_SD3_HY_PonyV6_PGv25_aPencilXL_rsPhoto_simpleai_0909.zip补充额外模型包。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_ckpt_SD3_HY_PonyV6_PGv25_aPencilXL_rsPhoto_simpleai_0909.zip下载扩展模型包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√动漫、混元、PG2.5、小马V6、写实、SD3m全部验证通过可使用对应预置包!RESET!
    echo.
)

echo -----------------开始检测Flux低显存扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\checkpoints\flux-hyp8-Q5_K_M.gguf,8421981408
echo SimpleModels\checkpoints\flux1-dev-bnb-nf4-v2.safetensors,12044280207
echo SimpleModels\clip\clip_l.safetensors,246144152
echo SimpleModels\clip\t5xxl_fp8_e4m3fn.safetensors,4893934904
echo SimpleModels\vae\ae.safetensors,335304388
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有Flux低显存包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_flux1_nf4v2_Q5km_t5f8_simpleai_0909.zip补充Flux低显存模型包。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_flux1_nf4v2_Q5km_t5f8_simpleai_0909.zip下载Flux低显存模型包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√Flux低显存包全部验证通过，可使用Flux、Flux火箭预置包文生图功能。!RESET!
    echo.
)

echo -----------------开始检测Flux全量扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\checkpoints\flux1-dev.safetensors,23802932552
echo SimpleModels\clip\clip_l.safetensors,246144152
echo SimpleModels\clip\t5xxl_fp16.safetensors,9787841024
echo SimpleModels\vae\ae.safetensors,335304388
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有Flux全量模型包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_flux1_fp16_simpleai_0909.zip补充Flux全量模型包。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_flux1_fp16_simpleai_0909.zip下载FLUX全量扩展包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√Flux全量模型包全部验证通过，可使用Flux预置包文生图功能！!RESET!
    echo.
)

echo -----------------开始检测Flux_AIO扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\checkpoints\flux-hyp8-Q5_K_M.gguf,8421981408
echo SimpleModels\checkpoints\flux1-fill-dev-hyp8-Q4_K_S.gguf,6809920800
echo SimpleModels\clip\clip_l.safetensors,246144152
echo SimpleModels\clip\EVA02_CLIP_L_336_psz14_s6B.pt,856461210
echo SimpleModels\clip\t5xxl_fp16.safetensors,9787841024
echo SimpleModels\clip\t5xxl_fp8_e4m3fn.safetensors,4893934904
echo SimpleModels\clip_vision\sigclip_vision_patch14_384.safetensors,856505640
echo SimpleModels\controlnet\flux.1-dev_controlnet_union_pro.safetensors,6603953920
echo SimpleModels\controlnet\flux.1-dev_controlnet_upscaler.safetensors,3583232168
echo SimpleModels\controlnet\parsing_bisenet.pth,53289463
echo SimpleModels\controlnet\lllyasviel\Annotators\ZoeD_M12_N.pt,1443406099
echo SimpleModels\insightface\models\antelopev2\1k3d68.onnx,143607619
echo SimpleModels\insightface\models\antelopev2\2d106det.onnx,5030888
echo SimpleModels\insightface\models\antelopev2\genderage.onnx,1322532
echo SimpleModels\insightface\models\antelopev2\glintr100.onnx,260665334
echo SimpleModels\insightface\models\antelopev2\scrfd_10g_bnkps.onnx,16923827
echo SimpleModels\loras\flux1-canny-dev-lora.safetensors,1244443944
echo SimpleModels\loras\flux1-depth-dev-lora.safetensors,1244440512
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有FluxAIO模型包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_flux_aio_simpleai_1214.zip。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_flux_aio_simpleai_1214.zip下载FLUX全功能扩展包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√Flux_AIO模型包全部验证通过，可使用Flux_AIO预置包所有图生图功能！!RESET!
    echo.
)

echo -----------------开始检测SD1.5_AIO扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\checkpoints\realisticVisionV60B1_v51VAE.safetensors,2132625894
echo SimpleModels\loras\sd_xl_offset_example-lora_1.0.safetensors,49553604
echo SimpleModels\clip\sd15_clip_model.fp16.safetensors,246144864
echo SimpleModels\controlnet\control_v11f1e_sd15_tile_fp16.safetensors,722601104
echo SimpleModels\controlnet\control_v11f1p_sd15_depth_fp16.safetensors,722601100
echo SimpleModels\controlnet\control_v11p_sd15_canny_fp16.safetensors,722601100
echo SimpleModels\controlnet\control_v11p_sd15_openpose_fp16.safetensors,722601100
echo SimpleModels\controlnet\lllyasviel\Annotators\ZoeD_M12_N.pt,1443406099
echo SimpleModels\inpaint\sd15_powerpaint_brushnet_clip_v2_1.bin,492401329
echo SimpleModels\inpaint\sd15_powerpaint_brushnet_v2_1.safetensors,3544366408
echo SimpleModels\insightface\models\buffalo_l\1k3d68.onnx,143607619
echo SimpleModels\insightface\models\buffalo_l\2d106det.onnx,5030888
echo SimpleModels\insightface\models\buffalo_l\det_10g.onnx,16923827
echo SimpleModels\insightface\models\buffalo_l\genderage.onnx,1322532
echo SimpleModels\insightface\models\buffalo_l\w600k_r50.onnx,174383860
echo SimpleModels\ipadapter\clip-vit-h-14-laion2B-s32B-b79K.safetensors,3944517836
echo SimpleModels\ipadapter\ip-adapter-faceid-plusv2_sd15.bin,156558509
echo SimpleModels\ipadapter\ip-adapter_sd15.safetensors,44642768
echo SimpleModels\loras\ip-adapter-faceid-plusv2_sd15_lora.safetensors,51059544
echo SimpleModels\upscale_models\4x-UltraSharp.pth,66961958
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有SD1.5_AIO模型包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_sd15_aio_simpleai_1214.zip。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_sd15_aio_simpleai_1214.zip下载SD15全功能扩展包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√SD1.5_AIO模型包全部验证通过，可使用SD1.5_AIO预置包所有图生图功能！!RESET!
    echo.
)

echo -----------------开始检测Kolors_AIO扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\checkpoints\kolors_unet_fp16.safetensors,5159140240
echo SimpleModels\clip_vision\kolors_clip_ipa_plus_vit_large_patch14_336.bin,1711974081
echo SimpleModels\controlnet\kolors_controlnet_canny.safetensors,2526129624
echo SimpleModels\controlnet\kolors_controlnet_depth.safetensors,2526129624
echo SimpleModels\controlnet\kolors_controlnet_pose.safetensors,2526129624
echo SimpleModels\controlnet\lllyasviel\Annotators\ZoeD_M12_N.pt,1443406099
echo SimpleModels\diffusers\Kolors\model_index.json,427
echo SimpleModels\diffusers\Kolors\MODEL_LICENSE,14920
echo SimpleModels\diffusers\Kolors\README.md,4707
echo SimpleModels\diffusers\Kolors\scheduler\scheduler_config.json,606
echo SimpleModels\diffusers\Kolors\text_encoder\config.json,1323
echo SimpleModels\diffusers\Kolors\text_encoder\configuration_chatglm.py,2332
echo SimpleModels\diffusers\Kolors\text_encoder\modeling_chatglm.py,55722
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00001-of-00007.bin,1827781090
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00002-of-00007.bin,1968299480
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00003-of-00007.bin,1927415036
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00004-of-00007.bin,1815225998
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00005-of-00007.bin,1968299544
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00006-of-00007.bin,1927415036
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00007-of-00007.bin,1052808542
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model.bin.index.json,20437
echo SimpleModels\diffusers\Kolors\text_encoder\quantization.py,14692
echo SimpleModels\diffusers\Kolors\text_encoder\tokenization_chatglm.py,12223
echo SimpleModels\diffusers\Kolors\text_encoder\tokenizer.model,1018370
echo SimpleModels\diffusers\Kolors\text_encoder\tokenizer_config.json,249
echo SimpleModels\diffusers\Kolors\text_encoder\vocab.txt,1018370
echo SimpleModels\diffusers\Kolors\tokenizer\tokenization_chatglm.py,12223
echo SimpleModels\diffusers\Kolors\tokenizer\tokenizer.model,1018370
echo SimpleModels\diffusers\Kolors\tokenizer\tokenizer_config.json,249
echo SimpleModels\diffusers\Kolors\tokenizer\vocab.txt,1018370
echo SimpleModels\diffusers\Kolors\unet\config.json,1785
echo SimpleModels\diffusers\Kolors\vae\config.json,611
echo SimpleModels\insightface\models\antelopev2\1k3d68.onnx,143607619
echo SimpleModels\insightface\models\antelopev2\2d106det.onnx,5030888
echo SimpleModels\insightface\models\antelopev2\genderage.onnx,1322532
echo SimpleModels\insightface\models\antelopev2\glintr100.onnx,260665334
echo SimpleModels\insightface\models\antelopev2\scrfd_10g_bnkps.onnx,16923827
echo SimpleModels\ipadapter\kolors_ipa_faceid_plus.bin,2385842603
echo SimpleModels\ipadapter\kolors_ip_adapter_plus_general.bin,1013163359
echo SimpleModels\loras\Hyper-SDXL-8steps-lora.safetensors,787359648
echo SimpleModels\unet\kolors_inpainting.safetensors,5159169040
echo SimpleModels\upscale_models\4x-UltraSharp.pth,66961958
echo SimpleModels\vae\sdxl_fp16.vae.safetensors,167335342
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有Kolors_AIO模型包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_kolors_aio_simpleai_1214.zip。压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包.
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_kolors_aio_simpleai_1214.zip下载可图全功能扩展包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√Kolors_AIO模型包全部验证通过，可使用Kolors_AIO预置包所有图生图功能！!RESET!
    echo.
)

echo -----------------开始检测SD3x-medium扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\checkpoints\sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors,11638004202
echo SimpleModels\clip\clip_l.safetensors,246144152
echo SimpleModels\clip\t5xxl_fp8_e4m3fn.safetensors,4893934904
echo SimpleModels\vae\sd3x_fp16.vae.safetensors,167666654
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有SD3x-medium扩展包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors
    echo 放置于SimpleModels\checkpoints文件夹内。其他文件缺失在https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels搜寻。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors，其他文件在https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels搜寻。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√SD3x-medium扩展模型全部验证通过，可使用SD3x预置包选择SD3.5-medium模型文生图功能！!RESET!
    echo.
)

echo -----------------开始检测SD3x-Large扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\clip\clip_g.safetensors,1389382176
echo SimpleModels\clip\clip_l.safetensors,246144152
echo SimpleModels\clip\t5xxl_fp16.safetensors,9787841024
echo SimpleModels\clip\t5xxl_fp8_e4m3fn.safetensors,4893934904
echo SimpleModels\vae\sd3x_fp16.vae.safetensors,167666654
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有SD3x模型包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_sd35_large_clips_simpleai_1214.zip 压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包。
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/models_sd35_large_clips_simpleai_1214.zip下载sd3x扩展包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√SD3x-Large模型包全部验证通过，可使用SD3x预置包选择SD3.5Large模型文生图功能！!RESET!
    echo.
)

echo -----------------开始检测MiniCPMv26反推扩展包-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\.gitattributes,1657
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\.mdl,49
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\.msc,1655
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\.mv,36
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\added_tokens.json,629
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\config.json,1951
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\configuration.json,27
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\configuration_minicpm.py,3280
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\generation_config.json,121
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\image_processing_minicpmv.py,16579
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\merges.txt,1671853
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\modeling_minicpmv.py,15738
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\modeling_navit_siglip.py,41835
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\preprocessor_config.json,714
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\processing_minicpmv.py,9962
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\pytorch_model-00001-of-00002.bin,4454731094
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\pytorch_model-00002-of-00002.bin,1503635286
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\pytorch_model.bin.index.json,233389
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\README.md,2124
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\resampler.py,34699
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\special_tokens_map.json,1041
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\test.py,1162
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\tokenization_minicpmv_fast.py,1659
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\tokenizer.json,7032006
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\tokenizer_config.json,5663
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\vocab.json,2776833
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!有MiniCPMv26反推扩展包文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!请使用工具下载以下链接https://hf-mirror.com/metercai/SimpleSDXL2/blob/main/models_minicpm_v2.6_prompt_simpleai_1224.zip 压缩包放于SimpleAI根目录再运行此脚本，按照指引解压安装模型包。
    echo 或于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 少量文件于https://hf-mirror.com/metercai/SimpleSDXL2/tree/main/SimpleModels收集补充，大量文件请于https://hf-mirror.com/metercai/SimpleSDXL2/blob/main/models_minicpm_v2.6_prompt_simpleai_1224.zip下载MiniCPMv26反推扩展包覆盖。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√MiniCPMv26反推扩展包全部验证通过，可使用MiniCPMv26反推、翻译、扩展提示词功能！!RESET!
    echo.
)
echo -----------------开始检测贺年卡所需文件-----------------
:: 定义文件及其期望大小
(
echo SimpleModels\loras\flux_graffiti_v1.safetensors,612893792
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\.gitattributes,1657
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\.mdl,49
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\.msc,1655
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\.mv,36
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\added_tokens.json,629
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\config.json,1951
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\configuration.json,27
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\configuration_minicpm.py,3280
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\generation_config.json,121
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\image_processing_minicpmv.py,16579
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\merges.txt,1671853
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\modeling_minicpmv.py,15738
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\modeling_navit_siglip.py,41835
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\preprocessor_config.json,714
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\processing_minicpmv.py,9962
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\pytorch_model-00001-of-00002.bin,4454731094
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\pytorch_model-00002-of-00002.bin,1503635286
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\pytorch_model.bin.index.json,233389
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\README.md,2124
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\resampler.py,34699
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\special_tokens_map.json,1041
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\test.py,1162
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\tokenization_minicpmv_fast.py,1659
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\tokenizer.json,7032006
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\tokenizer_config.json,5663
echo SimpleModels\llms\MiniCPMv2_6-prompt-generator\vocab.json,2776833
echo SimpleModels\checkpoints\flux-hyp8-Q5_K_M.gguf,8421981408
echo SimpleModels\checkpoints\flux1-fill-dev-hyp8-Q4_K_S.gguf,6809920800
echo SimpleModels\clip\clip_l.safetensors,246144152
echo SimpleModels\clip\EVA02_CLIP_L_336_psz14_s6B.pt,856461210
echo SimpleModels\clip\t5xxl_fp16.safetensors,9787841024
echo SimpleModels\clip\t5xxl_fp8_e4m3fn.safetensors,4893934904
echo SimpleModels\clip_vision\sigclip_vision_patch14_384.safetensors,856505640
echo SimpleModels\controlnet\flux.1-dev_controlnet_union_pro.safetensors,6603953920
echo SimpleModels\controlnet\flux.1-dev_controlnet_upscaler.safetensors,3583232168
echo SimpleModels\controlnet\parsing_bisenet.pth,53289463
echo SimpleModels\controlnet\lllyasviel\Annotators\ZoeD_M12_N.pt,1443406099
echo SimpleModels\insightface\models\antelopev2\1k3d68.onnx,143607619
echo SimpleModels\insightface\models\antelopev2\2d106det.onnx,5030888
echo SimpleModels\insightface\models\antelopev2\genderage.onnx,1322532
echo SimpleModels\insightface\models\antelopev2\glintr100.onnx,260665334
echo SimpleModels\insightface\models\antelopev2\scrfd_10g_bnkps.onnx,16923827
echo SimpleModels\loras\flux1-canny-dev-lora.safetensors,1244443944
echo SimpleModels\loras\flux1-depth-dev-lora.safetensors,1244440512
echo SimpleModels\checkpoints\kolors_unet_fp16.safetensors,5159140240
echo SimpleModels\clip_vision\kolors_clip_ipa_plus_vit_large_patch14_336.bin,1711974081
echo SimpleModels\controlnet\kolors_controlnet_canny.safetensors,2526129624
echo SimpleModels\controlnet\kolors_controlnet_depth.safetensors,2526129624
echo SimpleModels\controlnet\kolors_controlnet_pose.safetensors,2526129624
echo SimpleModels\controlnet\lllyasviel\Annotators\ZoeD_M12_N.pt,1443406099
echo SimpleModels\diffusers\Kolors\model_index.json,427
echo SimpleModels\diffusers\Kolors\MODEL_LICENSE,14920
echo SimpleModels\diffusers\Kolors\README.md,4707
echo SimpleModels\diffusers\Kolors\scheduler\scheduler_config.json,606
echo SimpleModels\diffusers\Kolors\text_encoder\config.json,1323
echo SimpleModels\diffusers\Kolors\text_encoder\configuration_chatglm.py,2332
echo SimpleModels\diffusers\Kolors\text_encoder\modeling_chatglm.py,55722
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00001-of-00007.bin,1827781090
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00002-of-00007.bin,1968299480
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00003-of-00007.bin,1927415036
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00004-of-00007.bin,1815225998
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00005-of-00007.bin,1968299544
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00006-of-00007.bin,1927415036
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model-00007-of-00007.bin,1052808542
echo SimpleModels\diffusers\Kolors\text_encoder\pytorch_model.bin.index.json,20437
echo SimpleModels\diffusers\Kolors\text_encoder\quantization.py,14692
echo SimpleModels\diffusers\Kolors\text_encoder\tokenization_chatglm.py,12223
echo SimpleModels\diffusers\Kolors\text_encoder\tokenizer.model,1018370
echo SimpleModels\diffusers\Kolors\text_encoder\tokenizer_config.json,249
echo SimpleModels\diffusers\Kolors\text_encoder\vocab.txt,1018370
echo SimpleModels\diffusers\Kolors\tokenizer\tokenization_chatglm.py,12223
echo SimpleModels\diffusers\Kolors\tokenizer\tokenizer.model,1018370
echo SimpleModels\diffusers\Kolors\tokenizer\tokenizer_config.json,249
echo SimpleModels\diffusers\Kolors\tokenizer\vocab.txt,1018370
echo SimpleModels\diffusers\Kolors\unet\config.json,1785
echo SimpleModels\diffusers\Kolors\vae\config.json,611
echo SimpleModels\insightface\models\antelopev2\1k3d68.onnx,143607619
echo SimpleModels\insightface\models\antelopev2\2d106det.onnx,5030888
echo SimpleModels\insightface\models\antelopev2\genderage.onnx,1322532
echo SimpleModels\insightface\models\antelopev2\glintr100.onnx,260665334
echo SimpleModels\insightface\models\antelopev2\scrfd_10g_bnkps.onnx,16923827
echo SimpleModels\ipadapter\kolors_ipa_faceid_plus.bin,2385842603
echo SimpleModels\ipadapter\kolors_ip_adapter_plus_general.bin,1013163359
echo SimpleModels\loras\Hyper-SDXL-8steps-lora.safetensors,787359648
echo SimpleModels\unet\kolors_inpainting.safetensors,5159169040
echo SimpleModels\upscale_models\4x-UltraSharp.pth,66961958
echo SimpleModels\vae\sdxl_fp16.vae.safetensors,167335342
) > files_and_sizes.txt

set "all_passed=true"  :: 初始化一个标志变量为true
set "missing_files="
set "size_mismatch="
set "size_mismatch_files="
:: 遍历所有文件及其期望大小
for /f "delims=" %%A in (files_and_sizes.txt) do (
    rem 使用逗号分隔文件信息
    for /f "tokens=1,2 delims=," %%B in ("%%A") do (
        set "target_file=%root%%%B"
        set "expected_size=%%C"

        if exist "!target_file!" (
            rem 获取文件大小
            for %%I in ("!target_file!") do (
                set "file_size=%%~zI"
            )

            rem 检查文件大小是否匹配预期值
            if "!expected_size!"=="" (
                echo 期望大小为空，请检查输入文件格式.
            ) else (
                if "!file_size!"=="!expected_size!" (
                    rem 验证通过，不打印任何信息
                ) else (
                    set "size_mismatch=1"
                    set "size_mismatch_files=!size_mismatch_files! !target_file!"
                    echo !RED!文件 !target_file! !RESET!错误类型：大小不匹配!RESET!
                    echo 当前大小: !file_size! 字节, 目标大小: !expected_size! 字节
                )
            )
        ) else (
            set "missing_files=!missing_files! !target_file!"
        )
    )
)
echo.
:: 输出缺失文件和!RESET!错误类型：大小不匹配的信息
if defined missing_files (
    echo !RED!贺年卡所需文件缺失，请检查以下文件:!RESET!
    for %%F in (!missing_files!) do (
        echo !RED!文件 %%F !RESET!错误类型：文件缺失!RESET!
    )
    echo !GREEN!（可选）!RESET!若提示缺失flux_graffiti_v1文件可点击生成自动下载。贺年卡依赖于FluxAIO与可图AIO运行，请检查关联的包体是否安装完毕。可于https://hf-mirror.com/metercai/SimpleSDXL2/收集补充。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

if defined size_mismatch (
    echo !RED!文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。!RESET!
    echo 贺年卡依赖于FluxAIO与可图AIO运行，请检查关联的包体是否安装完毕。可于https://hf-mirror.com/metercai/SimpleSDXL2/收集补充。
    echo.
    set "all_passed=false"  :: 将all_passed设为false
)

:: 删除生成的文件
del files_and_sizes.txt

:: 如果所有文件都通过验证，打印“全部通过”
if "%all_passed%"=="true" (
    echo !GREEN!√贺年卡全部验证通过！!RESET!
    echo.
)

rem 获取脚本所在目录的上一级目录
pushd "%~dp0.."
set "root=%cd%\"
popd

echo Root directory: %root%
echo.
for /F %%i in ('dir /B "%root%*.zip" ^| find /C /V ""') do set "file_count=%%i"

set "loop_count=0"
for %%f in ("%root%*.zip") do (
    set /a loop_count+=1
    if !loop_count!==1 (
        echo ...
        echo ...
        echo ...
        echo ...
        echo ...
        echo ----------------------------------------------------------
        echo 准备解压%root%目录下的%file_count%个zip模型包，时间较长，需要耐心等待。安装完成后，请及时移除zip包，不必多次安装。安装过程中若中断，会导致解压出的文件残缺，需重新启动本程序再次解压覆盖。有任何疑问可进SimpleSDXL的QQ群求助：938075852      
    )
    echo 开始解压模型包：%%f
    echo.
    echo 按任意键继续。
    pause
    powershell -nologo -noprofile -command "Expand-Archive -Path '%%f' -DestinationPath '%root%' -Force"
)

echo All done.
echo.
echo 按任意键继续。
pause

endlocal
exit /b 0
