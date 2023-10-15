#!/bin/sh

# webui.py
WEBUI="webui.py"
sed -i "s|label='Preview',|label='预览',|" $WEBUI
sed -i "s|label='Gallery',|label='图集',|" $WEBUI
sed -i "s|placeholder=\"Type prompt here.\",|placeholder=\"输入提示词。\",|" $WEBUI
sed -i "s|label=\"Generate\", value=\"Generate\",|label=\"生成\", value=\"生成\",|" $WEBUI
sed -i "s|label=\"Skip\", value=\"Skip\",|label=\"跳过\", value=\"跳过\",|" $WEBUI
sed -i "s|label=\"Stop\", value=\"Stop\",|label=\"终止\", value=\"终止\",|" $WEBUI
sed -i "s|label='Upscale or Variation'|label='强化与变换'|" $WEBUI
sed -i "s|label='Drag above image to here',|label='将图片拖入这里',|" $WEBUI
sed -i "s|label='Upscale or Variation:',|label='强化(Upscale)与变换(Vary)：',|" $WEBUI
sed -i "s|U0001F4D4 Document<|U0001F4D4 参考文档<|" $WEBUI
sed -i "s|label='Image Prompt'|label='依图生图'|" $WEBUI
sed -i "s|label='Stop At',|label='停在',|" $WEBUI
sed -i "s|label='Weight',|label='权重',|" $WEBUI
sed -i "s|label='Type',|label='类型',|" $WEBUI
sed -i "s|label='Input Image',|label='输入图片',|" $WEBUI
sed -i "s|label='Advanced',|label='高级选项',|" $WEBUI
sed -i "s|label='Inpaint or Outpaint (beta)'|label='修补与融合（测试）'|" $WEBUI
sed -i "s|'Outpaint Expansion (<a|'扩图 (<a|" $WEBUI
sed -i "s|label='Setting'|label='设置'|" $WEBUI
sed -i "s|label='Performance',|label='性能（Speed速度，Quality质量）',|" $WEBUI
sed -i "s|label='Aspect Ratios',|label='宽高比',|" $WEBUI
sed -i "s|info='width × height'|info='宽 × 高'|" $WEBUI
sed -i "s|label='Image Number',|label='出图数量',|" $WEBUI
sed -i "s|label='Negative Prompt',|label='反向提示词',|" $WEBUI
sed -i "s|placeholder=\"Type prompt here.\",|placeholder=\"输入提示词。\",|" $WEBUI
sed -i "s|info='Describing what you do not want to see.',|info='描述你不想看到的内容',|" $WEBUI
sed -i "s|label='Random',|label='随机种子',|" $WEBUI
sed -i "s|label='Seed',|label='种子',|" $WEBUI
sed -i "s|label='Style'|label='风格'|" $WEBUI
sed -i "s|label='Model'|label='模型'|" $WEBUI
sed -i "s|label='Image Style'|label='图片风格'|" $WEBUI
sed -i "s|label='Base Model (SDXL only)',|label='SDXL基础模型',|" $WEBUI
sed -i "s|label='Refiner (SDXL or SD 1.5)',|label='SDXL精炼模型',|" $WEBUI
sed -i "s|label='Refresh',|label='刷新',|" $WEBUI
sed -i "s|504 Refresh All Files',|504 全部刷新',|" $WEBUI
sed -i "s|label='Advanced'|label='高级'|" $WEBUI
sed -i "s|label='Sampling Sharpness',|label='采样的清晰度',|" $WEBUI
sed -i "s|info='Higher value means image and texture are sharper.'|info='越高图像和纹理越清晰'|" $WEBUI
sed -i "s|label='Guidance Scale',|label='提示词引导系数',|" $WEBUI
sed -i "s|info='Higher value means style is cleaner, vivider, and more artistic.'|info='提示词作用的强度，越高风格越干净、生动、更具艺术感'|" $WEBUI
sed -i "s|label='Developer Debug Mode',|label='开发者模式',|" $WEBUI
sed -i "s|label='Developer Debug Tools'|label='开发者调试工具'|" $WEBUI
sed -i "s|label='Positive ADM Guidance Scaler',|label='正向ADM指导缩放',|" $WEBUI
sed -i "s|info='The scaler multiplied to positive ADM (use 1.0 to disable). '|info='用于乘以正向ADM的缩放器 (使用1.0以禁用). '|" $WEBUI
sed -i "s|label='Negative ADM Guidance Scaler',|label='负向ADM指导缩放',|" $WEBUI
sed -i "s|info='The scaler multiplied to negative ADM (use 1.0 to disable). '|info='用于乘以负向ADM的缩放器 (使用1.0以禁用). '|" $WEBUI
sed -i "s|label='ADM Guidance End At Step',|label='ADM指导结束步长',|" $WEBUI
sed -i "s|info='When to end the guidance from positive/negative ADM. '|info='从正向/负向ADM结束指导的时间'|" $WEBUI
sed -i "s|label='Refiner swap method',|label='精炼交换方式',|" $WEBUI
sed -i "s|label='CFG Mimicking from TSNR',|label='CFG模拟TSNR',|" $WEBUI
sed -i "s|info='Enabling Fooocus\\'s implementation of CFG mimicking for TSNR '|info='启用Fooocus的CFG模拟TSNR实现'|" $WEBUI
sed -i "s|'(effective when real CFG > mimicked CFG).'|'（实际生效需满足真实CFG大于模拟CFG的条件）'|" $WEBUI
sed -i "s|label='Sampler',|label='采样器',|" $WEBUI
sed -i "s|info='Only effective in non-inpaint mode.'|info='仅在非修复模式下有效'|" $WEBUI
sed -i "s|label='Scheduler',|label='调度器',|" $WEBUI
sed -i "s|info='Scheduler of Sampler.'|info='采样器调度程序'|" $WEBUI
sed -i "s|label='Forced Overwrite of Sampling Step',|label='强制覆盖采样步长',|" $WEBUI
sed -i "s|info='Set as -1 to disable. For developer debugging.'|info='设为-1以禁用。用于开发者调试'|" $WEBUI
sed -i "s|label='Forced Overwrite of Refiner Switch Step',|label='强制覆盖优化开关步长',|" $WEBUI
sed -i "s|label='Forced Overwrite of Generating Width',|label='强制覆盖生成宽度',|" $WEBUI
sed -i "s|label='Forced Overwrite of Generating Heigh',|label='强制覆盖生成高度',|" $WEBUI
sed -i "s|label='Forced Overwrite of Denoising Strength of \"Vary\"',|label='强制覆盖变换的去噪强度',|" $WEBUI
sed -i "s|info='Set as negative number to disable. For developer debugging.'|info='设为负数以禁用。用于开发者调试'|" $WEBUI
sed -i "s|label='Forced Overwrite of Denoising Strength of \"Upscale\",|label='强制覆盖强化的去噪强度',|" $WEBUI
sed -i "s|label='Inpaint Engine',|label='修补引擎',|" $WEBUI
sed -i "s|info='Version of Fooocus inpaint model'|info='Fooocus修补引擎版本'|" $WEBUI
sed -i "s|label='Control Debug'|label='调试控制'|" $WEBUI
sed -i "s|label='Debug Preprocessor of ControlNets',|label='ControlNets的调试预处理器',|" $WEBUI
sed -i "s|label='Mixing Image Prompt and Vary/Upscale',|label='混合图像提示和变换/增强',|" $WEBUI
sed -i "s|label='Mixing Image Prompt and Inpaint',|label='混合图像提示和修复',|" $WEBUI
sed -i "s|label='Softness of ControlNet',|label='ControlNet柔化',|" $WEBUI
sed -i "s|info='Similar to the Control Mode in A1111 (use 0.0 to disable). '|info='类似于A1111中的控制模式（使用0.0禁用）'|" $WEBUI
sed -i "s|label='Canny'|label='锐化Canny',|" $WEBUI
sed -i "s|label='Canny Low Threshold',|label='锐化的低阈值',|" $WEBUI
sed -i "s|label='Canny High Threshold',|label='锐化的高阈值',|" $WEBUI

# sdxl_styles.py

