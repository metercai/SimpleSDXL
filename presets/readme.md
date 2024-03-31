## 什么是预置包?
<div align=center><img width="450" src="https://v2.token.tm/img/SimpleSDXL_preset_1.png"></div>

* **预置包**，顾名思义就是预先设置好的一组环境配置参数。从上面的基础架构图中可以看出，预置包是Fooocus/SimpleSDXL的重要组成部分。
* 预置包代表了图片生成的一种预设风格，一类预设主题，亦是一种预设的方法范例。它将变幻无穷的图片创作分门别类，通过全面的参数设定确保出图的基本效果和质量。
* 预置包可以大幅降低普通用户使用SDXL出图的使用门槛。 因为即使相同的提示词，在不同的模型和参数下，SDXL生成的图片也会千差万别，质量也都参差不齐。把所有影响出图的环境和参数打包在一起，形成预置包后，使用相同的预置包，可以保障相似的出图效果。
* 预置包是积累和沉淀出图经验的有力工具。出图新手和老手的差别就在于谁在环境配置和参数设置上的经验更丰富，能更快更好的达成预期目标。用好预置包工具，可以快速缩短新手和老手之间的距离。
* 预置包也是学习出图技巧的最佳案例。通过对预置包内容的学习和配置，可以加速对图片生成原理的理解，快速掌握图片生成的技巧。积累自己的预置包后，就可以大幅提升出图的效率和质量。

## 预置包里都包含哪些内容
- 生成图片所需的模型，比如主模型，精炼模型，LoRA局部或风格模型等。
- 运行模型的参数配置，比如采样器、调度器，引导系数、采样锐度等。
- 系统功能的默认设置，比如默认的出图数量、图片尺寸、是否支持蒙板上传等。
- 默认正反提示词示例，包括示范样张的正反提示词。
- 图片生成的流程设置，针对某种场景和目标的特殊执行流程。(计划中)
- 预置包所涉及的资源，包括模型的MUID及下载源，风格样式的定义等。
- 预置包的来源和说明，说明使用方法和来源作者的链接。

## SimpleSDXL比Fooocus在预置包上有哪些增强
- 预置包使用时可在线切换，不用每次重新启动。
- 预置包在加载时会主动检测模型文件，自动下载缺失的模型。
- 预置包可以从当前环境中抽取生成，可以快速定制自己的预置包。
- 预置包使用统一的模型资源ID，在流通和使用中保障了可用性和一致性。
- 预置包在界面上有显著的介绍和范例入口，方便新手学习和经验交流。

## 如何制作预置包？
- 首先在SimpleSDXL上把生成图片所需模型和参数都配置好，尝试生成图片，直到满意。
- 从界面底部打开"参数工具"选项，在工具箱中选择"生成预置包"，根据提示输入预置包名称后，系统会在presets目录自动生成名称对应的预置包json配置文件。
- 为了让更多人认识、了解和使用你创作的预置包，可以添加预置包的简介页。简介页是以html文件形式保存在`presets/html`目录下，也可以用URL形式加入配置文件。
- 配置文件和说明文件放到位后，系统在每次启动或预置包切换时，会更新顶部的预置包导航，如果存在简介页则会自动出现在右侧设置Tab顶部。
- 原创的预置包，可以推荐到社区，发布到公共的presets目录下，供大家分享学习和使用。
- 目前SimpleSDXL支持的预置包配置参数如下，后续可以根据使用场景进行添加，可以在Github Issues和QQ群:938075852 提出需求。

  ```
  default_prompt                                    # 默认提示词
  default_prompt_negative                           # 默认负向提示词
  default_styles                                    # 默认风格
  default_aspect_ratio                              # 默认宽高比
  defautl_output_format                             # 默认图片格式
  default_image_number                              # 默认出图数量
  default_max_image_number                          # 默认最大出图数量
  default_performance                               # 默认生成模式
  default_model                                     # 默认基础模型
  default_refiner                                   # 默认精炼模型
  default_refiner_switch                            # 默认精炼切入点
  default_loras                                     # 默认LoRA
  default_loras_min_weight                          # 默认LoRA的最小值
  default_loras_max_weight                          # 默认LoRA的最大值
  default_sampler                                   # 默认采样器
  default_scheduler                                 # 默认调度器
  default_cfg_scale                                 # 默认引导系数CFG
  default_sample_sharpness                          # 默认采样锐度
  default_adm_scaler_positive                       # 默认正向ADM引导系数
  default_adm_scaler_negative                       # 默认反向ADM引导系数
  default_adm_scaler_end                            # 默认ADM引导结束点
  default_cfg_tsnr                                  # 默认TSNR模拟CFG
  default_overwrite_step                            # 默认重写采样步数STEP
  default_overwrite_switch                          # 默认重写精炼器切入步数
  default_inpaint_engine                            # 默认重绘引擎版本
  default_mixing_image_prompt_and_vary_upscale      # 默认图像提示与增强变化混合
  default_mixing_image_prompt_and_inpaint           # 默认图像提示与重绘混合
  default_inpaint_mask_upload_checkbox              # 默认开启重绘蒙板
  default_freeu                                     # 默认FreeU参数: [b1, b2, s1, s2]
  default_seed                                      # 默认种子
  default_backfill_prompt                           # 默认浏览图片回填提示词
  default_translation_timing                        # 默认翻译器介入时点
  default_translation_methods                       # 默认翻译器模型
  checkpoint_downloads                              # 模型下载
  lora_downloads                                    # LoRA下载
  embeddings_downloads                              # 嵌入模型下载
  styles_definition                                 # 自定义风格
  reference                                         # 预置包说明链接
  

## 如何配置预置包简介？
- 预置包简介是对预置包能力和效果的介绍和引导。
- 简介页显示在操作界面右上角，高度为110px的窗口内，起到预置包的简要说明和入口引导作用。
- 建议单独制作预置包的整体说明页，包括预置包的使用场景，配置的模型，效果特点，使用技巧，样张示例等详细内容。
- 预置包简介有两种配置方式：
<br>**一，本地html文件** 在`presets/html`目录下，创建和预置包json配置文件同名的.inc.html文件作为该配置包的简介页，比如：`default.json` 对应的概要说明页为 `html/default.inc.html`。
<br>**二，网络URL** 在预置包json配置文件内新增属性项："reference"，指向外部的网络URL。预置包启用后，会通过网络调取该页面展示在右上侧窗口内。
<br><br>两种方式在调用时都会通过url参数方式传入两个环境变量，"__theme"表示当前的背景样式：dark(夜黑)或light(明亮)，"__lang"表示界面语言：cn(中文)或default(英文)。概要说明页可根据参数自行适配。在presets目录下提供的`html/default.inc.html`文件作为样本示范供参考。

## 如何获取制作好的预置包？
- SimpleSDXL会优选社区用户制作的预置包更新到presets目录下，上到导航条，供大家使用。
- 加入SimpleSDXL用户交流QQ群：938075852，通过群内获取。
<img width="250" src="https://v2.token.tm/img/qqgroup.jpg">
