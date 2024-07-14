![image](./examples/workflow.png)

ComfyUI上Kolors的实现

参考自 https://github.com/kijai/ComfyUI-KwaiKolorsWrapper

使用ComfyUI原生采样

工作流在examples/workflow.png中获取


unet模型放置在 models/unet/ 文件夹下
下载地址:
https://huggingface.co/Kwai-Kolors/Kolors/blob/main/unet/diffusion_pytorch_model.fp16.safetensors


chatglm3放置在 models/LLM/ 文件夹下
下载地址:
https://huggingface.co/Kijai/ChatGLM3-safetensors/blob/main/chatglm3-fp16.safetensors


Implementation of Kolors on ComfyUI

Reference from https://github.com/kijai/ComfyUI-KwaiKolorsWrapper

Using ComfyUI Native Sampling

The workflow is obtained in examples/workflow.png


The unet model is placed in the models/unet/ folder
Download link:
https://huggingface.co/Kwai-Kolors/Kolors/blob/main/unet/diffusion_pytorch_model.fp16.safetensors


The chatglm3 is placed in the models/LLM/ folder
Download link:
https://huggingface.co/Kijai/ChatGLM3-safetensors/tree/main




## FAQ
Error occurred when executing MZ_ChatGLM3Loader: 'ChatGLMModel' object has no attribute 'transformer'
+ 检查ChatGLM3Loader节点选择的模型是否已经正确下载

RuntimeError: Only Tensors of floating point dtype can require gradients
+ 尝试使用fp16版本的模型: https://huggingface.co/Kijai/ChatGLM3-safetensors/blob/main/chatglm3-fp16.safetensors

module 'comfy.model_detection' has no attribute 'unet_prefix_from_state_dict'
+ 更新ComfyUI到最新版本



## Star History

<a href="https://star-history.com/#MinusZoneAI/ComfyUI-Kolors-MZ&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-Kolors-MZ&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-Kolors-MZ&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-Kolors-MZ&type=Date" />
 </picture>
</a>


## Contact
- 绿泡泡: minrszone
- Bilibili: [minus_zone](https://space.bilibili.com/5950992)
- 小红书: [MinusZoneAI](https://www.xiaohongshu.com/user/profile/5f072e990000000001005472)
- 爱发电: [MinusZoneAI](https://afdian.net/@MinusZoneAI)
