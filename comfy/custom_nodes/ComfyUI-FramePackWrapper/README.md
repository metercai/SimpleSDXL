# ComfyUI Wrapper for [FramePack by lllyasviel](https://lllyasviel.github.io/frame_pack_gitpage/)

# WORK IN PROGRESS

Mostly working, took some liberties to make it run faster.

Uses all the native models for text encoders, VAE and sigclip:

https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files

https://huggingface.co/Comfy-Org/sigclip_vision_384/tree/main

And the transformer model itself is either autodownloaded from here:

https://huggingface.co/lllyasviel/FramePackI2V_HY/tree/main

to `ComfyUI\models\diffusers\lllyasviel\FramePackI2V_HY`

Or from single file, in `ComfyUI\models\diffusion_models`:

https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/FramePackI2V_HY_fp8_e4m3fn.safetensors
https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/FramePackI2V_HY_bf16.safetensors
