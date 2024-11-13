# PuLID-Flux for ComfyUI
[PuLID-Flux](https://github.com/ToTheBeginning/PuLID) ComfyUI implementation (Alpha version)

![pulid_flux_einstein](examples/pulid_flux_einstein.png)

### :new: Version Updates
* V0.1.0: Working node with weight, start_at, end_at support (attn_mask not working)

## Notes
This project was heavily inspired by [cubiq/PuLID_ComfyUI](https://github.com/cubiq/PuLID_ComfyUI). It is just a prototype that uses some convenient model `hacks` for the encoder section. I wanted to test the model’s quality before reimplementing it in a more formal manner. For better results I recommend the `16bit` or `8bit GGUF` model version of Flux1-dev (the 8e5m2 returns blurry backgrounds). 
In the `examples` directory you'll find some basic workflows. 

## Supported Flux models:
##### 32bit/16bit (~22GB VRAM): [model](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors), [encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)
##### 8bit gguf (~12GB VRAM): [model](https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf), [encoder](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/blob/main/t5-v1_1-xxl-encoder-Q8_0.gguf)
##### 8 bit FP8 e5m2 (~12GB VRAM): [model](https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8-e5m2.safetensors), [encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors)
##### 8 bit FP8 e4m3fn (~12GB VRAM): [model](https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8-e4m3fn.safetensors), [encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors)
##### Clip and VAE (for all models): [clip](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors), [vae](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors)

For GGUF models you will need to install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) 


## Installation
 - Install this repo into `ComfyUI/custom_nodes`
```
git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git
```
- Install all the packages listed in the `requirements.txt` file into the Python environment where you run ComfyUI. I prefer not to use automatic installation scripts, as I dislike when scripts install software without my knowledge. :wink:
- You need one of the mentioned `Flux.1-dev` models. Download the model into `ComfyUI/models/unet`, clip and encoder into `ComfyUI/models/clip`, VAE into `ComfyUI/models/vae`. 
- [PuLID Flux pre-trained model](https://huggingface.co/guozinan/PuLID/blob/main/pulid_flux_v0.9.0.safetensors?download=true) goes in `ComfyUI/models/pulid/`.
- The EVA CLIP is EVA02-CLIP-L-14-336, should be downloaded automatically (will be located in the huggingface directory). If for some reason the auto-download fails (and you get face_analysis.py, **init assert 'detection' in self.models exception**), download this [EVA-CLIP](https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt?download=true) model manually, put the file to your `ComfyUI/models/clip`and restart ComfyUI.

- `facexlib` dependency needs to be installed, the models are downloaded at first use. 
- Finally you need InsightFace with [AntelopeV2](https://huggingface.co/MonsterMMORPG/tools/tree/main), the unzipped models should be placed in `ComfyUI/models/insightface/models/antelopev2`.

## Known issues
- ApplyPulidFlux doesn't work on HW with CUDA compute < v8.0, (when Flux FP8 it needs bfloat16).
- When the ApplyPulidFlux node is disconnected after first run, the Flux model is still influenced by the node. 
- ApplyPulidFlux attn_mask is not working (in progress).

## Credits

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

[PuLID for Flux](https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md) - tuning-free ID customization solution for FLUX.1-dev

cubiq [PuLID_ComfyUI](https://github.com/cubiq/PuLID_ComfyUI) - PuLID ComfyUI native implementation (Thanks for the awesome work what you do Matteo :wink: ).
