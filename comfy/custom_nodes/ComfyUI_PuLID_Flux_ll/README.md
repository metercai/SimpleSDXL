[中文文档](README_CN.md)

- Solved [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux) model pollution problem.
- Supported use with `TeaCache` (Need use with [ComfyUI_Patches_ll](https://github.com/lldacing/ComfyUI_Patches_ll)).
- Supported use with [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed), supported by `Comfy-WaveSpeed` in [commit-36ba3c8](https://github.com/chengzeyi/Comfy-WaveSpeed/commit/36ba3c8b74735d4521828507a4bf323df1a9a9d0).
- Supported simple use with `First Block Cache` (Can use with [ComfyUI_Patches_ll](https://github.com/lldacing/ComfyUI_Patches_ll)).

Must uninstall or disable `ComfyUI-PuLID-Flux` and other PuLID-Flux nodes before install this plugin. Due to certain reasons, I used the same node's name `ApplyPulidFlux`.

Need upgrade ComfyUI Version>=0.3.7

## Update logs
### 2025.02.19
- Fix: when selecting a face from multiple faces as a reference, embeddings and alignment features maybe not from the same face.
### 2025.02.18
- Supported selecting a face from multiple faces as a reference. [Example workflow](examples/PuLID_select_ref_face.png).
### 2025.01.27
- Changed the model path of facexlib to `ComfyUI/models/facexlib/`.
- When automatically downloading, modify the path of Antelope v2 model to `ComfyUI/models/insightface/models/antelopev2/`.
- Changed the model path of EVA_CLIP_L_14_336 to `ComfyUI/models/clip/`.

## Preview (Image with WorkFlow)
![save api extended](examples/PuLID_with_speedup.png)
![save api extended](examples/PuLID_with_attn_mask.png)
![save api extended](examples/PuLID_select_ref_face.png)

## Install

- Manual
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.git
    cd ComfyUI_PuLID_Flux_ll
    pip install -r requirements.txt
    # restart ComfyUI
```

Tips:

- If you use `ComfyUI_windows_portable` and encounter the following error, please see https://github.com/deepinsight/insightface/issues/2576
```
insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.cpp(36): fatal error C1083: 无法打开包括文件: “Python.h”: No such file or directory
      error: command 'd:\\installed\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.42.34433\\bin\\HostX86\\x64\\cl.exe' failed with exit code 2
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for insightface
Failed to build insightface
```

## Models
### Available Flux models
- 32bit/16bit (~22GB VRAM): [model](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors), [encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)
- 8bit gguf (~12GB VRAM): [model](https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf), [encoder](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/blob/main/t5-v1_1-xxl-encoder-Q8_0.gguf)
- 8 bit FP8 e5m2 (~12GB VRAM): [model](https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8-e5m2.safetensors), [encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors)
- 8 bit FP8 e4m3fn (~12GB VRAM): [model](https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8-e4m3fn.safetensors), [encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors)
- Clip and VAE (for all models): [clip](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors), [vae](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors)

#### For GGUF models you will need to install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) 

### PuLID models
- Download [PuLID-Flux](https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors?download=true) => `ComfyUI/models/pulid/`.
- (Support auto-download) Download [EVA02-CLIP-L-14-336](https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt?download=true) => `ComfyUI/models/clip/`.
- (Support auto-download) Download all models like `*.onnx` from [AntelopeV2](https://huggingface.co/MonsterMMORPG/tools/tree/main) => `ComfyUI/models/insightface/models/antelopev2/`.
- (Support auto-download) Download [parsing_bisenet](https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth), [parsing_parsenet](https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth) and [Resnet50](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth) => `ComfyUI/models/facexlib/`.

## Nodes
- PulidFluxModelLoader
  - See chapter [PuLID models](#pulid-models)
- PulidFluxInsightFaceLoader
  - See chapter [PuLID models](#pulid-models)
- PulidFluxEvaClipLoader
  - See chapter [PuLID models](#pulid-models)
- ApplyPulidFlux
  - Solved the model pollution problem of the original plugin ComfyUI-PuLID-Flux
  - `attn_mask` ~~may not work correctly (I have no idea how to apply it, I have tried multiple methods and the results have been not satisfactory)~~ works now.
  - If you want use with [TeaCache](https://github.com/ali-vilab/TeaCache), must put it before node [`FluxForwardOverrider` and `ApplyTeaCachePatch`](https://github.com/lldacing/ComfyUI_Patches_ll).
  - If you want use with [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed), must put it before node `ApplyFBCacheOnModel`.
- FixPulidFluxPatch (Deprecated)
  - If you want use with [TeaCache](https://github.com/ali-vilab/TeaCache), must link it after node `ApplyPulidFlux`, and link node [`FluxForwardOverrider` and `ApplyTeaCachePatch`](https://github.com/lldacing/ComfyUI_Patches_ll) after it.
- PulidFluxOptions
  - `input_faces_order` - Sorting rule for detected bboxes.
    - `left-right`: Sort the left boundary of bbox by column from left to right.
    - `right-left`: Reverse order of `left-right` (Sort the left boundary of bbox by column from right to left).
    - `top-bottom`: Sort the top boundary of bbox by row from top to bottom.
    - `bottom-top`: Reverse order of `top-bottom` (Sort the top boundary of bbox by row from bottom to top).
    - `small-large`: Sort the area of bbox from small to large.
    - `large-small`: Sort the area of bbox from large to small.
  - `input_faces_index` - The target index of the sorted bboxes.
  - `input_faces_align_mode` - Choose the detection method for aligning facial features. 
    - `0`: Old version method, When there is a face in an image, the selected facial embedding amount and alignment features maybe not consistent. 
    - `1`: Keep the selected facial embedding amount and alignment features consistent.
    - There is a slight difference between the two mode, with the `align_face` value of `1` resulting smaller area than the `embed_face` value of `0`.
- PulidFluxFaceDetector
  - Can check the facial features applied in `ApplyPulidFlux`.
  - When `input_faces_align_mode = 0`, the `embed_face` and `align_face` should be the same face, but they are generated by different detectors, and the number detected may be not consistent, so they may be not the same face.
  - When `input_faces_align_mode = 1`, the `embed_face` and `align_face` are always the same face, they are generated by same detectors.
  - `face_bbox_image` - Draw the detected facial bounding box (the result of the `embed_face`'s detector).

## Thanks

[ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID)

[ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)

[TeaCache](https://github.com/ali-vilab/TeaCache)

[Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)
