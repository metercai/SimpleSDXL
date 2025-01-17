[中文文档](README_CN.md)

- Solved [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux) model pollution problem.
- Supported use with `TeaCache` (Need use with [ComfyUI_Patches_ll](https://github.com/lldacing/ComfyUI_Patches_ll)).
- Supported use with [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed), supported by `Comfy-WaveSpeed` in [commit-36ba3c8](https://github.com/chengzeyi/Comfy-WaveSpeed/commit/36ba3c8b74735d4521828507a4bf323df1a9a9d0).

Must uninstall or disable `ComfyUI-PuLID-Flux` and other PuLID-Flux nodes before install this plugin. Due to certain reasons, I used the same node's name `ApplyPulidFlux`.


## Preview (Image with WorkFlow)
![save api extended](examples/PuLID_with_speedup.png)
![save api extended](examples/PuLID_with_attn_mask.png)

## Install

- Manual
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.git
    cd ComfyUI_PuLID_Flux_ll
    pip install -r requirements.txt
    # restart ComfyUI
```

## Model
Please see [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)


## Nodes
- PulidFluxModelLoader
  - See [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- PulidFluxInsightFaceLoader
  - See [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- PulidFluxEvaClipLoader
  - See [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- ApplyPulidFlux
  - Solved the model pollution problem of the original plugin ComfyUI-PuLID-Flux
  - `attn_mask` ~~may not work correctly (I have no idea how to apply it, I have tried multiple methods and the results have been not satisfactory)~~ works now.
  - If you want use with [TeaCache](https://github.com/ali-vilab/TeaCache), must put it before node [`FluxForwardOverrider` and `ApplyTeaCachePatch`](https://github.com/lldacing/ComfyUI_Patches_ll).
  - If you want use with [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed), must put it before node `ApplyFBCacheOnModel`.
- FixPulidFluxPatch (Deprecated)
  - If you want use with [TeaCache](https://github.com/ali-vilab/TeaCache), must ~~link it after node `ApplyPulidFlux`, and~~ link node [`FluxForwardOverrider` and `ApplyTeaCachePatch`](https://github.com/lldacing/ComfyUI_Patches_ll) after it.

## Thanks

[ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID)

[ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)

[TeaCache](https://github.com/ali-vilab/TeaCache)

[Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)