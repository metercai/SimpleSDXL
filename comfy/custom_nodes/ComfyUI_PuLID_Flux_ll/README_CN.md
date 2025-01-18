[English](README.md)

- 解决插件 [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux) 存在的模型污染问题。
- 支持使用[TeaCache](https://github.com/ali-vilab/TeaCache)加速（`TeaCache`加速需要配合[ComfyUI_Patches_ll](https://github.com/lldacing/ComfyUI_Patches_ll)使用）。
- 支持使用[Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)加速, Comfy-WaveSpeed在[提交记录-36ba3c8](https://github.com/chengzeyi/Comfy-WaveSpeed/commit/36ba3c8b74735d4521828507a4bf323df1a9a9d0)中提供支持。

在安装此插件之前，必须卸载或禁用`ComfyUI-PuLID-Flux`和其他PuLID Flux节点, 因为由于某些原因，我使用了同样的节点名`ApplyPulidFlux`

## 预览 (图片含工作流)
![save api extended](examples/PuLID_with_speedup.png)
![save api extended](examples/PuLID_with_attn_mask.png)

## 安装

- 手动
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.git
    cd ComfyUI_PuLID_Flux_ll
    pip install -r requirements.txt
    # 重启 ComfyUI
```
## 模型
查看[ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)

## 节点
- PulidFluxModelLoader
  - 同 [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- PulidFluxInsightFaceLoader
  - 同 [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- PulidFluxEvaClipLoader
  - 同 [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- ApplyPulidFlux
  - 解决了原插件中模型污染的问题
  - `attn_mask`~~可能不能正确工作， 因为我不知道如何实现它， 尝试了多种方式效果都未能达到预期~~，可以正常工作了。
  - 使用 [TeaCache](https://github.com/ali-vilab/TeaCache)加速, 必须加在[`FluxForwardOverrider` and `ApplyTeaCachePatch`](https://github.com/lldacing/ComfyUI_Patches_ll)之前.
  - 使用 [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)加速, 必须加在[`ApplyFBCacheOnModel`](https://github.com/lldacing/ComfyUI_Patches_ll)之前.
- FixPulidFluxPatch (已弃用)
  - 如果想使用 [TeaCache](https://github.com/ali-vilab/TeaCache)加速, 必须~~加在 `ApplyPulidFlux` 节点之后, 并~~在后面连接节点 [`FluxForwardOverrider` and `ApplyTeaCachePatch`](https://github.com/lldacing/ComfyUI_Patches_ll).

## 感谢

[ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID)

[ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)

[TeaCache](https://github.com/ali-vilab/TeaCache)

[Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)