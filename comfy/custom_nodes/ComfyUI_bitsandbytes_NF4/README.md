Now on the [manager](https://github.com/ltdrdata/ComfyUI-Manager) for easy installation.

A quickly written custom node that uses code from [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) to support the [nf4 flux dev checkpoint](https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/blob/main/flux1-dev-bnb-nf4.safetensors) and [nf4 flux schnell checkpoint](https://huggingface.co/silveroxides/flux1-nf4-weights/blob/main/flux1-schnell-bnb-nf4.safetensors).

Requires installing bitsandbytes.

Make sure your ComfyUI is updated.

The node is: CheckpointLoaderNF4, just plug it in your flux workflow instead of the regular one.
