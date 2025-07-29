import torch
import torch.nn as nn
from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
from diffusers.utils import is_accelerate_available
from contextlib import nullcontext

if is_accelerate_available():
    import accelerate
    from accelerate import init_empty_weights

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_with_gguf_linear(model, compute_dtype, state_dict, prefix="", modules_to_not_convert=[], patches=None):
    def _should_convert_to_gguf(state_dict, prefix):
        weight_key = prefix + "weight"
        return weight_key in state_dict and isinstance(state_dict[weight_key], GGUFParameter)

    has_children = list(model.children())
    if not has_children:
        return

    for name, module in model.named_children():
        module_prefix = prefix + name + "."
        _replace_with_gguf_linear(module, compute_dtype, state_dict, module_prefix, modules_to_not_convert, patches)

        if (
            isinstance(module, nn.Linear)
            and _should_convert_to_gguf(state_dict, module_prefix)
            and name not in modules_to_not_convert
        ):
            in_features = state_dict[module_prefix + "weight"].shape[1]
            out_features = state_dict[module_prefix + "weight"].shape[0]

            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                model._modules[name] = GGUFLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype
                )
            #set_lora_params(model._modules[name], patches, module_prefix)
            model._modules[name].source_cls = type(module)
            # Force requires_grad to False to avoid unexpected errors
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix=""):
    # Recursively set lora_diffs and lora_strengths for all GGUFLinear layers
    for name, child in module.named_children():
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix)
    if isinstance(module, GGUFLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue  # For now skip LoRA for head layers
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            module.lora = (lora_diffs, lora_strengths)


class GGUFLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora = None

    def forward(self, inputs):
        weight = self.dequantize_without_compile()
        weight = weight.to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None

        if hasattr(self, "lora") and self.lora is not None:
            weight = self.apply_lora(weight).to(self.compute_dtype)

        output = torch.nn.functional.linear(inputs, weight, bias)
        return output
    
    @torch.compiler.disable()
    def dequantize_without_compile(self):
        return dequantize_gguf_tensor(self.weight)

    @torch.compiler.disable()
    def apply_lora(self, weight):
        for lora_diff, lora_strength in zip(self.lora[0], self.lora[1]):
            patch_diff = torch.mm(
                lora_diff[0].flatten(start_dim=1).to(weight.device),
                lora_diff[1].flatten(start_dim=1).to(weight.device)
            ).reshape(weight.shape)
            alpha = lora_diff[2] / lora_diff[1].shape[0] if lora_diff[2] is not None else 1.0
            scale = lora_strength * alpha
            weight = weight.add(patch_diff, alpha=scale)
        return weight