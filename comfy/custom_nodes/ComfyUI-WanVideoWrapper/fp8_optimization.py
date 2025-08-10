import torch
import torch.nn as nn
from .utils import log

#based on ComfyUI's and MinusZoneAI's fp8_linear optimization
def fp8_linear_forward(cls, base_dtype, input):
    weight_dtype = cls.weight.dtype
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            input_shape = input.shape
            
            scale_weight = getattr(cls, 'scale_weight', None)
            if scale_weight is None:
                scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
            else:
                scale_weight = scale_weight.to(input.device)
            
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            input = torch.clamp(input, min=-448, max=448, out=input)
            inn = input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous() #always e4m3fn because e5m2 * e5m2 is not supported

            bias = cls.bias.to(base_dtype) if cls.bias is not None else None

            o = torch._scaled_mm(inn, cls.weight.t(), out_dtype=base_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)

            return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
        else:
            return cls.original_forward(input.to(base_dtype))
    else:
        return cls.original_forward(input)


@torch.compiler.disable()
def apply_lora(weight, lora, step=None):
    for lora_diff, lora_strength in zip(lora[0], lora[1]):
        if isinstance(lora_strength, list):
            lora_strength = lora_strength[step]
            if lora_strength == 0.0:
                continue
        elif lora_strength == 0.0:
            continue
        patch_diff = torch.mm(
            lora_diff[0].flatten(start_dim=1).to(weight.device),
            lora_diff[1].flatten(start_dim=1).to(weight.device)
        ).reshape(weight.shape)
        alpha = lora_diff[2] / lora_diff[1].shape[0] if lora_diff[2] is not None else 1.0
        scale = lora_strength * alpha
        weight = weight.add(patch_diff, alpha=scale)
    return weight


def linear_with_lora_and_scale_forward(cls, input):
    # Handles both scaled and unscaled, with or without LoRA
    has_scale = hasattr(cls, "scale_weight")
    weight = cls.weight.to(input.dtype)
    bias = cls.bias.to(input.dtype) if cls.bias is not None else None

    if has_scale:
        scale_weight = cls.scale_weight.to(input.device)
        if weight.numel() < input.numel():
            weight = weight * scale_weight
        else:
            input = input * scale_weight

    lora = getattr(cls, "lora", None)
    if lora is not None:
        weight = apply_lora(weight, lora, cls.step).to(input.dtype)

    return torch.nn.functional.linear(input, weight, bias)

def convert_fp8_linear(module, base_dtype, params_to_keep={}, scale_weight_keys=None):
    log.info("FP8 matmul enabled")
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(submodule, nn.Linear):
                if scale_weight_keys is not None:
                    scale_key = f"{name}.scale_weight"
                    if scale_key in scale_weight_keys:
                        print("Setting scale_weight for", name)
                        setattr(submodule, "scale_weight", scale_weight_keys[scale_key])
                original_forward = submodule.forward
                setattr(submodule, "original_forward", original_forward)
                setattr(submodule, "forward", lambda input, m=submodule: fp8_linear_forward(m, base_dtype, input))
 
def convert_linear_with_lora_and_scale(module, scale_weight_keys=None, patches=None, params_to_keep={}):
    log.info("Patching Linear layers...")
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            # Set scale_weight if present
            if scale_weight_keys is not None:
                scale_key = f"{name}.scale_weight"
                if scale_key in scale_weight_keys:
                    setattr(submodule, "scale_weight", scale_weight_keys[scale_key])

            # Set LoRA if present
            if hasattr(submodule, "lora"):
                #print(f"removing old LoRA in {name}" )
                delattr(submodule, "lora")
            if patches is not None:
                patch_key1 = f"diffusion_model.{name}.weight"
                patch_key_compiled = f"diffusion_model.{name.replace('_orig_mod.', '')}.weight"
                patch = patches.get(patch_key1, []) or patches.get(patch_key_compiled, [])
                if len(patch) != 0:
                    lora_diffs = []
                    for p in patch:
                        lora_obj = p[1]
                        if "head" in name:
                            continue  # For now skip LoRA for head layers
                        elif hasattr(lora_obj, "weights"):
                            lora_diffs.append(lora_obj.weights)
                        elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                            lora_diffs.append(lora_obj[1])
                        else:
                            continue
                    lora_strengths = [p[0] for p in patch]
                    lora = (lora_diffs, lora_strengths)
                    setattr(submodule, "lora", lora)
                    #print(f"Added LoRA to {name} with {len(lora_diffs)} diffs and strengths {lora_strengths}")

            # Set forward if Linear and has either scale or lora
            if isinstance(submodule, nn.Linear):
                has_scale = hasattr(submodule, "scale_weight")
                has_lora = hasattr(submodule, "lora")
                if not hasattr(submodule, "original_forward"):
                    setattr(submodule, "original_forward", submodule.forward)
                if has_scale or has_lora:
                    setattr(submodule, "forward", lambda input, m=submodule: linear_with_lora_and_scale_forward(m, input))
                    setattr(submodule, "step", 0)  # Initialize step for LoRA scheduling

def remove_lora_from_module(module):
    unloaded = False
    for name, submodule in module.named_modules():
        if hasattr(submodule, "lora"):
            if not unloaded:
                log.info("Unloading all LoRAs")
                unloaded = True
            delattr(submodule, "lora")
