import copy
from typing import Optional

import PIL
import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import comfy.samplers
import nodes
from typing import Optional

class SeamlessTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tiling": (["enable", "x_only", "y_only", "disable"],),
                "copy_model": (["Make a copy", "Modify in place"],),
            },
        }

    CATEGORY = "SeamlessTile"

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"

    def run(self, model, copy_model, tiling):
        if copy_model == "Modify in place":
            model_copy = model
        else:
            model_copy = copy.deepcopy(model)
            
        if tiling == "enable":
            make_circular_asymm(model_copy.model, True, True)
        elif tiling == "x_only":
            make_circular_asymm(model_copy.model, True, False)
        elif tiling == "y_only":
            make_circular_asymm(model_copy.model, False, True)
        else:
            make_circular_asymm(model_copy.model, False, False)
        return (model_copy,)


# asymmetric tiling from https://github.com/tjm35/asymmetric-tiling-sd-webui/blob/main/scripts/asymmetric_tiling.py
def make_circular_asymm(model, tileX: bool, tileY: bool):
    for layer in [
        layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)
    ]:
        layer.padding_modeX = 'circular' if tileX else 'constant'
        layer.padding_modeY = 'circular' if tileY else 'constant'
        layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
        layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
        layer._conv_forward = __replacementConv2DConvForward.__get__(layer, Conv2d)
    return model


def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


class CircularVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tiling": (["enable", "x_only", "y_only", "disable"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "SeamlessTile"

    def decode(self, samples, vae, tiling):
        vae_copy = copy.deepcopy(vae)
        
        if tiling == "enable":
            make_circular_asymm(vae_copy.first_stage_model, True, True)
        elif tiling == "x_only":
            make_circular_asymm(vae_copy.first_stage_model, True, False)
        elif tiling == "y_only":
            make_circular_asymm(vae_copy.first_stage_model, False, True)
        else:
            make_circular_asymm(vae_copy.first_stage_model, False, False)
        
        result = (vae_copy.decode(samples["samples"]),)
        return result


class MakeCircularVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "tiling": (["enable", "x_only", "y_only", "disable"],),
                "copy_vae": (["Make a copy", "Modify in place"],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "run"
    CATEGORY = "SeamlessTile"

    def run(self, vae, tiling, copy_vae):
        if copy_vae == "Modify in place":
            vae_copy = vae
        else:
            vae_copy = copy.deepcopy(vae)
        
        if tiling == "enable":
            make_circular_asymm(vae_copy.first_stage_model, True, True)
        elif tiling == "x_only":
            make_circular_asymm(vae_copy.first_stage_model, True, False)
        elif tiling == "y_only":
            make_circular_asymm(vae_copy.first_stage_model, False, True)
        else:
            make_circular_asymm(vae_copy.first_stage_model, False, False)
        
        return (vae_copy,)


class OffsetImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "x_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
                "y_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "SeamlessTile"

    def run(self, pixels, x_percent, y_percent):
        n, y, x, c = pixels.size()
        y = round(y * y_percent / 100)
        x = round(x * x_percent / 100)
        return (pixels.roll((y, x), (1, 2)),)

class TiledKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"model": ("MODEL", ),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "tiling": (["enable", "x_only", "y_only", "disable"],),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "positive": ("CONDITIONING", ),
                 "negative": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "SeamlessTile"
    def apply_circular(self, model, enable):
        for layer in [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def sample(self, model, seed, tiling, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        self.apply_circular(model.model, tiling in ["enable", "x_only", "y_only"])
        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)



class Asymmetric_Tiled_KSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"model": ("MODEL", ),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "tileX": ("INT", {"default": 1, "min": 0, "max": 1}),
                 "tileY": ("INT", {"default": 1, "min": 0, "max": 1}),
                 "startStep": ("INT", {"default": 0, "min": 0, "max": 10000}),
                 "stopStep": ("INT", {"default": -1, "min": -1, "max": 10000}),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "positive": ("CONDITIONING", ),
                 "negative": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "SeamlessTile"

    def apply_asymmetric_tiling(self, model, tileX, tileY):
        for layer in [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_modeX = 'circular' if tileX else 'constant'
            layer.padding_modeY = 'circular' if tileY else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            print(layer.paddingX, layer.paddingY)

    def __hijackConv2DMethods(self, model, tileX: bool, tileY: bool, startStep: int, stopStep: int):
        for layer in [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]:
            layer.padding_modeX = 'circular' if tileX else 'constant'
            layer.padding_modeY = 'circular' if tileY else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            layer.paddingStartStep = startStep
            layer.paddingStopStep = stopStep
            
            def make_bound_method(method, current_layer):
                def bound_method(self, *args, **kwargs):  # Add 'self' here
                    return method(current_layer, *args, **kwargs)
                return bound_method
                
            bound_method = make_bound_method(self.__replacementConv2DConvForward, layer)
            layer._conv_forward = bound_method.__get__(layer, type(layer))

    def __replacementConv2DConvForward(self, layer, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        step = nodes.common_ksampler.current_step  # Assuming there's a way to get the current step
        if ((layer.paddingStartStep < 0 or step >= layer.paddingStartStep) and (layer.paddingStopStep < 0 or step <= layer.paddingStopStep)):
            working = torch.nn.functional.pad(input, layer.paddingX, mode=layer.padding_modeX)
            working = torch.nn.functional.pad(working, layer.paddingY, mode=layer.padding_modeY)
        else:
            working = torch.nn.functional.pad(input, layer.paddingX, mode='constant')
            working = torch.nn.functional.pad(working, layer.paddingY, mode='constant')
        return torch.nn.functional.conv2d(working, weight, bias, layer.stride, (0, 0), layer.dilation, layer.groups)


    def __restoreConv2DMethods(self, model):
        for layer in [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]:
            layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)
            

    def sample(self, model, seed, tileX, tileY, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, startStep=0, stopStep=-1):
        self.__hijackConv2DMethods(model.model, tileX == 1, tileY == 1, startStep, stopStep)
        result = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        self.__restoreConv2DMethods(model.model)
        return result

