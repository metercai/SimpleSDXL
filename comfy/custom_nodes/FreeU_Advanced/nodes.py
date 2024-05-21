#code originally taken from: https://github.com/ChenyangSi/FreeU (under MIT License)

import torch
import torch as th
import torch.fft as fft
import math

def normalize(latent, target_min=None, target_max=None):
    """
    Normalize a tensor `latent` between `target_min` and `target_max`.

    Args:
        latent (torch.Tensor): The input tensor to be normalized.
        target_min (float, optional): The minimum value after normalization.
            - When `None` min will be tensor min range value.
        target_max (float, optional): The maximum value after normalization.
            - When `None` max will be tensor max range value.

    Returns:
        torch.Tensor: The normalized tensor
    """
    min_val = latent.min()
    max_val = latent.max()

    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val

    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled

def hslerp(a, b, t):
    """
    Perform Hybrid Spherical Linear Interpolation (HSLERP) between two tensors.

    This function combines two input tensors `a` and `b` using HSLERP, which is a specialized
    interpolation method for smooth transitions between orientations or colors.

    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.

    Returns:
        tensor: The result of HSLERP interpolation between `a` and `b`.

    Note:
        HSLERP provides smooth transitions between orientations or colors, particularly useful
        in applications like image processing and 3D graphics.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")

    num_channels = a.size(1)

    interpolation_tensor = torch.zeros(1, num_channels, 1, 1, device=a.device, dtype=a.dtype)
    interpolation_tensor[0, 0, 0, 0] = 1.0

    result = (1 - t) * a + t * b

    if t < 0.5:
        result += (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    else:
        result -= (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor

    return result

blending_modes = {
    # Args:
    #   - a (tensor): Latent input 1
    #   - b (tensor): Latent input 2
    #   - t (float): Blending factor

    # Interpolates between tensors a and b using normalized linear interpolation.
    'bislerp': lambda a, b, t: normalize((1 - t) * a + t * b),
    # Transfer the color from `b` to `a` by t` factor
    'colorize': lambda a, b, t: a + (b - a) * t,
    # Interpolates between tensors a and b using cosine interpolation.
    'cosine interp': lambda a, b, t: (a + b - (a - b) * torch.cos(t * torch.tensor(math.pi))) / 2,
    # Interpolates between tensors a and b using cubic interpolation.
    'cuberp': lambda a, b, t: a + (b - a) * (3 * t ** 2 - 2 * t ** 3),
    # Interpolates between tensors a and b using normalized linear interpolation,
    # with a twist when t is greater than or equal to 0.5.
    'hslerp': hslerp,
    # Adds tensor b to tensor a, scaled by t.
    'inject': lambda a, b, t: a + b * t,
    # Interpolates between tensors a and b using linear interpolation.
    'lerp': lambda a, b, t: (1 - t) * a + t * b,
    # Simulates a brightening effect by adding tensor b to tensor a, scaled by t.
    'linear dodge': lambda a, b, t: normalize(a + b * t),
}

mscales = {
    "Default": None,
    "Bandpass": [
        (5, 0.0),    # Low-pass filter
        (15, 1.0),   # Pass-through filter (allows mid-range frequencies)
        (25, 0.0),   # High-pass filter
    ],
    "Low-Pass": [
        (10, 1.0),   # Allows low-frequency components, suppresses high-frequency components
    ],
    "High-Pass": [
        (10, 0.0),   # Suppresses low-frequency components, allows high-frequency components
    ],
    "Pass-Through": [
        (10, 1.0),   # Passes all frequencies unchanged, no filtering
    ],
    "Gaussian-Blur": [
        (10, 0.5),   # Blurs the image by allowing a range of frequencies with a Gaussian shape
    ],
    "Edge-Enhancement": [
        (10, 2.0),   # Enhances edges and high-frequency features while suppressing low-frequency details
    ],
    "Sharpen": [
        (10, 1.5),   # Increases the sharpness of the image by emphasizing high-frequency components
    ],
    "Multi-Bandpass": [
        [(5, 0.0), (15, 1.0), (25, 0.0)],  # Multi-scale bandpass filter
    ],
    "Multi-Low-Pass": [
        [(5, 1.0), (10, 0.5), (15, 0.2)],  # Multi-scale low-pass filter
    ],
    "Multi-High-Pass": [
        [(5, 0.0), (10, 0.5), (15, 0.8)],  # Multi-scale high-pass filter
    ],
    "Multi-Pass-Through": [
        [(5, 1.0), (10, 1.0), (15, 1.0)],  # Pass-through at different scales
    ],
    "Multi-Gaussian-Blur": [
        [(5, 0.5), (10, 0.8), (15, 0.2)],  # Multi-scale Gaussian blur
    ],
    "Multi-Edge-Enhancement": [
        [(5, 1.2), (10, 1.5), (15, 2.0)],  # Multi-scale edge enhancement
    ],
    "Multi-Sharpen": [
        [(5, 1.5), (10, 2.0), (15, 2.5)],  # Multi-scale sharpening
    ],
}

# forward function from comfy.ldm.modules.diuffusionmodules.openaimodel
# Hopefully temporary replacement
def __temp__forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", getattr(self, "default_image_only_indicator", None))
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            h = apply_control(h, control, 'input')
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control(h, control, 'middle')

        if "middle_block_patch" in transformer_patches:
            patch = transformer_patches["middle_block_patch"]
            for p in patch:
                h = p(h, transformer_options)

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

print("Patching UNetModel.forward")
import comfy.ldm.modules.diffusionmodules.openaimodel
from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, apply_control
from  comfy.ldm.modules.diffusionmodules.util import timestep_embedding
comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = __temp__forward
if comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward is __temp__forward:
    print("UNetModel.forward has been successfully patched.")
else:
    print("UNetModel.forward patching failed.")

def Fourier_filter(x, threshold, scale, scales=None, strength=1.0):
    # FFT
    if isinstance(x, list):
        x = x[0]
    if isinstance(x, torch.Tensor):
        x_freq = fft.fftn(x.float(), dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W), device=x.device)

        crow, ccol = H // 2, W // 2
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale

        if scales is not None:
            if isinstance(scales[0], tuple):
                # Single-scale mode
                for scale_params in scales:
                    if len(scale_params) == 2:
                        scale_threshold, scale_value = scale_params
                        scaled_scale_value = scale_value * strength
                        scale_mask = torch.ones((B, C, H, W), device=x.device)
                        scale_mask[..., crow - scale_threshold:crow + scale_threshold, ccol - scale_threshold:ccol + scale_threshold] = scaled_scale_value
                        mask = mask + (scale_mask - mask) * strength
            else:
                # Multi-scale mode
                for scale_params in scales:
                    if isinstance(scale_params, list):
                        for scale_tuple in scale_params:
                            if len(scale_tuple) == 2:
                                scale_threshold, scale_value = scale_tuple
                                scaled_scale_value = scale_value * strength
                                scale_mask = torch.ones((B, C, H, W), device=x.device)
                                scale_mask[..., crow - scale_threshold:crow + scale_threshold, ccol - scale_threshold:ccol + scale_threshold] = scaled_scale_value
                                mask = mask + (scale_mask - mask) * strength

        x_freq = x_freq * mask

        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

        return x_filtered.to(x.dtype)

    return x

class WAS_FreeU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "target_block": (["output_block", "middle_block", "input_block", "all"],),
                    "multiscale_mode": (list(mscales.keys()),),
                    "multiscale_strength": ("FLOAT", {"default": 1.0, "max": 1.0, "min": 0, "step": 0.001}),
                    "slice_b1": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 1}),
                    "slice_b2": ("INT", {"default": 320, "min": 64, "max": 640, "step": 1}),
                    "b1": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "b2": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                },
                "optional": {
                    "b1_mode": (list(blending_modes.keys()),),
                    "b1_blend": ("FLOAT", {"default": 1.0, "max": 100, "min": 0, "step": 0.001}),
                    "b2_mode": (list(blending_modes.keys()),),
                    "b2_blend": ("FLOAT", {"default": 1.0, "max": 100, "min": 0, "step": 0.001}),
                    "threshold": ("INT", {"default": 1.0, "max": 10, "min": 1, "step": 1}),
                    "use_override_scales": (["false", "true"],),
                    "override_scales": ("STRING", {"default": '''# OVERRIDE SCALES

# Sharpen
# 10, 1.5''', "multiline": True}),
                }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, target_block, multiscale_mode, multiscale_strength, slice_b1, slice_b2, b1, b2, s1, s2, b1_mode="add", b1_blend=1.0, b2_mode="add", b2_blend=1.0, threshold=1.0, use_override_scales="false", override_scales=""):

        min_slice = 64
        max_slice_b1 = 1280
        max_slice_b2 = 640
        slice_b1 = max(min(max_slice_b1, slice_b1), min_slice)
        slice_b2 = max(min(min(slice_b1, max_slice_b2), slice_b2), min_slice)

        scales_list = []
        if use_override_scales == "true":
            if override_scales.strip() != "":
                scales_str = override_scales.strip().splitlines()
                for line in scales_str:
                    if not line.strip().startswith('#') and not line.strip().startswith('!') and not line.strip().startswith('//'):
                        scale_values = line.split(',')
                        if len(scale_values) == 2:
                            scales_list.append((int(scale_values[0]), float(scale_values[1])))

        if use_override_scales == "true" and not scales_list:
            print("No valid override scales found. Using default scale.")
            scales_list = None

        scales = mscales[multiscale_mode] if use_override_scales == "false" else scales_list

        print(f"FreeU Plate Portions: {slice_b1} over {slice_b2}")
        print(f"FreeU Multi-Scales: {scales}")

        def block_patch(h, transformer_options):
            if h.shape[1] == 1280:
                h_t = h[:,:slice_b1]
                h_r = h_t * b1
                h[:,:slice_b1] = blending_modes[b1_mode](h_t, h_r, b1_blend)
            if h.shape[1] == 640:
                h_t = h[:,:slice_b2]
                h_r = h_t * b2
                h[:,:slice_b2] = blending_modes[b2_mode](h_t, h_r, b2_blend)
            return h

        def block_patch_hsp(h, hsp, transformer_options):
            if h.shape[1] == 1280:
                h = block_patch(h, transformer_options)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s1, scales=scales, strength=multiscale_strength)
            if h.shape[1] == 640:
                h = block_patch(h, transformer_options)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s2, scales=scales, strength=multiscale_strength)
            return h, hsp

        print(f"Patching {target_block}")

        m = model.clone()
        if target_block == "all" or target_block == "output_block":
            m.set_model_output_block_patch(block_patch_hsp)
        if target_block == "all" or target_block == "input_block":
            m.set_model_input_block_patch(block_patch)
        if target_block == "all" or target_block == "middle_block":
            m.set_model_patch(block_patch, "middle_block_patch")
        return (m, )

class WAS_FreeU_V2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "input_block": ("BOOLEAN", {"default": False}),
                    "middle_block": ("BOOLEAN", {"default": False}),
                    "output_block": ("BOOLEAN", {"default": False}),
                    "multiscale_mode": (list(mscales.keys()),),
                    "multiscale_strength": ("FLOAT", {"default": 1.0, "max": 1.0, "min": 0, "step": 0.001}),
                    "slice_b1": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 1}),
                    "slice_b2": ("INT", {"default": 320, "min": 64, "max": 640, "step": 1}),
                    "b1": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "b2": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                },
                "optional": {
                    "threshold": ("INT", {"default": 1.0, "max": 10, "min": 1, "step": 1}),
                    "use_override_scales": (["false", "true"],),
                    "override_scales": ("STRING", {"default": '''# OVERRIDE SCALES

# Sharpen
# 10, 1.5''', "multiline": True}),
                }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, input_block, middle_block, output_block, multiscale_mode, multiscale_strength, slice_b1, slice_b2, b1, b2, s1, s2, threshold=1.0, use_override_scales="false", override_scales=""):

        min_slice = 64
        max_slice_b1 = 1280
        max_slice_b2 = 640
        slice_b1 = max(min(max_slice_b1, slice_b1), min_slice)
        slice_b2 = max(min(min(slice_b1, max_slice_b2), slice_b2), min_slice)

        scales_list = []
        if use_override_scales == "true":
            if override_scales.strip() != "":
                scales_str = override_scales.strip().splitlines()
                for line in scales_str:
                    if not line.strip().startswith('#') and not line.strip().startswith('!') and not line.strip().startswith('//'):
                        scale_values = line.split(',')
                        if len(scale_values) == 2:
                            scales_list.append((int(scale_values[0]), float(scale_values[1])))

        if use_override_scales == "true" and not scales_list:
            print("No valid override scales found. Using default scale.")
            scales_list = None

        scales = mscales[multiscale_mode] if use_override_scales == "false" else scales_list

        def _hidden_mean(h):
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            return hidden_mean

        def block_patch(h, transformer_options):
            if h.shape[1] == 1280:
                hidden_mean = _hidden_mean(h)
                h[:,:slice_b1] = h[:,:slice_b1] * ((b1 - 1 ) * hidden_mean + 1)
            if h.shape[1] == 640:
                hidden_mean = _hidden_mean(h)
                h[:,:slice_b2] = h[:,:slice_b2] * ((b2 - 1 ) * hidden_mean + 1)
            return h

        def block_patch_hsp(h, hsp, transformer_options):
            if h.shape[1] == 1280:
                h = block_patch(h, transformer_options)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s1, scales=scales, strength=multiscale_strength)
            if h.shape[1] == 640:
                h = block_patch(h, transformer_options)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s2, scales=scales, strength=multiscale_strength)
            return h, hsp

        m = model.clone()
        if output_block:
            print("Patching output block")
            m.set_model_output_block_patch(block_patch_hsp)
        if input_block:
            print("Patching input block")
            m.set_model_input_block_patch(block_patch)
        if middle_block:
            print("Patching middle block")
            m.set_model_patch(block_patch, "middle_block_patch")
        return (m, )

NODE_CLASS_MAPPINGS = {
    "FreeU (Advanced)": WAS_FreeU,
    "FreeU_V2 (Advanced)": WAS_FreeU_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeU (Advanced)": "FreeU (Advanced Plus)",
    "FreeU_V2 (Advanced)": "FreeU V2 (Advanced Plus)",
}
