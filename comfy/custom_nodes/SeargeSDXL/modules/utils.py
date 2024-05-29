"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 Searge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import torch

from torch.nn.functional import pad


def next_multiple_of(value, factor):
    return int(int((value + factor - 1) // factor) * factor)


def get_image_size(image):
    if image is None:
        return (None, None,)

    (_, height, width, _) = image.shape

    return (width, height,)


def get_mask_size(mask):
    if mask is None:
        return (None, None,)

    (_, height, width) = mask.shape

    return (width, height,)


def get_latent_size(latent):
    if latent is None or "samples" not in latent:
        return (None, None,)

    samples = latent["samples"]

    (_, _, height, width) = samples.shape

    return (width, height,)


def get_latent_pixel_size(latent):
    (width, height) = get_latent_size(latent)

    if width is None or height is None:
        return (None, None,)

    return (width * 8, height * 8,)


def slerp(factor, input1, input2):
    dims = input1.shape

    input1 = input1.reshape(dims[0], -1)
    input2 = input2.reshape(dims[0], -1)

    input1_norm = input1 / torch.norm(input1, dim=1, keepdim=True)
    input2_norm = input2 / torch.norm(input2, dim=1, keepdim=True)

    input1_norm[input1_norm != input1_norm] = 0.0
    input2_norm[input2_norm != input2_norm] = 0.0

    omega = torch.acos((input1_norm * input2_norm).sum(1))
    sin_omega = torch.sin(omega)

    result = ((torch.sin((1.0 - factor) * omega) / sin_omega).unsqueeze(1) * input1
              + (torch.sin(factor * omega) / sin_omega).unsqueeze(1) * input2)

    return result.reshape(dims)


def slerp_latents(latent1, latent2, factor):
    result = slerp(factor, latent1.clone(), latent2.clone())
    return result


def bilateral_blur(inp, kernel_size, sigma_color, sigma_space, border_type='reflect', color_distance_type='l1'):
    if isinstance(sigma_color, torch.Tensor):
        sigma_color = sigma_color.to(device=inp.device, dtype=inp.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)

    pad_y, pad_x = (ky - 1) // 2, (kx - 1) // 2

    padded_input = pad(inp, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_input - inp.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()

    color_kernel = (-0.5 / sigma_color ** 2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, inp.device, inp.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def _unpack_2d_ks(kernel_size):
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        ky, kx = kernel_size

    return (int(ky), int(kx))


def get_gaussian_kernel2d(kernel_size, sigma, device, dtype):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)
    else:
        sigma = torch.tensor([[sigma, sigma]], device=device, dtype=dtype)

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, device, dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, device, dtype)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def get_gaussian_kernel1d(kernel_size, sigma, device, dtype):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    batch_size = sigma.shape[0]

    x = (torch.arange(kernel_size, device=sigma.device, dtype=sigma.dtype) - kernel_size // 2).expand(batch_size, -1)

    if kernel_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)
