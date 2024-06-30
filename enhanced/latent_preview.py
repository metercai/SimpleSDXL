import torch
from PIL import Image
import numpy as np
import ldm_patched.modules.model_management as model_management

MAX_PREVIEW_RESOLUTION = 512

def preview_to_image(latent_image):
    latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                        .mul(0xFF)  # to 0..255
                        ).to(device="cpu", dtype=torch.uint8, non_blocking=model_management.device_supports_non_blocking(latent_image.device))

    return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return preview_image
        #return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].movedim(0, 2)
        return preview_to_image(x_sample)

class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        latent_image = x0[0].permute(1, 2, 0) @ self.latent_rgb_factors
        return preview_to_image(latent_image)

def get_previewer(latent_format):
    previewer = None
    if latent_format.latent_rgb_factors is not None:
        previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
    return previewer


