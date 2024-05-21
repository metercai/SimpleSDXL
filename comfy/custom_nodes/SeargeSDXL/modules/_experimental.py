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

import comfy

from nodes import common_ksampler


class SeargeSamplerAdvanced:
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05}),
                "scale_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.5}),
                "scale_denoise": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 1.0, "step": 0.05}),
                "scale_method": (s.UPSCALE_METHODS,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"

    CATEGORY = "Searge/_deprecated_/_test_"

    def sample(self, model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
               denoise, scale_factor, scale_denoise, scale_method):

        if scale_factor < 1.001:
            return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                   latent_image, denoise=denoise, disable_noise=False, start_step=0,
                                   last_step=steps, force_full_denoise=True)

        total_steps = int(steps * (scale_factor + 0.001))

        result = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                 latent_image, denoise=denoise, disable_noise=False, start_step=0,
                                 last_step=steps, force_full_denoise=True)

        samples = result[0]

        extra_steps = max(0, total_steps - steps)
        width = samples["samples"].shape[3]
        height = samples["samples"].shape[2]
        new_width = int(width * (scale_factor + 0.001))
        new_height = int(height * (scale_factor + 0.001))

        s = samples.copy()

        print("Starting at {0}x{0}".format(width, height))

        step_size = 5

        # restore_steps = extra_steps
        restore_steps = int((steps + 1) // 2)
        # restore_steps = int((extra_steps + 1) // 2)

        extra_step = 0
        while extra_step < extra_steps:
            target_width = width + int((new_width - width) * (extra_step + step_size) // extra_steps)
            target_height = height + int((new_height - height) * (extra_step + step_size) // extra_steps)

            print("Scaling to {0}x{0}".format(target_width, target_height))

            s["samples"] = comfy.utils.common_upscale(samples["samples"], target_width, target_height,
                                                      scale_method, "disabled")

            samples = s

            result = common_ksampler(model, noise_seed, restore_steps, cfg, sampler_name, scheduler, positive,
                                     negative, samples, denoise=scale_denoise, disable_noise=False, start_step=0,
                                     last_step=restore_steps, force_full_denoise=True)

            samples = result[0]
            extra_step += step_size

        return (samples,)


# --------------------====================--------------------

def gaussian_latent_noise(width=128, height=128, seed=-1, fac=0.5, batch_size=1, nul=0.0, srnd=False, ver="xl"):
    limit = {
        "v1": {
            "min": {"A": -5.5618, "B": -17.1368, "C": -10.3445, "D": -8.6218},
            "max": {"A": 13.5369, "B": 11.1997, "C": 16.3043, "D": 10.6343},
            "nul": {"A": -5.3870, "B": -14.2931, "C": 6.2738, "D": 7.1220},
        },
        "xl": {
            "min": {"A": -22.2127, "B": -20.0131, "C": -17.7673, "D": -14.9434},
            "max": {"A": 17.9334, "B": 26.3043, "C": 33.1648, "D": 8.9380},
            "nul": {"A": -21.9287, "B": 3.8783, "C": 2.5879, "D": 2.5435},
        }
    }

    # seed
    if seed >= 0:
        torch.manual_seed(seed)

    limit = limit[ver]

    out = []
    for i in range(batch_size):
        if srnd:  # shared random
            rand = torch.rand([height, width])
            lat = torch.stack([
                (limit["min"]["A"] + torch.clone(rand) * (limit["max"]["A"] - limit["min"]["A"])),
                (limit["min"]["B"] + torch.clone(rand) * (limit["max"]["B"] - limit["min"]["B"])),
                (limit["min"]["C"] + torch.clone(rand) * (limit["max"]["C"] - limit["min"]["C"])),
                (limit["min"]["D"] + torch.clone(rand) * (limit["max"]["D"] - limit["min"]["D"])),
            ])

        else:  # separate random
            lat = torch.stack([
                (limit["min"]["A"] + torch.rand([height, width]) * (limit["max"]["A"] - limit["min"]["A"])),
                (limit["min"]["B"] + torch.rand([height, width]) * (limit["max"]["B"] - limit["min"]["B"])),
                (limit["min"]["C"] + torch.rand([height, width]) * (limit["max"]["C"] - limit["min"]["C"])),
                (limit["min"]["D"] + torch.rand([height, width]) * (limit["max"]["D"] - limit["min"]["D"])),
            ])

        tnul = torch.stack([  # black image
            torch.ones([height, width]) * limit["nul"]["A"],
            torch.ones([height, width]) * limit["nul"]["B"],
            torch.ones([height, width]) * limit["nul"]["C"],
            torch.ones([height, width]) * limit["nul"]["D"],
        ])

        out.append(((lat * fac) * (1.0 - nul) + tnul * nul) / 2)

    return {"samples": torch.stack(out)}
