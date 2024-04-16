# ZoeDepth
# https://github.com/isl-org/ZoeDepth

import os
import cv2
import numpy as np
import torch

from einops import rearrange
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config


class ZoeDetector:
    # remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
    # modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")
    def __init__(self, model_path):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if DEVICE == "cpu":
            print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")
        if not os.path.exists(model_path):
            raise Exception(f"ZoeDepth not found in f{model_path}")
        model = ZoeDepth.build_from_config(
            get_config("zoedepth", "infer")
        )
        model.load_state_dict(torch.load(model_path)['model'])
        model.to(DEVICE)
        model.eval()
        self.model = model

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, RGB):
        assert RGB.ndim == 3
        assert RGB.shape[2] == 3
        with torch.no_grad():
            # preprocess
            RGB = rearrange(
                torch.from_numpy(RGB).float().cuda() / 255.0,
                'h w c -> 1 c h w'
            )
            # infer
            Depth = self.model.infer(RGB).detach().squeeze().cpu().numpy()
            # postprocess
            d_min = np.percentile(Depth, 2)
            d_max = np.percentile(Depth, 85)
            if np.allclose(d_min, d_max):
                Depth = Depth * 0  # Avoid 0-division
            else:
                Depth = (Depth - d_min) / (d_max - d_min)
            Depth = 1.0 - Depth
            Depth = (Depth * 255.0).clip(0, 255).astype(np.uint8)
            return Depth