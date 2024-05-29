
import os

import cv2
import numpy as np
import torch
from PIL import Image

from ..utils import HWC3, resize_image
from .leres.depthmap import estimateleres
from .leres.multi_depth_model_woauxi import RelDepthModel
from .leres.net_tools import strip_prefix_if_present


class LeresDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None):
        filename = filename or "res101.pth"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = pretrained_model_or_path
            
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        model = RelDepthModel(backbone='resnext101')
        model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
        del checkpoint

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, input_image, thr_a=0, thr_b=0, detect_resolution=512, image_resolution=512, output_type="np"):
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        height, width, dim = input_image.shape

        with torch.no_grad():

            depth = estimateleres(input_image, self.model, width, height)

            numbytes=2
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2**(8*numbytes))-1

            # check output before normalizing and mapping to 16 bit
            if depth_max - depth_min > np.finfo("float").eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = np.zeros(depth.shape)
            
            # single channel, 16 bit image
            depth_image = out.astype("uint16")

            # convert to uint8
            depth_image = cv2.convertScaleAbs(depth_image, alpha=(255.0/65535.0))

            # remove near
            if thr_a != 0:
                thr_a = ((thr_a/100)*255) 
                depth_image = cv2.threshold(depth_image, thr_a, 255, cv2.THRESH_TOZERO)[1]

            # invert image
            depth_image = cv2.bitwise_not(depth_image)

            # remove bg
            if thr_b != 0:
                thr_b = ((thr_b/100)*255)
                depth_image = cv2.threshold(depth_image, thr_b, 255, cv2.THRESH_TOZERO)[1]

        detected_map = depth_image
        detected_map = HWC3(detected_map)      

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
