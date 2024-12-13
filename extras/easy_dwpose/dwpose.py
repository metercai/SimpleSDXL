from typing import Callable, Dict, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import torch
import ldm_patched.modules.model_management as model_management

from extras.easy_dwpose.body_estimation import Wholebody, resize_image
from extras.easy_dwpose.draw import draw_openpose
from modules.config import paths_controlnet, downloading_controlnet_dwpose

class DWposeDetector:
    def __init__(self, device: str = "сpu"):
        device = model_management.get_torch_device()
        modir_path_det, modir_path_pose = downloading_controlnet_dwpose()
        self.pose_estimation = Wholebody(
            device=device, model_det=modir_path_det, model_pose=modir_path_pose
        )

    def _format_pose(self, candidates, scores, width, height):
        num_candidates, _, locs = candidates.shape

        candidates[..., 0] /= float(width)
        candidates[..., 1] /= float(height)

        bodies = candidates[:, :18].copy()
        bodies = bodies.reshape(num_candidates * 18, locs)

        body_scores = scores[:, :18]
        for i in range(len(body_scores)):
            for j in range(len(body_scores[i])):
                if body_scores[i][j] > 0.3:
                    body_scores[i][j] = int(18 * i + j)
                else:
                    body_scores[i][j] = -1

        faces = candidates[:, 24:92]
        faces_scores = scores[:, 24:92]

        hands = np.vstack([candidates[:, 92:113], candidates[:, 113:]])
        hands_scores = np.vstack([scores[:, 92:113], scores[:, 113:]])

        pose = dict(
            bodies=bodies,
            body_scores=body_scores,
            hands=hands,
            hands_scores=hands_scores,
            faces=faces,
            faces_scores=faces_scores,
        )

        return pose

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        detect_resolution: int = 512,
        draw_pose: Optional[Callable] = draw_openpose,
        output_type: str = "pil",
        **kwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict]:
        if type(image) != np.ndarray:
            image = np.array(image.convert("RGB"))

        image = image.copy()
        original_height, original_width, _ = image.shape

        image = resize_image(image, target_resolution=detect_resolution)
        height, width, _ = image.shape

        candidates, scores = self.pose_estimation(image)

        pose = self._format_pose(candidates, scores, width, height)

        if not draw_pose:
            return pose

        pose_image = draw_pose(pose, height=height, width=width, **kwargs)
        pose_image = cv2.resize(pose_image, (original_width, original_height), cv2.INTER_LANCZOS4)

        if output_type == "pil":
            pose_image = PIL.Image.fromarray(pose_image)
        elif output_type == "np":
            pass
        else:
            raise ValueError("output_type should be 'pil' or 'np'")

        return pose_image
