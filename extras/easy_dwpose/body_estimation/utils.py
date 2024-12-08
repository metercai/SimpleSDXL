import cv2
import numpy as np


def resize_image(input_image: np.ndarray, target_resolution: input = 512, dividable_by: int = 64) -> np.ndarray:
    height, width, _ = input_image.shape

    k = float(target_resolution) / min(height, width)

    target_width = width * k
    target_width = int(np.round(target_width / dividable_by)) * dividable_by

    target_height = height * k
    target_height = int(np.round(target_height / dividable_by)) * dividable_by

    return cv2.resize(
        input_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )
