# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import warnings
from typing import Callable, List, NamedTuple, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from . import util
from .body import Body, BodyResult, Keypoint
from .face import Face
from .hand import Hand

import ldm_patched.modules.model_management as model_management
from modules.util import HWC3
from modules.config import downloading_controlnet_openpose

HandResult = List[Keypoint]
FaceResult = List[Keypoint]

class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Union[HandResult, None]
    right_hand: Union[HandResult, None]
    face: Union[FaceResult, None]

def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True, xinsr_stick_scaling=False):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints, xinsr_stick_scaling)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas

def encode_poses_as_dict(poses: List[PoseResult], canvas_height: int, canvas_width: int) -> str:
    """ Encode the pose as a dict following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """
    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None
        
        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return {
        'people': [
            {
                'pose_keypoints_2d': compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d":compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        'canvas_height': canvas_height,
        'canvas_width': canvas_width,
    }
    
class OpenposeDetector:
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    def __init__(self, body_estimation, hand_estimation=None, face_estimation=None):
        self.body_estimation = body_estimation
        self.hand_estimation = hand_estimation
        self.face_estimation = face_estimation

    @classmethod
    def from_pretrained(cls):
        body_model_path, hand_model_path, face_model_path = downloading_controlnet_openpose()

        body_estimation = Body(body_model_path)
        hand_estimation = Hand(hand_model_path)
        face_estimation = Face(face_model_path)

        return cls(body_estimation, hand_estimation, face_estimation)

    def to(self):
        device = model_management.get_torch_device()
        self.body_estimation.to(device)
        self.hand_estimation.to(device)
        self.face_estimation.to(device)
        return self

    def detect_hands(self, body: BodyResult, oriImg) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in util.handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :]).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                
                hand_result = [
                    Keypoint(x=peak[0], y=peak[1])
                    for peak in peaks
                ]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = util.faceDetect(body, oriImg)
        if face is None:
            return None
        
        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y:y+w, x:x+w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            return [
                Keypoint(x=peak[0], y=peak[1])
                for peak in peaks
            ]
        
        return None

    def detect_poses(self, oriImg, include_hand=False, include_face=False) -> List[PoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[PoseResult]: A list of PoseResult objects containing the detected poses.
        """
        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = (None,) * 3
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)
                
                results.append(PoseResult(BodyResult(
                    keypoints=[
                        Keypoint(
                            x=keypoint.x / float(W),
                            y=keypoint.y / float(H)
                        ) if keypoint is not None else None
                        for keypoint in body.keypoints
                    ], 
                    total_score=body.total_score,
                    total_parts=body.total_parts
                ), left_hand, right_hand, face))
            
            return results
        
    def __call__(self, input_image, detect_resolution=512, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", image_and_json=False, upscale_method="INTER_CUBIC", xinsr_stick_scaling=False, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        poses = self.detect_poses(input_image, include_hand=include_hand, include_face=include_face)
        canvas = draw_poses(poses, input_image.shape[0], input_image.shape[1], draw_body=include_body, draw_hand=include_hand, draw_face=include_face, xinsr_stick_scaling=xinsr_stick_scaling) 
        detected_map = HWC3(remove_pad(canvas))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        
        if image_and_json:
            return (detected_map, encode_poses_as_dict(poses, detected_map.shape[0], detected_map.shape[1]))
        return detected_map

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
def get_upscale_method(method_str):
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    return getattr(cv2, method_str)

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

#https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L17
#Added upscale_method, mode params
def resize_image_with_pad(input_image, resolution, upscale_method = "", skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad

def common_input_validate(input_image, output_type, **kwargs):
    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"

    return (input_image, output_type)

