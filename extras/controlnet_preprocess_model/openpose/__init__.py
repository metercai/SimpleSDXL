# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os

import pylab as p

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from .face import Face


# from annotator.util import annotator_ckpts_path


# body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
# hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
# face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        canvas = util.draw_handpose(canvas, hands)

    if draw_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


class OpenPose:
    def __init__(self, paths):
        # body_modelpath = os.path.join(annotator_ckpts_path, "body_pose_model.pth")
        # hand_modelpath = os.path.join(annotator_ckpts_path, "hand_pose_model.pth")
        # face_modelpath = os.path.join(annotator_ckpts_path, "facenet.pth")

        # if not os.path.exists(body_modelpath):
        #     from basicsr.utils.download_util import load_file_from_url
        #     load_file_from_url(body_model_path, model_dir=annotator_ckpts_path)
        #
        # if not os.path.exists(hand_modelpath):
        #     from basicsr.utils.download_util import load_file_from_url
        #     load_file_from_url(hand_model_path, model_dir=annotator_ckpts_path)
        #
        # if not os.path.exists(face_modelpath):
        #     from basicsr.utils.download_util import load_file_from_url
        #     load_file_from_url(face_model_path, model_dir=annotator_ckpts_path)

        # paths = {list(p.keys())[0]: list(p.values())[0] for p in paths}

        self.body_estimation = Body(paths[1])
        self.hand_estimation = Hand(paths[2])
        self.face_estimation = Face(paths[3])

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, RGB, hand_and_face=True, return_is_index=False):
        assert RGB.ndim == 3
        assert RGB.shape[2] == 3
        BGR = RGB[:, :, ::-1].copy()  # RGB to BGR
        H, W, C = BGR.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(BGR)
            hands = []
            faces = []
            if hand_and_face:
                # Hand
                hands_list = util.handDetect(candidate, subset, RGB)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(RGB[y:y + w, x:x + w, :]).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        hands.append(peaks.tolist())
                # Face
                faces_list = util.faceDetect(candidate, subset, RGB)
                for x, y, w in faces_list:
                    heatmaps = self.face_estimation(RGB[y:y + w, x:x + w, :])
                    peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        faces.append(peaks.tolist())
            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)
            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            if return_is_index:
                return pose
            else:
                return draw_pose(pose, H, W)