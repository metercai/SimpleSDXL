import math
import time

import cv2
import numpy as np

import onnx
import onnxruntime

def create_onnx_session(onnx_path, providers) -> onnxruntime.InferenceSession:
    start = time.perf_counter()
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # providers = (
    #     [
    #         (
    #             "CUDAExecutionProvider",
    #             {
    #                 "device_id": int(gpu_id),
    #                 "arena_extend_strategy": "kNextPowerOfTwo",
    #                 "cudnn_conv_algo_search": "EXHAUSTIVE",
    #                 "do_copy_in_default_stream": True,
    #             },
    #         ),
    #         #"CPUExecutionProvider",
    #     ]
    #     #if (gpu_id is not None and gpu_id >= 0)
    #     #else ["CPUExecutionProvider"]
    # )
    sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
    # print(
    #     "create onnx session cost: {:.3f}s. {}".format(
    #         time.perf_counter() - start, onnx_path
    #     )
    # )
    return sess


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, dx0=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # self.min_cutoff = float(min_cutoff)
        # self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dx_prev = float(dx0)
        # self.t_e = fcmin

    def __call__(self, x, x_prev, fcmin=1.0, min_cutoff=1.0, beta=0.0):
        if x_prev is None:
            return x
        # t_e = 1
        a_d = smoothing_factor(fcmin, self.d_cutoff)
        dx = (x - x_prev) / fcmin
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = smoothing_factor(fcmin, cutoff)
        x_hat = exponential_smoothing(a, x, x_prev)
        self.dx_prev = dx_hat
        return x_hat


def get_warp_mat_bbox(
    face_bbox, base_angle, dst_size=128, expand_ratio=0.15, aug_angle=0.0, aug_scale=1.0
):
    face_x_min, face_y_min, face_x_max, face_y_max = face_bbox
    face_x_center = (face_x_min + face_x_max) / 2
    face_y_center = (face_y_min + face_y_max) / 2
    face_width = face_x_max - face_x_min
    face_height = face_y_max - face_y_min
    scale = dst_size / max(face_width, face_height) * (1 - expand_ratio) * aug_scale
    M = cv2.getRotationMatrix2D(
        (face_x_center, face_y_center), angle=base_angle + aug_angle, scale=scale
    )
    offset = [dst_size / 2 - face_x_center, dst_size / 2 - face_y_center]
    M[:, 2] += offset
    return M


def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points


def get_warp_mat_bbox_by_gt_pts_float(
    gt_pts, base_angle=0.0, dst_size=128, expand_ratio=0.15, return_info=False
):
    # step 1
    face_x_min, face_x_max = np.min(gt_pts[:, 0]), np.max(gt_pts[:, 0])
    face_y_min, face_y_max = np.min(gt_pts[:, 1]), np.max(gt_pts[:, 1])
    face_x_center = (face_x_min + face_x_max) / 2
    face_y_center = (face_y_min + face_y_max) / 2
    M_step_1 = cv2.getRotationMatrix2D(
        (face_x_center, face_y_center), angle=base_angle, scale=1.0
    )
    pts_step_1 = transform_points(gt_pts, M_step_1)
    face_x_min_step_1, face_x_max_step_1 = np.min(pts_step_1[:, 0]), np.max(
        pts_step_1[:, 0]
    )
    face_y_min_step_1, face_y_max_step_1 = np.min(pts_step_1[:, 1]), np.max(
        pts_step_1[:, 1]
    )
    # step 2
    face_width = face_x_max_step_1 - face_x_min_step_1
    face_height = face_y_max_step_1 - face_y_min_step_1
    scale = dst_size / max(face_width, face_height) * (1 - expand_ratio)
    M_step_2 = cv2.getRotationMatrix2D(
        (face_x_center, face_y_center), angle=base_angle, scale=scale
    )
    pts_step_2 = transform_points(gt_pts, M_step_2)
    face_x_min_step_2, face_x_max_step_2 = np.min(pts_step_2[:, 0]), np.max(
        pts_step_2[:, 0]
    )
    face_y_min_step_2, face_y_max_step_2 = np.min(pts_step_2[:, 1]), np.max(
        pts_step_2[:, 1]
    )
    face_x_center_step_2 = (face_x_min_step_2 + face_x_max_step_2) / 2
    face_y_center_step_2 = (face_y_min_step_2 + face_y_max_step_2) / 2

    M = cv2.getRotationMatrix2D(
        (face_x_center, face_y_center), angle=base_angle, scale=scale
    )
    offset = [dst_size / 2 - face_x_center_step_2, dst_size / 2 - face_y_center_step_2]
    M[:, 2] += offset

    if not return_info:
        return M
    else:
        transform_info = {
            "M": M,
            "center_x": face_x_center,
            "center_y": face_y_center,
            "rotate_angle": base_angle,
            "scale": scale,
        }
        return transform_info
