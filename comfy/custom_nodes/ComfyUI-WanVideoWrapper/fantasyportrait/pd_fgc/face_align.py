import cv2
import numpy as np

from .face_det import FaceDet
from .face_utils import (create_onnx_session, get_warp_mat_bbox,
                         get_warp_mat_bbox_by_gt_pts_float, transform_points)


class FaceAlignment(object):
    def __init__(self, providers=["CUDAExecutionProvider"], alignment_model_path="", det_model_path=""):
        expand_ratio = 0.15

        self.face_alignment_net_222 = create_onnx_session(
            alignment_model_path, providers=providers
        )
        self.onnx_input_name_222 = self.face_alignment_net_222.get_inputs()[0].name
        self.onnx_output_name_222 = [
            output.name for output in self.face_alignment_net_222.get_outputs()
        ]
        self.face_image_size = 128

        self.face_detector = FaceDet(det_model_path, providers=providers)
        self.expand_ratio = expand_ratio

    def onnx_infer(self, input_uint8):
        assert input_uint8.shape[0] == input_uint8.shape[1] == self.face_image_size
        onnx_input = (
            input_uint8.transpose((2, 0, 1)).astype(np.float32)[np.newaxis, :, :, :]
            / 255.0
        )
        landmark, euler, prob = self.face_alignment_net_222.run(
            self.onnx_output_name_222, {self.onnx_input_name_222: onnx_input}
        )

        landmark = (
            np.reshape(landmark[0], (2, -1)).transpose((1, 0)) * self.face_image_size
        )
        left_eye_corner = landmark[74]
        right_eye_corner = landmark[96]
        radian = np.arctan2(
            right_eye_corner[1] - left_eye_corner[1],
            right_eye_corner[0] - left_eye_corner[0] + 0.00000001,
        )
        euler_rad = np.array([euler[0, 0], euler[0, 1], radian], dtype=np.float32)
        prob = prob[0]

        return landmark, euler_rad, prob

    def forward(self, src_image, face_box=None, pre_pts=None, iterations=3):
        if pre_pts is None:
            if face_box is None:
                # Detect max size face
                bounding_boxes, _, score = self.face_detector.detect(src_image)
                print("facedet score", score)
                if len(bounding_boxes) == 0:
                    return None
                bbox = np.zeros(4, dtype=np.float32)
                if len(bounding_boxes) >= 1:
                    max_area = 0.0
                    for each_bbox in bounding_boxes:
                        area = (each_bbox[2] - each_bbox[0]) * (
                            each_bbox[3] - each_bbox[1]
                        )
                        if area > max_area:
                            bbox[:4] = each_bbox[:4]
                        max_area = area
                else:
                    bbox = bounding_boxes[0, :4]
            else:
                bbox = face_box.copy()
            M_Face = get_warp_mat_bbox(
                bbox, 0, self.face_image_size, expand_ratio=self.expand_ratio
            )
        else:
            left_eye_corner = pre_pts[74]
            right_eye_corner = pre_pts[96]

            radian = np.arctan2(
                right_eye_corner[1] - left_eye_corner[1],
                right_eye_corner[0] - left_eye_corner[0] + 0.00000001,
            )
            M_Face = get_warp_mat_bbox_by_gt_pts_float(
                pre_pts,
                np.rad2deg(radian),
                self.face_image_size,
                expand_ratio=self.expand_ratio,
            )

        face_input = cv2.warpAffine(
            src_image, M_Face, (self.face_image_size, self.face_image_size)
        )
        landmarks, euler, prob = self.onnx_infer(face_input)
        landmarks = transform_points(landmarks, M_Face, invert=True)

        # Repeat
        for i in range(iterations - 1):
            M_Face = get_warp_mat_bbox_by_gt_pts_float(
                landmarks,
                np.rad2deg(euler[2]),
                self.face_image_size,
                expand_ratio=self.expand_ratio,
            )
            face_input = cv2.warpAffine(
                src_image, M_Face, (self.face_image_size, self.face_image_size)
            )
            landmarks, euler, prob = self.onnx_infer(face_input)
            landmarks = transform_points(landmarks, M_Face, invert=True)

        return_dict = {
            "pt222": landmarks,
            "euler_rad": euler,
            "prob": prob,
            "M_Face": M_Face,
            "face_input": face_input,
        }

        return return_dict
