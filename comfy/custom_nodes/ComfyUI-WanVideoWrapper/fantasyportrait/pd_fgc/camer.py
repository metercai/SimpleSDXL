import math
import numpy as np

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, dx0=0.0, d_cutoff=1.0):
        self.d_cutoff = float(d_cutoff)
        self.dx_prev = float(dx0)

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


def cult_dis(old_kpts, new_kpts):
    dis = np.sqrt(
        np.square(new_kpts[:, 0] - old_kpts[:, 0])
        + np.square(new_kpts[:, 1] - old_kpts[:, 1])
    )
    return dis


class Smoother222(object):
    def __init__(self):
        # face config
        self.face_idx = list(range(0, 33))
        self.face_down_idx = list(range(9, 24))
        self.filter_face = OneEuroFilter()
        # nose config
        self.nose_idx = list(range(33, 48))
        self.filter_nose = OneEuroFilter()
        # eyebrow config
        self.eyebrow_idx = list(range(48, 74))
        self.filter_eyebrow = OneEuroFilter()
        # eye config
        self.left_eye_idx = list(range(74, 96))
        self.filter_left_eye = OneEuroFilter()
        self.right_eye_idx = list(range(96, 118))
        self.filter_right_eye = OneEuroFilter()
        # mouth config
        self.mouth_idx = list(range(118, 182))
        self.filter_mouth = OneEuroFilter()
        # pupil config
        self.left_pupil_idx = list(range(182, 202))
        self.filter_left_pupil = OneEuroFilter()
        self.right_pupil_idx = list(range(202, 222))
        self.filter_right_pupil = OneEuroFilter()
        self.prev_points = None

    def smooth(self, new_points, face_dis):
        if self.prev_points is None:
            self.prev_points = new_points.copy()
            return new_points
        dis = cult_dis(self.prev_points, new_points) / face_dis
        smooth_points = new_points.copy()

        # smooth face
        if np.mean(dis[self.face_down_idx]) < 0.005:
            ratio_tmp = np.mean(dis[self.face_down_idx]) / 0.005
            fcmin_tmp = 0.05 * ratio_tmp
            beta_tmp = 0.05 * ratio_tmp
            smooth_points[self.face_idx] = self.filter_face(
                new_points[self.face_idx],
                self.prev_points[self.face_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        elif np.mean(dis[self.face_down_idx]) < 0.02:
            ratio_tmp = (np.mean(dis[self.face_down_idx]) - 0.005) / (0.02 - 0.005)
            fcmin_tmp = 0.05 + (0.3 - 0.05) * ratio_tmp
            beta_tmp = 0.05 + (0.3 - 0.05) * ratio_tmp
            smooth_points[self.face_idx] = self.filter_face(
                new_points[self.face_idx],
                self.prev_points[self.face_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        else:
            smooth_points[self.face_idx] = self.filter_face(
                new_points[self.face_idx],
                self.prev_points[self.face_idx],
                fcmin=0.3,
                beta=0.3,
            )
        # smooth nose
        if np.mean(dis[self.nose_idx]) < 0.003:
            # stable
            ratio_tmp = np.mean(dis[self.nose_idx]) / 0.003
            fcmin_tmp = 0.03 * ratio_tmp
            beta_tmp = 0.03 * ratio_tmp
            smooth_points[self.nose_idx] = self.filter_nose(
                new_points[self.nose_idx],
                self.prev_points[self.nose_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        elif np.mean(dis[self.nose_idx]) < 0.02:
            ratio_tmp = (np.mean(dis[self.nose_idx]) - 0.003) / (0.02 - 0.003)
            fcmin_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            beta_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            smooth_points[self.nose_idx] = self.filter_nose(
                new_points[self.nose_idx],
                self.prev_points[self.nose_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        else:
            # filter
            smooth_points[self.nose_idx] = self.filter_nose(
                new_points[self.nose_idx],
                self.prev_points[self.nose_idx],
                fcmin=0.7,
                beta=0.7,
            )
        # smooth eyebrow
        if np.mean(dis[self.eyebrow_idx]) < 0.003:
            # stable
            ratio_tmp = np.mean(dis[self.eyebrow_idx]) / 0.003
            fcmin_tmp = 0.02 * ratio_tmp
            beta_tmp = 0.02 * ratio_tmp
            smooth_points[self.eyebrow_idx] = self.filter_eyebrow(
                new_points[self.eyebrow_idx],
                self.prev_points[self.eyebrow_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        elif np.mean(dis[self.eyebrow_idx]) < 0.02:
            # filter
            ratio_tmp = (np.mean(dis[self.eyebrow_idx]) - 0.003) / (0.02 - 0.003)
            fcmin_tmp = 0.02 + (0.5 - 0.02) * ratio_tmp
            beta_tmp = 0.02 + (0.5 - 0.02) * ratio_tmp
            smooth_points[self.eyebrow_idx] = self.filter_eyebrow(
                new_points[self.eyebrow_idx],
                self.prev_points[self.eyebrow_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        else:
            # filter
            smooth_points[self.eyebrow_idx] = self.filter_eyebrow(
                new_points[self.eyebrow_idx],
                self.prev_points[self.eyebrow_idx],
                fcmin=0.5,
                beta=0.5,
            )
        # smooth eye
        if np.mean(dis[self.left_eye_idx]) < 0.003:
            # stable
            ratio_tmp = np.mean(dis[self.left_eye_idx]) / 0.003
            fcmin_tmp = 0.03 * ratio_tmp
            beta_tmp = 0.03 * ratio_tmp
            smooth_points[self.left_eye_idx] = self.filter_left_eye(
                new_points[self.left_eye_idx],
                self.prev_points[self.left_eye_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        elif np.mean(dis[self.left_eye_idx]) < 0.02:
            # filter
            ratio_tmp = (np.mean(dis[self.left_eye_idx]) - 0.003) / (0.02 - 0.003)
            fcmin_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            beta_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            smooth_points[self.left_eye_idx] = self.filter_left_eye(
                new_points[self.left_eye_idx],
                self.prev_points[self.left_eye_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        else:
            # fast
            smooth_points[self.left_eye_idx] = self.filter_left_eye(
                new_points[self.left_eye_idx],
                self.prev_points[self.left_eye_idx],
                fcmin=0.7,
                beta=0.7,
            )
        if np.mean(dis[self.right_eye_idx]) < 0.003:
            # stable
            ratio_tmp = np.mean(dis[self.right_eye_idx]) / 0.003
            fcmin_tmp = 0.03 * ratio_tmp
            beta_tmp = 0.03 * ratio_tmp
            smooth_points[self.right_eye_idx] = self.filter_right_eye(
                new_points[self.right_eye_idx],
                self.prev_points[self.right_eye_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        elif np.mean(dis[self.right_eye_idx]) < 0.02:
            # filter
            ratio_tmp = (np.mean(dis[self.right_eye_idx]) - 0.003) / (0.02 - 0.003)
            fcmin_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            beta_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            smooth_points[self.right_eye_idx] = self.filter_right_eye(
                new_points[self.right_eye_idx],
                self.prev_points[self.right_eye_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        else:
            # fast
            smooth_points[self.right_eye_idx] = self.filter_right_eye(
                new_points[self.right_eye_idx],
                self.prev_points[self.right_eye_idx],
                fcmin=0.7,
                beta=0.7,
            )

        # smooth mouth
        if np.mean(dis[self.mouth_idx]) < 0.003:
            # stable
            ratio_tmp = np.mean(dis[self.mouth_idx]) / 0.003
            fcmin_tmp = 0.05 * ratio_tmp
            beta_tmp = 0.05 * ratio_tmp
            smooth_points[self.mouth_idx] = self.filter_mouth(
                new_points[self.mouth_idx],
                self.prev_points[self.mouth_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        elif np.mean(dis[self.mouth_idx]) < 0.02:
            # filter
            ratio_tmp = (np.mean(dis[self.mouth_idx]) - 0.003) / (0.02 - 0.003)
            fcmin_tmp = 0.05 + (0.7 - 0.05) * ratio_tmp
            beta_tmp = 0.05 + (0.7 - 0.05) * ratio_tmp
            smooth_points[self.mouth_idx] = self.filter_mouth(
                new_points[self.mouth_idx],
                self.prev_points[self.mouth_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        else:
            # fast
            smooth_points[self.mouth_idx] = self.filter_mouth(
                new_points[self.mouth_idx],
                self.prev_points[self.mouth_idx],
                fcmin=0.7,
                beta=0.7,
            )

        # smooth pupil
        if np.mean(dis[self.left_pupil_idx]) < 0.003:
            # stable
            ratio_tmp = np.mean(dis[self.left_pupil_idx]) / 0.003
            fcmin_tmp = 0.03 * ratio_tmp
            beta_tmp = 0.03 * ratio_tmp
            smooth_points[self.left_pupil_idx] = self.filter_left_pupil(
                new_points[self.left_pupil_idx],
                self.prev_points[self.left_pupil_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        elif np.mean(dis[self.left_pupil_idx]) < 0.02:
            # filter
            ratio_tmp = (np.mean(dis[self.left_pupil_idx]) - 0.003) / (0.02 - 0.003)
            fcmin_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            beta_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            smooth_points[self.left_pupil_idx] = self.filter_left_pupil(
                new_points[self.left_pupil_idx],
                self.prev_points[self.left_pupil_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        else:
            # fast
            smooth_points[self.left_pupil_idx] = self.filter_left_pupil(
                new_points[self.left_pupil_idx],
                self.prev_points[self.left_pupil_idx],
                fcmin=0.7,
                beta=0.7,
            )
        if np.mean(dis[self.right_pupil_idx]) < 0.003:
            # stable
            ratio_tmp = np.mean(dis[self.right_pupil_idx]) / 0.003
            fcmin_tmp = 0.03 * ratio_tmp
            beta_tmp = 0.03 * ratio_tmp
            smooth_points[self.right_pupil_idx] = self.filter_right_pupil(
                new_points[self.right_pupil_idx],
                self.prev_points[self.right_pupil_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        elif np.mean(dis[self.right_pupil_idx]) < 0.02:
            # filter
            ratio_tmp = (np.mean(dis[self.right_pupil_idx]) - 0.003) / (0.02 - 0.003)
            fcmin_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            beta_tmp = 0.03 + (0.7 - 0.03) * ratio_tmp
            smooth_points[self.right_pupil_idx] = self.filter_right_pupil(
                new_points[self.right_pupil_idx],
                self.prev_points[self.right_pupil_idx],
                fcmin=fcmin_tmp,
                beta=beta_tmp,
            )
        else:
            # fast
            smooth_points[self.right_pupil_idx] = self.filter_right_pupil(
                new_points[self.right_pupil_idx],
                self.prev_points[self.right_pupil_idx],
                fcmin=0.7,
                beta=0.7,
            )

        # update pre points
        self.prev_points = smooth_points
        return smooth_points


class CameraDemo(object):
    def __init__(self, face_alignment_module, reset=False):
        self.face_alignment_module = face_alignment_module
        self.face_prob_th = 0.0001
        self.min_face = 96
        self.face_image_size = self.face_alignment_module.face_image_size
        self.trackingFaces = []
        self.reset = reset

    def reset_track(self):
        self.trackingFaces = []

    def forward(self, src_image, reset=False, pre_rect=None):

        if self.reset or reset:
            self.trackingFaces = []

        if len(self.trackingFaces) == 0:
            if pre_rect is not None:
                detected_faces = [pre_rect]
            else:
                detected_faces, _, _ = self.face_alignment_module.face_detector.detect(
                    src_image
                )
            for face_rect in detected_faces:
                new_tracking_object = {
                    "face_rect": face_rect,
                    "rotate_angle": 0.0,
                    "pre_kpt_222": None,
                    "face_dis": np.sqrt(
                        np.square((face_rect[2] - face_rect[0]))
                        + np.square((face_rect[3] - face_rect[1]))
                    ),
                    "smoother_222": Smoother222(),
                    "prob": 0,
                }
                self.trackingFaces.append(new_tracking_object)
        else:
            detected_faces, _, _ = self.face_alignment_module.face_detector.detect(
                src_image
            )
            for face_rect in detected_faces:
                new_tracking_object = {
                    "face_rect": face_rect,
                    "rotate_angle": 0.0,
                    "pre_kpt_222": None,
                    "face_dis": np.sqrt(
                        np.square((face_rect[2] - face_rect[0]))
                        + np.square((face_rect[3] - face_rect[1]))
                    ),
                    "smoother_222": Smoother222(),
                    "prob": 0,
                }
                self.trackingFaces.append(new_tracking_object)

        delete_idx_list = []
        for face_idx, tracking_face in enumerate(self.trackingFaces):
            if tracking_face["pre_kpt_222"] is not None:
                result_dict = self.face_alignment_module.forward(
                    src_image, pre_pts=tracking_face["pre_kpt_222"], iterations=3
                )
            else:
                result_dict = self.face_alignment_module.forward(
                    src_image, face_box=tracking_face["face_rect"], iterations=3
                )

            if result_dict["prob"] < self.face_prob_th:
                if not face_idx in delete_idx_list:
                    delete_idx_list.append(face_idx)
                continue

            landmarks_final = tracking_face["smoother_222"].smooth(
                result_dict["pt222"], tracking_face["face_dis"]
            )
            tracking_face["pre_kpt_222"] = landmarks_final

            left_eye_corner = landmarks_final[74]
            right_eye_corner = landmarks_final[96]

            radian = np.arctan2(
                right_eye_corner[1] - left_eye_corner[1],
                right_eye_corner[0] - left_eye_corner[0] + 0.00000001,
            )
            rotate_angle = np.rad2deg(radian)
            face_x_min, face_x_max = np.min(landmarks_final[:, 0]), np.max(
                landmarks_final[:, 0]
            )
            face_y_min, face_y_max = np.min(landmarks_final[:, 1]), np.max(
                landmarks_final[:, 1]
            )
            face_bbox = [face_x_min, face_y_min, face_x_max, face_y_max]
            face_dis = np.linalg.norm(landmarks_final[0] - landmarks_final[32])

            if (
                face_x_max - face_x_min < self.min_face
                or face_y_max - face_y_min < self.min_face
            ):
                if not face_idx in delete_idx_list:
                    delete_idx_list.append(face_idx)

            euler_pred = result_dict["euler_rad"]
            pitch = np.rad2deg(euler_pred[0])
            yaw = np.rad2deg(euler_pred[1])
            roll = np.rad2deg(euler_pred[2])
            # print("pitch, yaw, roll", pitch, yaw, roll)

            # one filter model
            max_euler = abs(pitch) + (abs(yaw) * 0.6)
            face_dis *= 1.0 + max_euler / 18.0

            # two filter model
            tracking_face["face_rect"] = face_bbox
            tracking_face["rotate_angle"] = rotate_angle
            tracking_face["face_dis"] = face_dis
            tracking_face["prob"] = result_dict["prob"]
            tracking_face["pitch"] = pitch
            tracking_face["yaw"] = yaw
            tracking_face["roll"] = roll
            tracking_face["euler_rad"] = result_dict["euler_rad"]

        if len(self.trackingFaces) > 1:
            for face_idx, tracking_face_target in enumerate(self.trackingFaces):
                if face_idx in delete_idx_list:
                    continue
                for idx, tracking_face in enumerate(self.trackingFaces):
                    if idx in delete_idx_list:
                        continue
                    if face_idx == idx:
                        continue
                    iou_temp = self.count_iou(
                        tracking_face_target["face_rect"], tracking_face["face_rect"]
                    )
                    # prog 2
                    if iou_temp > 0.12:
                        if (
                            self.area(tracking_face_target["face_rect"])
                            - self.area(tracking_face["face_rect"])
                            < 0
                        ):
                            if not face_idx in delete_idx_list:
                                delete_idx_list.append(face_idx)
                        else:
                            if not idx in delete_idx_list:
                                delete_idx_list.append(idx)

        idx_offset = 0
        for delete_idx in sorted(delete_idx_list):
            self.trackingFaces.pop(delete_idx - idx_offset)
            idx_offset += 1

        return self.trackingFaces

    def count_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def area(self, bbox):
        w = bbox[3] - bbox[1]
        h = bbox[2] - bbox[0]
        return w * h
