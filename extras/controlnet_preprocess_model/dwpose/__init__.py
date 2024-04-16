# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import copy
from . import util
from .wholebody import Wholebody as Wholebody_orig
from rtmlib import Wholebody, draw_skeleton
import face_alignment
from scipy.spatial.transform import Rotation

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self, paths):

        # self.pose_estimation = Wholebody_orig(paths)
        device = 'cpu'  # cpu, cuda
        backend = 'onnxruntime' 
        
        self.pose_estimation = Wholebody(to_openpose=True, 
                                                mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                                                backend=backend, device=device) # 提供动作
        
    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            return draw_pose(pose, H, W)

def transfer_pose(action_pose, ratio_pose):
    def transfer_keypoints(ratio_points, action_points, type):
        # 避免除以零的情况
        epsilon = 1e-6  # 一个很小的数，用于避免除以零

        if type == 'bodies':
            # 计算action和ratio的质心
            action_centroid = np.mean(action_points, axis=0)
            ratio_centroid = np.mean(ratio_points, axis=0)
            # 计算action的最大距离（作为缩放比例）
            action_max_distance = np.max(np.linalg.norm(action_points - action_centroid, axis=1))

            action_max_distance = np.maximum(action_max_distance, epsilon)

            # 计算ratio的最大距离（作为参考比例）
            ratio_max_distance = np.max(np.linalg.norm(ratio_points - ratio_centroid, axis=1))

            # 计算action和ratio的缩放比例
            scale_ratio = ratio_max_distance / action_max_distance

            # 缩放action的动作并移动到ratio的质心位置
            action_keypoints_transformed = (action_points - action_centroid) * scale_ratio + ratio_centroid
        elif type == 'hands':
            action_keypoints_transformed = np.zeros(action_points.shape)
            print(f'hands shape: {action_points.shape}')
            # 要分左手和右手
            # 手 1
            # 计算action和ratio的质心
            action_centroid = np.mean(action_points[0], axis=0)
            ratio_centroid = np.mean(ratio_points[0], axis=0)
            # 计算action的最大距离（作为缩放比例）
            action_max_distance = np.max(np.linalg.norm(action_points[0] - action_centroid, axis=1))
            action_max_distance = np.maximum(action_max_distance, epsilon)

            # 计算ratio的最大距离（作为参考比例）
            ratio_max_distance = np.max(np.linalg.norm(ratio_points[0] - ratio_centroid, axis=1))

            # 计算action和ratio的缩放比例
            scale_ratio0 = ratio_max_distance / action_max_distance
            action_keypoints_transformed0 = (action_points[0]  - action_centroid) * scale_ratio0 + ratio_centroid
            
            # 手 2
            # 计算action和ratio的质心
            action_centroid = np.mean(action_points[1], axis=0)
            ratio_centroid = np.mean(ratio_points[1], axis=0)
            # 计算action的最大距离（作为缩放比例）
            action_max_distance = np.max(np.linalg.norm(action_points[1] - action_centroid, axis=1))
            action_max_distance = np.maximum(action_max_distance, epsilon)

            # 计算ratio的最大距离（作为参考比例）
            ratio_max_distance = np.max(np.linalg.norm(ratio_points[1] - ratio_centroid, axis=1))

            # 计算action和ratio的缩放比例
            scale_ratio1 = ratio_max_distance / action_max_distance
            # 缩放action的动作并移动到ratio的质心位置
            action_keypoints_transformed1 = (action_points[1] - action_centroid) * scale_ratio1 + ratio_centroid
            action_keypoints_transformed[0] = action_keypoints_transformed0
            action_keypoints_transformed[1] = action_keypoints_transformed1
        elif type=='faces':
            # face
            action_keypoints_transformed = np.zeros(action_points.shape)
            
            # 计算action和ratio的质心
            action_centroid = np.mean(action_points, axis=0)
            ratio_centroid = np.mean(ratio_points, axis=0)
            # 计算action的最大距离（作为缩放比例）
            action_max_distance = np.max(np.linalg.norm(action_points - action_centroid, axis=1))
            
            action_max_distance = np.maximum(action_max_distance, epsilon)

            # 计算ratio的最大距离（作为参考比例）
            ratio_max_distance = np.max(np.linalg.norm(ratio_points - ratio_centroid, axis=1))

            # 计算action和ratio的缩放比例
            scale_ratio = ratio_max_distance / action_max_distance

            # 缩放action的动作并移动到ratio的质心位置
            action_keypoints_transformed0 = (action_points[0] - action_centroid) * scale_ratio + ratio_centroid
            action_keypoints_transformed[0] = action_keypoints_transformed0
        return action_keypoints_transformed
    
    transfered_pose = {}
    for key in ratio_pose:
        
        if key == 'bodies':
            candidate_points = transfer_keypoints(ratio_pose['bodies']['candidate'], action_pose['bodies']['candidate'], key)
            # subset_points = transfer_keypoints(ratio_pose['bodies']['subset'], action_pose['bodies']['subset'])
            subset_points = action_pose['bodies']['subset']
            transfered_points = dict(candidate=candidate_points, subset=subset_points)
        else:
            transfered_points = transfer_keypoints(ratio_pose[key], action_pose[key], key)
        # import pdb
        # pdb.set_trace()
        transfered_pose.update({key: transfered_points})
    return transfered_pose

def scale_keypoints(keypoints, scale_factor):
    center = np.mean(keypoints, axis=0)
    scaled_keypoints = (keypoints - center) * scale_factor + center
    return scaled_keypoints
def angle_between_points(p1, p2):
    """
    Calculate the angle between two points relative to the horizontal axis.
    """
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    return np.arctan2(delta_y, delta_x)
def rotate_point(p, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = p

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy
def align_angle(o1, o2, p1, p2):
    """
    Rotate p2 around p1 so that the angle between p1 and p2 is the same as the angle between o1 and o2.
    """
    # Calculate the angles
    angle_o1o2 = angle_between_points(o1, o2)
    # print('angle_o1o2',angle_o1o2)
    angle_p1p2 = angle_between_points(p1, p2)
    # print('angle_p1p2', angle_p1p2)

    # Calculate the angle difference
    angle_difference = angle_o1o2 - angle_p1p2

    # Rotate p2 around p1 by the angle difference
    p2_rotated = rotate_point(p2, p1, angle_difference)

    return p2_rotated

def rotate_face(face_act, face_rat):
    # 计算两个点云之间的质心
    centroid_act = np.mean(face_act, axis=0)
    centroid_rat = np.mean(face_rat, axis=0)

    # 将两个点云的质心移到原点
    point_act_centered = face_act - centroid_act
    point_rat_centered = face_rat - centroid_rat

    # 计算协方差矩阵
    H = np.dot(point_rat_centered.T, point_act_centered)
    # 使用SVD分解协方差矩阵
    U, S, Vt = np.linalg.svd(H)
    # 计算旋转矩阵
    R_matrix = np.dot(Vt.T, U.T)
    # 确保旋转矩阵的行列式为1（因为旋转矩阵的行列式应该是1）
    if np.linalg.det(R_matrix) < 0:
        Vt[2, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)

    rotation = Rotation.from_matrix(R_matrix)
    point_rat_rotated = rotation.apply(point_rat_centered)
    # # 将 ratio 中的每个点通过旋转矩阵进行变换
    # point_rat_rotated = np.dot(point_rat_centered, R_matrix.T)
    # 将旋转后的点云移回原来的位置
    point_rat_rotated += centroid_rat

    return point_rat_rotated[:,:2]

def move_points(points, target):
    centroid = np.mean(points, axis=0)
    points_final = points - centroid + target
    return points_final

def calculate_bounding_box(point_cloud):
    # 找到x和y坐标的最小值和最大值
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    # 返回边界框的坐标
    return (min_x, min_y, max_x, max_y)

def trans4(body, img_ref, body_tmp , img):
    rhand = copy.deepcopy(body['hands'][0])
    lhand = copy.deepcopy(body['hands'][1])
    rhand_tmp = copy.deepcopy(body_tmp['hands'][0])
    lhand_tmp = copy.deepcopy(body_tmp['hands'][1])

    face = copy.deepcopy(body['faces'][0])
    body = copy.deepcopy(body['bodies']['candidate'])

    nose = body[0]

    face_tmp = copy.deepcopy(body_tmp['faces'][0])
    body_tmp = copy.deepcopy(body_tmp['bodies']['candidate'])

    nose_tmp = body_tmp[0]
    # face[:, 0] = face[:, 0] + (nose_tmp[0] - nose[0])
    # face[:, 1] = face[:, 1] + (nose_tmp[1] - nose[1])



    body[:, 0] = body[:, 0] * img_ref.shape[1]
    body[:, 1] = body[:, 1] * img_ref.shape[0]

    body_tmp[:, 0] = body_tmp[:, 0] * img.shape[1]
    body_tmp[:, 1] = body_tmp[:, 1] * img.shape[0]

    body[2] = align_angle(body_tmp[1], body_tmp[2], body[1], body[2])  # 2
    body[5] = align_angle(body_tmp[1], body_tmp[5], body[1], body[5])  # 5
    body[3] = align_angle(body_tmp[2], body_tmp[3], body[2], body[3])  # 3
    body[6] = align_angle(body_tmp[5], body_tmp[6], body[5], body[6])  # 6
    body[4] = align_angle(body_tmp[3], body_tmp[4], body[3], body[4])  # 4
    body[7] = align_angle(body_tmp[6], body_tmp[7], body[6], body[7])  # 7
    body[8] = align_angle(body_tmp[1], body_tmp[8], body[1], body[8])  # 8
    body[11] = align_angle(body_tmp[1], body_tmp[11], body[1], body[11])  # 11
    body[9] = align_angle(body_tmp[8], body_tmp[9], body[8], body[9])  # 9
    body[12] = align_angle(body_tmp[11], body_tmp[12], body[11], body[12])  # 12
    body[10] = align_angle(body_tmp[9], body_tmp[10], body[9], body[10])  # 10
    body[13] = align_angle(body_tmp[12], body_tmp[13], body[12], body[13])  # 10
    body[0] = align_angle(body_tmp[1], body_tmp[0], body[1], body[0])  # 0
    body[14] = align_angle(body_tmp[0], body_tmp[14], body[0], body[14])  # 14
    body[15] = align_angle(body_tmp[0], body_tmp[15], body[0], body[15])  # 15
    body[16] = align_angle(body_tmp[14], body_tmp[16], body[14], body[16])  # 15
    body[17] = align_angle(body_tmp[15], body_tmp[17], body[15], body[17])  # 17
    body[:, 0] = body[:, 0] / img_ref.shape[1]
    body[:, 1] = body[:, 1] / img_ref.shape[0]
    body_tmp[:, 0] = body_tmp[:, 0] / img.shape[1]
    body_tmp[:, 1] = body_tmp[:, 1] / img.shape[0]

    lhand_b = copy.deepcopy(body[4])
    rhand_b = copy.deepcopy(body[7])
    lhand_b_tmp = copy.deepcopy(body_tmp[4])
    rhand_b_tmp = copy.deepcopy(body_tmp[7])
    face_b = copy.deepcopy(body[0])
    face_tmp_b = copy.deepcopy(face[30])

    lhand_tmp[:, 0] = lhand_tmp[:, 0] + (lhand_b[0] - lhand_b_tmp[0])
    lhand_tmp[:, 1] = lhand_tmp[:, 1] + (lhand_b[1] - lhand_b_tmp[1])

    rhand_tmp[:, 0] = rhand_tmp[:, 0] + (rhand_b[0] - rhand_b_tmp[0])
    rhand_tmp[:, 1] = rhand_tmp[:, 1] + (rhand_b[1] - rhand_b_tmp[1])

    face[:, 0] = face[:, 0] - (face_tmp_b[0] - face_b[0])
    face[:, 1] = face[:, 1] - (face_tmp_b[1] - face_b[1])

    # return scale_keypoints(body,1), scale_keypoints(face,1.0) ,lhand_tmp ,rhand_tmp
    return scale_keypoints(body,1), scale_keypoints(face,1.0), lhand_tmp, rhand_tmp

class DWposeDetectorTrans:
    def __init__(self, paths):
        device = 'cpu'  # cpu, cuda
        backend = 'onnxruntime' 
        
        self.pose_estimation = Wholebody(to_openpose=True, 
                                                mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                                                backend=backend, device=device) # 提供动作
        # self.pose_estimation = Wholebody(paths) 
        # 3D face detector
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda')
    def __call__(self, actImg, ratImg):
        actImg = actImg.copy()
        ratImg = ratImg.copy()
        HA, WA, CA = actImg.shape
        HR, WR, CR = ratImg.shape
        with torch.no_grad():
            # parse action image pose
            candidate, subset = self.pose_estimation(actImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(WA)
            candidate[..., 1] /= float(HA)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            # foot = candidate[:,18:24]
            
            faces = candidate[:,24:92]
            faces_fdepth = faces
            faces_fdepth[0,:,0] *= WA
            faces_fdepth[0,:,1] *= HA
            act_face = self.face_detector.get_landmarks(actImg, detected_faces=[calculate_bounding_box(faces_fdepth[0])])
            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies_action = dict(candidate=body, subset=score)
            pose_action = dict(bodies=bodies_action, hands=hands, faces=faces)
            ################################################################
            # parse ratio image pose
            candidate, subset = self.pose_estimation(ratImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(WR)
            candidate[..., 1] /= float(HR)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            # foot = candidate[:,18:24]
            
            faces = candidate[:,24:92]
            faces_fdepth = faces
            faces_fdepth[0,:,0] *= WR
            faces_fdepth[0,:,1] *= HR
            rat_face = self.face_detector.get_landmarks(ratImg, detected_faces=[calculate_bounding_box(faces_fdepth[0])])
            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies_ratio = dict(candidate=body, subset=score)
            pose_ratio = dict(bodies=bodies_ratio, hands=hands, faces=faces)
            ################################################################
            pose_ratio['bodies']['candidate'], pose_ratio['faces'][0], pose_ratio['hands'][0], pose_ratio['hands'][1] = trans4(pose_ratio, ratImg, pose_action, actImg)
            # pose = transfer_pose(pose_action, pose_ratio)
            
            if rat_face is None or act_face is None:
                # rotated_face = np.ones((68,2))*(-1)
                pose_ratio['faces'][0,:,:] = np.ones((68,2))*(-1)
            else:
                rotated_face = rotate_face(act_face[0], rat_face[0])
            
                rotated_face[:, 0] /= WR
                rotated_face[:, 1] /= HR
                rotated_face = move_points(rotated_face[:,:2], pose_ratio['bodies']['candidate'][0])
                pose_ratio['faces'][0,:,:] = rotated_face
            
            return draw_pose(pose_ratio, HR, WR)
            # return draw_pose(pose_action, HA, WA)