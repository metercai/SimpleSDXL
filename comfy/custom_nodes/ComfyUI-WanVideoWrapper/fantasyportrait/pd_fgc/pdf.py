import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...utils import log


def np_bgr_to_tensor(img_np, dtype):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) / 255.0 * 2 - 1
    return torch.tensor(img_rgb).permute(2, 0, 1).to(dtype=dtype)


def image_preprocess(np_bgr, size, dtype=torch.float32):
    img_np = cv2.resize(np_bgr, size)
    return np_bgr_to_tensor(img_np, dtype)


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def warp_face_pd_fgc(image, landmarks222, save_size=224):
    pt5_idx = [182, 202, 36, 149, 133]
    dst_pt5 = (
        np.array(
            [
                [0.3843, 0.27],
                [0.62, 0.2668],
                [0.503, 0.4185],
                [0.406, 0.5273],
                [0.5977, 0.525],
            ]
        )
        * save_size
    )
    src_pt5 = landmarks222[pt5_idx]

    M = umeyama(src_pt5, dst_pt5, True)[0:2]
    warped = cv2.warpAffine(image, M, (save_size, save_size), flags=cv2.INTER_CUBIC)

    return warped


def get_drive_expression_pd_fgc(
    pd_fpg_motion, images, landmarks, device, dtype=torch.float32
):
    emo_list = []

    motion_model = pd_fpg_motion.to(device=device)
    with tqdm(total=len(images)) as pbar:
        for frame, landmark in zip(images, landmarks):
            emo_image = warp_face_pd_fgc(frame, landmark, save_size=224)
            input_tensor = (
                image_preprocess(emo_image, (224, 224), dtype)
                .to(device=device)
                .unsqueeze(0)
            )
            # headpose_emb, eye_embed, emo_embed, mouth_feat
            # emo_tensor = motion_model(input_tensor)
            # emo_list.append(emo_tensor)
            # headpose_emb [1, 6]; eye_embed [1, 6]; emo_embed [1, 30]; mouth_feat [1, 512]
            headpose_emb, eye_embed, emo_embed, mouth_feat = motion_model(input_tensor)
            emotion = {
                "headpose_emb": headpose_emb.cpu(),
                "eye_embed": eye_embed.cpu(),
                "emo_embed": emo_embed.cpu(),
                "mouth_feat": mouth_feat.cpu(),
            }
            emo_list.append(emotion)

            pbar.set_description("PD_FPG_MOTION")
            pbar.update()

    # neg_tensor = motion_model(torch.ones_like(input_tensor)*-1).cpu()

    # ret_tensor = torch.cat(emo_list, dim=0)
    # pd_fpg_motion.to(device='cpu')
    # return dict(pd_fpg=ret_tensor.unsqueeze(0), neg_pd_fpg=neg_tensor.unsqueeze(0))
    return emo_list


def det_landmarks(face_aligner, frame_list, comfy_pbar):
    rect_list = []
    new_frame_list = []

    assert len(frame_list) > 0
    face_aligner.reset_track()

    with tqdm(total=len(frame_list)) as pbar:
        for i, frame in enumerate(frame_list):
            faces = face_aligner.forward(frame)
            if len(faces) > 0:
                face = sorted(
                    faces,
                    key=lambda x: (x["face_rect"][2] - x["face_rect"][0])
                    * (x["face_rect"][3] - x["face_rect"][1]),
                )[-1]
                rect_list.append(face["face_rect"])
                new_frame_list.append(frame)
            else:
                log.warning(f"No face detected in the frame {i}, inserting empty frame.")
                rect_list.append(None)  # Add placeholder
                new_frame_list.append(None)  # Add placeholder
            pbar.set_description("DET stage1")
            pbar.update()
    comfy_pbar.update(1)

    face_aligner.reset_track()
    save_frame_list = []
    save_landmark_list = []
    with tqdm(total=len(new_frame_list)) as pbar:
        for i, (frame, rect) in enumerate(zip(new_frame_list, rect_list)):
            if frame is None or rect is None:
                save_frame_list.append(None)
                save_landmark_list.append(None)
                log.warning(f"No face detected in the frame {i}, inserting empty landmark.")
            else:
                faces = face_aligner.forward(frame, pre_rect=rect)
                if len(faces) > 0:
                    face = sorted(
                        faces,
                        key=lambda x: (x["face_rect"][2] - x["face_rect"][0])
                        * (x["face_rect"][3] - x["face_rect"][1]),
                    )[-1]
                    landmarks = face["pre_kpt_222"]
                    save_frame_list.append(frame)
                    save_landmark_list.append(landmarks)
                else:
                    save_frame_list.append(None)
                    save_landmark_list.append(None)
                    log.warning(f"No face detected in the frame {i}, inserting empty landmark.")
            pbar.set_description("DET stage2")
            pbar.update()

    comfy_pbar.update(1)

    face_aligner.reset_track()
    return save_frame_list, save_landmark_list, rect_list


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias
    )


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.dropout = nn.Dropout(0.5)

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module("b1_" + str(level), ConvBlock(256, 256))

        self.add_module("b2_" + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module("b2_plus_" + str(level), ConvBlock(256, 256))

        self.add_module("b3_" + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules["b1_" + str(level)](up1)
        up1 = self.dropout(up1)
        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules["b2_" + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules["b2_plus_" + str(level)](low2)

        low3 = low2
        low3 = self._modules["b3_" + str(level)](low3)
        up1size = up1.size()
        rescale_size = (up1size[2], up1size[3])
        up2 = F.interpolate(low3, size=rescale_size, mode="bilinear")

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class FAN_use(nn.Module):
    def __init__(self):
        super(FAN_use, self).__init__()
        self.num_modules = 1

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        hg_module = 0
        self.add_module("m" + str(hg_module), HourGlass(1, 4, 256))
        self.add_module("top_m_" + str(hg_module), ConvBlock(256, 256))
        self.add_module(
            "conv_last" + str(hg_module),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        )
        self.add_module(
            "l" + str(hg_module), nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)
        )
        self.add_module("bn_end" + str(hg_module), nn.BatchNorm2d(256))

        if hg_module < self.num_modules - 1:
            self.add_module(
                "bl" + str(hg_module),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            )
            self.add_module(
                "al" + str(hg_module),
                nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0),
            )

        self.avgpool = nn.MaxPool2d((2, 2), 2)
        self.conv6 = nn.Conv2d(68, 1, 3, 2, 1)
        self.fc = nn.Linear(28 * 28, 512)
        self.bn5 = nn.BatchNorm2d(68)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        i = 0
        hg = self._modules["m" + str(i)](previous)

        ll = hg
        ll = self._modules["top_m_" + str(i)](ll)

        ll = self._modules["bn_end" + str(i)](self._modules["conv_last" + str(i)](ll))
        tmp_out = self._modules["l" + str(i)](F.relu(ll))

        net = self.relu(self.bn5(tmp_out))
        net = self.conv6(net)
        net = net.view(-1, net.shape[-2] * net.shape[-1])
        net = self.relu(net)
        net = self.fc(net)
        return net


class FanEncoder(nn.Module):
    def __init__(self, pose_dim=6, eye_dim=6):
        super(FanEncoder, self).__init__()
        self.model = FAN_use()

        self.to_mouth = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512)
        )
        self.mouth_embed = nn.Sequential(
            nn.ReLU(), nn.Linear(512, 512 - pose_dim - eye_dim)
        )

        self.to_headpose = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512)
        )
        self.headpose_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, pose_dim))

        self.to_eye = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512)
        )
        self.eye_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, eye_dim))

        self.to_emo = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512)
        )
        self.emo_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 30))

    def forward_feature(self, x):
        net = self.model(x)
        return net

    def forward(self, x):
        x = self.model(x)
        mouth_feat = self.to_mouth(x)
        headpose_feat = self.to_headpose(x)
        headpose_emb = self.headpose_embed(headpose_feat)
        eye_feat = self.to_eye(x)
        eye_embed = self.eye_embed(eye_feat)
        emo_feat = self.to_emo(x)
        emo_embed = self.emo_embed(emo_feat)

        return headpose_emb, eye_embed, emo_embed, mouth_feat
