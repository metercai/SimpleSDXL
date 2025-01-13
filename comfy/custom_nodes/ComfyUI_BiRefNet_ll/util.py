import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as T


def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def refine_foreground(image_tensor, mask_tensor, r1=90, r2=7):
    if r1 % 2 == 0:
        r1 += 1

    if r2 % 2 == 0:
        r2 += 1

    return FB_blur_fusion_foreground_estimator_2(image_tensor, mask_tensor, r1=r1, r2=r2)[0]


def FB_blur_fusion_foreground_estimator_2(image_tensor, alpha_tensor, r1=90, r2=7):
    # https://github.com/Photoroom/fast-foreground-estimation
    if alpha_tensor.dim() == 3:
        alpha_tensor = alpha_tensor.unsqueeze(0)  # Add batch
    F, blur_B = FB_blur_fusion_foreground_estimator(image_tensor, image_tensor, image_tensor, alpha_tensor, r=r1)
    return FB_blur_fusion_foreground_estimator(image_tensor, F, blur_B, alpha_tensor, r=r2)


def FB_blur_fusion_foreground_estimator(image_tensor, F_tensor, B_tensor, alpha_tensor, r=90):
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    blurred_alpha = T.functional.gaussian_blur(alpha_tensor, r)

    blurred_FA = T.functional.gaussian_blur(F_tensor * alpha_tensor, r)
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = T.functional.gaussian_blur(B_tensor * (1 - alpha_tensor), r)
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F_tensor = blurred_F + alpha_tensor * (image_tensor - alpha_tensor * blurred_F - (1 - alpha_tensor) * blurred_B)
    F_tensor = torch.clamp(F_tensor, 0, 1)
    return F_tensor, blurred_B


def apply_mask_to_image(image, mask):
    """
    Apply a mask to an image and set non-masked parts to transparent.

    Args:
        image (torch.Tensor): Image tensor of shape (h, w, c) or (1, h, w, c).
        mask (torch.Tensor): Mask tensor of shape (1, 1, h, w) or (h, w).

    Returns:
        torch.Tensor: Masked image tensor of shape (h, w, c+1) with transparency.
    """
    # 判断 image 的形状
    if image.dim() == 3:
        pass
    elif image.dim() == 4:
        image = image.squeeze(0)
    else:
        raise ValueError("Image should be of shape (h, w, c) or (1, h, w, c).")

    h, w, c = image.shape
    # 判断 mask 的形状
    if mask.dim() == 4:
        mask = mask.squeeze(0).squeeze(0)  # 去掉前2个维度 (h,w)
    elif mask.dim() == 3:
        mask = mask.squeeze(0)
    elif mask.dim() == 2:
        pass
    else:
        raise ValueError("Mask should be of shape (1, 1, h, w) or (h, w).")

    assert mask.shape == (h, w), "Mask shape does not match image shape."

    # 将 mask 扩展到与 image 相同的通道数
    image_mask = mask.unsqueeze(-1).expand(h, w, c)

    # 应用遮罩，黑色部分是0，相乘后白色1的部分会被保留，其它部分变为了黑色
    masked_image = image * image_mask

    # 遮罩的黑白当做alpha通道的不透明度，黑色是0表示透明，白色是1表示不透明
    alpha = mask
    # alpha通道拼接到原图像的RGB中
    masked_image_with_alpha = torch.cat((masked_image[:, :, :3], alpha.unsqueeze(2)), dim=2)

    return masked_image_with_alpha.unsqueeze(0)


def normalize_mask(mask_tensor):
    max_val = torch.max(mask_tensor)
    min_val = torch.min(mask_tensor)

    if max_val == min_val:
        return mask_tensor

    normalized_mask = (mask_tensor - min_val) / (max_val - min_val)

    return normalized_mask

def add_mask_as_alpha(image, mask):
    """
    将 (b, h, w) 形状的 mask 添加为 (b, h, w, 3) 形状的 image 的第 4 个通道（alpha 通道）。
    """
    # 检查输入形状
    assert image.dim() == 4 and image.size(-1) == 3, "The shape of image should be (b, h, w, 3)."
    assert mask.dim() == 3, "The shape of mask should be (b, h, w)"
    assert image.size(0) == mask.size(0) and image.size(1) == mask.size(1) and image.size(2) == mask.size(2), "The batch, height, and width dimensions of the image and mask must be consistent"

    # 将 mask 扩展为 (b, h, w, 1)
    mask = mask[..., None]

    image = image * mask
    # 将 image 和 mask 拼接为 (b, h, w, 4)
    image_with_alpha = torch.cat([image, mask], dim=-1)

    return image_with_alpha

def filter_mask(mask, threshold=4e-3):
    mask_binary = mask > threshold
    filtered_mask = mask * mask_binary
    return filtered_mask
