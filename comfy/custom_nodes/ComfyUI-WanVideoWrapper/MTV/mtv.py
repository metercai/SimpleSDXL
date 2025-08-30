import torch
import numpy as np
from typing import Union, Tuple


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,      # torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def get_3d_rotary_pos_embed(
    embed_dim, crops_coords, grid_size, temporal_size, theta: int = 10000, use_real: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    if use_real is not True:
        raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")
    start, stop = crops_coords
    grid_size_h, grid_size_w = grid_size
    grid_h = np.linspace(start[0], stop[0], grid_size_h, endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], grid_size_w, endpoint=False, dtype=np.float32)
    grid_t = np.linspace(0, temporal_size, temporal_size, endpoint=False, dtype=np.float32)

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w
    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin


def get_3d_motion_spatial_embed(
    embed_dim: int, num_joints: int, joints_mean: np.ndarray, joints_std: np.ndarray, theta: float = 10000.0
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    assert embed_dim % 2 == 0 and embed_dim % 3 == 0

    def create_rope_pe(dim, pos, freqs_dtype=torch.float32):
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        freqs = (
            1.0
            / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        )  # [D/2]
        freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin

    pos_x = joints_mean[:, 0]
    pos_y = joints_mean[:, 1]
    pos_z = joints_mean[:, 2]

    normalized_pos_x = (pos_x - pos_x.mean())
    normalized_pos_y = (pos_y - pos_y.mean())
    normalized_pos_z = (pos_z - pos_z.mean())

    freqs_cos_x, freqs_sin_x = create_rope_pe(embed_dim // 3, normalized_pos_x)
    freqs_cos_y, freqs_sin_y = create_rope_pe(embed_dim // 3, normalized_pos_y)
    freqs_cos_z, freqs_sin_z = create_rope_pe(embed_dim // 3, normalized_pos_z)

    freqs_cos = torch.cat([freqs_cos_x, freqs_cos_y, freqs_cos_z], dim=-1)
    freqs_sin = torch.cat([freqs_sin_x, freqs_sin_y, freqs_sin_z], dim=-1)

    return freqs_cos, freqs_sin

def prepare_motion_embeddings(num_frames, num_joints, joints_mean, joints_std, theta=10000, device='cuda'):
    time_embed = get_1d_rotary_pos_embed(44, num_frames, theta, use_real=True)
    time_embed_cos = time_embed[0][:, None, :].expand(-1, num_joints, -1).reshape(num_frames*num_joints, -1)
    time_embed_sin = time_embed[1][:, None, :].expand(-1, num_joints, -1).reshape(num_frames*num_joints, -1)
    spatial_motion_embed = get_3d_motion_spatial_embed(84, num_joints, joints_mean, joints_std, theta)
    spatial_embed_cos = spatial_motion_embed[0][None, :, :].expand(num_frames, -1, -1).reshape(num_frames*num_joints, -1)
    spatial_embed_sin = spatial_motion_embed[1][None, :, :].expand(num_frames, -1, -1).reshape(num_frames*num_joints, -1)
    motion_embed_cos = torch.cat([time_embed_cos, spatial_embed_cos], dim=-1).to(device=device)
    motion_embed_sin = torch.cat([time_embed_sin, spatial_embed_sin], dim=-1).to(device=device)
    return motion_embed_cos, motion_embed_sin

def apply_rotary_emb(x, freqs_cis):
    cos, sin = freqs_cis  # [S, D]
    cos = cos[None, None]
    sin = sin[None, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out