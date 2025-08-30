import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        mid_channels=[128, 512], 
        out_channels=3072,
        downsample_time=[1, 1],
        downsample_joint=[1, 1],
        num_attention_heads=8,
        attention_head_dim=64,
        dim=3072,
        ):
        super(Encoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1)
        self.resnet1 = nn.ModuleList([ResBlock(mid_channels[0], mid_channels[0]) for _ in range(3)])
        self.downsample1 = Downsample(mid_channels[0], mid_channels[0], downsample_time[0], downsample_joint[0])
        self.resnet2 = ResBlock(mid_channels[0], mid_channels[1])
        self.resnet3 = nn.ModuleList([ResBlock(mid_channels[1], mid_channels[1]) for _ in range(3)])
        self.downsample2 = Downsample(mid_channels[1], mid_channels[1], downsample_time[1], downsample_joint[1])
        self.conv_out = nn.Conv2d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        for resnet in self.resnet1:
            x = resnet(x)
        x = self.downsample1(x)
        
        x = self.resnet2(x)
        for resnet in self.resnet3:
            x = resnet(x)
        x = self.downsample2(x)

        x = self.conv_out(x)

        return x



class VectorQuantizer(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.reset_codebook()
        self.reset_count = 0
        self.usage = torch.zeros((self.nb_code, 1))
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())
    
    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def preprocess(self, x):
        # [bs, c, f, j] -> [bs * f * j, c]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # [bs * f * j, dim=3072]
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)    # indexing: [bs * f * j, 32]
        return x

    def forward(self, x, return_vq=False):
        bs, c, f, j = x.shape   # SMPL data frames: [bs, 3072, f, j]

        # Preprocess
        x = self.preprocess(x)
        # return x.view(bs, f*j, c).contiguous(), None
        assert x.shape[-1] == self.code_dim

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        if return_vq:
            return x_d.view(bs, f*j, c).contiguous(), commit_loss
            # return (x_d, x_d.view(bs, f, j, c).permute(0, 3, 1, 2).contiguous()), commit_loss, perplexity

        # Postprocess
        x_d = x_d.view(bs, f, j, c).permute(0, 3, 1, 2).contiguous()

        return x_d, commit_loss




class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels=3072, 
        mid_channels=[512, 128], 
        out_channels=3,
        upsample_rate=None,
        frame_upsample_rate=[1.0, 1.0],
        joint_upsample_rate=[1.0, 1.0],
        dim=128,
        attention_head_dim=64,
        num_attention_heads=8,
        ):
        super(Decoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1)
        self.resnet1 = nn.ModuleList([ResBlock(mid_channels[0], mid_channels[0]) for _ in range(3)])
        self.upsample1 = Upsample(mid_channels[0], mid_channels[0], frame_upsample_rate=frame_upsample_rate[0], joint_upsample_rate=joint_upsample_rate[0])
        self.resnet2 = ResBlock(mid_channels[0], mid_channels[1])
        self.resnet3 = nn.ModuleList([ResBlock(mid_channels[1], mid_channels[1]) for _ in range(3)])
        self.upsample2 = Upsample(mid_channels[1], mid_channels[1], frame_upsample_rate=frame_upsample_rate[1], joint_upsample_rate=joint_upsample_rate[1])
        self.conv_out = nn.Conv2d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for resnet in self.resnet1:
            x = resnet(x)
        x = self.upsample1(x)

        x = self.resnet2(x)
        for resnet in self.resnet3:
            x = resnet(x)
        x = self.upsample2(x)

        x = self.conv_out(x)

        return x


class Upsample(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        upsample_rate=None, 
        frame_upsample_rate=None,
        joint_upsample_rate=None,
        ):
        super(Upsample, self).__init__()

        self.upsampler = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample_rate = upsample_rate
        self.frame_upsample_rate = frame_upsample_rate
        self.joint_upsample_rate = joint_upsample_rate
        self.upsample_rate = upsample_rate

    def forward(self, inputs):
        if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
            # split first frame
            x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

            if self.upsample_rate is not None:
                # import pdb; pdb.set_trace()
                x_first = F.interpolate(x_first, scale_factor=self.upsample_rate)
                x_rest = F.interpolate(x_rest, scale_factor=self.upsample_rate)
            else:
                # import pdb; pdb.set_trace()
                # x_first = F.interpolate(x_first, scale_factor=(self.frame_upsample_rate, self.joint_upsample_rate), mode="bilinear", align_corners=True)
                x_rest = F.interpolate(x_rest, scale_factor=(self.frame_upsample_rate, self.joint_upsample_rate), mode="bilinear", align_corners=True)
            x_first = x_first[:, :, None, :]
            inputs = torch.cat([x_first, x_rest], dim=2)
        elif inputs.shape[2] > 1:
            if self.upsample_rate is not None:
                inputs = F.interpolate(inputs, scale_factor=self.upsample_rate)
            else:
                inputs = F.interpolate(inputs, scale_factor=(self.frame_upsample_rate, self.joint_upsample_rate), mode="bilinear", align_corners=True)
        else:
            inputs = inputs.squeeze(2)
            if self.upsample_rate is not None:
                inputs = F.interpolate(inputs, scale_factor=self.upsample_rate)
            else:
                inputs = F.interpolate(inputs, scale_factor=(self.frame_upsample_rate, self.joint_upsample_rate), mode="linear", align_corners=True)
            inputs = inputs[:, :, None, :, :]

        b, c, t, j = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3).reshape(b * t, c, j)
        inputs = self.upsampler(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3)

        return inputs


class Downsample(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        frame_downsample_rate, 
        joint_downsample_rate
        ):
        super(Downsample, self).__init__()

        self.frame_downsample_rate = frame_downsample_rate
        self.joint_downsample_rate = joint_downsample_rate
        self.joint_downsample = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=self.joint_downsample_rate, padding=1)

    def forward(self, x):
        # (batch_size, channels, frames, joints) -> (batch_size * joints, channels, frames)
        if self.frame_downsample_rate > 1:
            batch_size, channels, frames, joints = x.shape
            x = x.permute(0, 3, 1, 2).reshape(batch_size * joints, channels, frames)
            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                    x_rest = F.avg_pool1d(x_rest, kernel_size=self.frame_downsample_rate, stride=self.frame_downsample_rate)

                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                # (batch_size * joints, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, joints)
                x = x.reshape(batch_size, joints, channels, x.shape[-1]).permute(0, 2, 3, 1)
            else:
                # (batch_size * joints, channels, frames) -> (batch_size * joints, channels, frames // 2)
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                # (batch_size * joints, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                x = x.reshape(batch_size, joints, channels, x.shape[-1]).permute(0, 2, 3, 1)
        
        # Pad the tensor
        # pad = (0, 1)
        # x = F.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, joints = x.shape
        # (batch_size, channels, frames, joints) -> (batch_size * frames, channels, joints)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * frames, channels, joints)
        x = self.joint_downsample(x)
        # (batch_size * frames, channels, joints) -> (batch_size, channels, frames, joints)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)
        return x



class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 group_num=32,
                 max_channels=512):
        super(ResBlock, self).__init__()
        skip = max(1, max_channels // out_channels - 1)
        self.block = nn.Sequential(
            nn.GroupNorm(group_num, in_channels, eps=1e-06, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=skip, dilation=skip),
            nn.GroupNorm(group_num, out_channels, eps=1e-06, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.conv_short = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        hidden_states = self.block(x)
        if hidden_states.shape != x.shape:
            x = self.conv_short(x)
        x = x + hidden_states
        return x



class SMPL_VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):
        super(SMPL_VQVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.vq = self.vq.to(device)
        self.device = device
        return self
    
    def encdec_slice_frames(self, x, frame_batch_size, encdec, return_vq):
        num_frames = x.shape[2]
        remaining_frames = num_frames % frame_batch_size
        x_output = []
    
        for i in range(num_frames // frame_batch_size):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, :, start_frame:end_frame]
            x_intermediate = encdec(x_intermediate)
            x_output.append(x_intermediate)
        if encdec == self.encoder and self.vq is not None:
            x_output, loss = self.vq(torch.cat(x_output, dim=2), return_vq=return_vq)
            return x_output, loss
        else:
            return torch.cat(x_output, dim=2), None, None
    
    def forward(self, x, return_vq=False):
        x = x.permute(0, 3, 1, 2)   
        x, loss = self.encdec_slice_frames(x, frame_batch_size=8, encdec=self.encoder, return_vq=return_vq)
    
        if return_vq:
            return x, loss
        x, _, _ = self.encdec_slice_frames(x, frame_batch_size=2, encdec=self.decoder, return_vq=return_vq)
        x = x.permute(0, 2, 3, 1)

        return x, loss
