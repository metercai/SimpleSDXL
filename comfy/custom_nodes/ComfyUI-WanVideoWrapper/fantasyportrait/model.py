import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class MultiProjModel(nn.Module):
    def __init__(self, adapter_in_dim=1024, cross_attention_dim=1024):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.eye_proj = torch.nn.Linear(6, cross_attention_dim, bias=False)
        self.emo_proj = torch.nn.Linear(30, cross_attention_dim, bias=False)
        self.mouth_proj = torch.nn.Linear(512, cross_attention_dim, bias=False)
        self.headpose_proj = torch.nn.Linear(6, cross_attention_dim, bias=False)

        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, adapter_embeds):
        B, num_frames, C = adapter_embeds.shape
        embeds = adapter_embeds
        split_sizes = [6, 6, 30, 512]
        headpose, eye, emo, mouth = torch.split(embeds, split_sizes, dim=-1)
        headpose = self.norm(self.headpose_proj(headpose))
        eye = self.norm(self.eye_proj(eye))
        emo = self.norm(self.emo_proj(emo))
        mouth = self.norm(self.mouth_proj(mouth))

        all_features = torch.stack([headpose, eye, emo, mouth], dim=2)
        result_final = all_features.view(B, num_frames * 4, self.cross_attention_dim)

        return result_final

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(
            -2, -1
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):  # x  (b, 512, 1)
        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)  # (b, 512, 1024)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents  # b 16 1024
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class PortraitAdapter(nn.Module):
    def __init__(self, adapter_in_dim: int, adapter_proj_dim: int, dtype: torch.dtype):
        super().__init__()

        self.adapter_in_dim = adapter_in_dim
        self.adapter_proj_dim = adapter_proj_dim
        self.proj_model = self.init_proj(self.adapter_proj_dim)
        self.dtype = dtype

        self.mouth_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=16,
            embedding_dim=512,
            output_dim=2048,
            ff_mult=4,
        )

        self.emo_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=4,
            embedding_dim=30,
            output_dim=2048,
            ff_mult=4,
        )

    def init_proj(self, cross_attention_dim=5120):
        proj_model = MultiProjModel(
            adapter_in_dim=self.adapter_in_dim, cross_attention_dim=cross_attention_dim
        )
        return proj_model

    def get_adapter_proj(self, adapter_fea=None, adapter_scale=1.0, mouth_scale=1.0, emo_scale=1.0):
        split_sizes = [6, 6, 30, 512]
        headpose, eye, emo, mouth = torch.split(
            adapter_fea, split_sizes, dim=-1
        )
        B, frames, dim = mouth.shape
        mouth = mouth.view(B * frames, 1, 512)
        emo = emo.view(B * frames, 1, 30)

        mouth_fea = self.mouth_proj_model(mouth) * mouth_scale
        emo_fea = self.emo_proj_model(emo) * emo_scale

        mouth_fea = mouth_fea.view(B, frames, 16, 2048)
        emo_fea = emo_fea.view(B, frames, 4, 2048)

        adapter_fea = self.proj_model(adapter_fea) * adapter_scale

        adapter_fea = adapter_fea.view(B, frames, 4, 2048)

        all_fea = torch.cat([adapter_fea, mouth_fea, emo_fea], dim=2)

        result_final = all_fea.view(B, frames * 24, 2048)

        return result_final
    
    
    def split_audio_adapter_sequence(self, adapter_proj_length, num_frames=80):
        tokens_pre_frame = adapter_proj_length / num_frames
        tokens_pre_latents_frame = tokens_pre_frame * 4
        half_tokens_pre_latents_frame = tokens_pre_latents_frame / 2
        pos_idx = []
        for i in range(int((num_frames - 1) / 4) + 1):
            if i == 0:
                pos_idx.append(0)
            else:
                begin_token_id = tokens_pre_frame * ((i - 1) * 4 + 1)
                end_token_id = tokens_pre_frame * (i * 4 + 1)
                pos_idx.append(int((sum([begin_token_id, end_token_id]) / 2)) - 1)
        pos_idx_range = [
            [
                idx - int(half_tokens_pre_latents_frame),
                idx + int(half_tokens_pre_latents_frame),
            ]
            for idx in pos_idx
        ]
        pos_idx_range[0] = [
            -(int(half_tokens_pre_latents_frame) * 2 - pos_idx_range[1][0]),
            pos_idx_range[1][0],
        ]
        return pos_idx_range
    
    
    def split_tensor_with_padding(self, input_tensor, pos_idx_range, expand_length=0):
        pos_idx_range = [
            [idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_range
        ]
        sub_sequences = []
        seq_len = input_tensor.size(1)
        max_valid_idx = seq_len - 1
        k_lens_list = []
        for start, end in pos_idx_range:
            pad_front = max(-start, 0)
            pad_back = max(end - max_valid_idx, 0)

            valid_start = max(start, 0)
            valid_end = min(end, max_valid_idx)

            if valid_start <= valid_end:
                valid_part = input_tensor[:, valid_start : valid_end + 1, :]
            else:
                valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2)))

            padded_subseq = F.pad(
                valid_part,
                (0, 0, 0, pad_back + pad_front, 0, 0),
                mode="constant",
                value=0,
            )
            k_lens_list.append(padded_subseq.size(-2) - pad_back - pad_front)

            sub_sequences.append(padded_subseq)
        return torch.stack(sub_sequences, dim=1), torch.tensor(
            k_lens_list, dtype=torch.long
        )