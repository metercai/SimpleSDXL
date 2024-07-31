import math
import torch
import torch.nn.functional as F
import torch.nn as nn

"""
Code below ported from https://github.com/openai/consistencydecoder
"""

def _extract_into_tensor(arr, timesteps, broadcast_shape):
	# from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L895    """
	res = arr[timesteps.to(torch.int).cpu()].float().to(timesteps.device)
	dims_to_append = len(broadcast_shape) - len(res.shape)
	return res[(...,) + (None,) * dims_to_append]

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
	# from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L45
	betas = []
	for i in range(num_diffusion_timesteps):
		t1 = i / num_diffusion_timesteps
		t2 = (i + 1) / num_diffusion_timesteps
		betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
	return torch.tensor(betas)

class ConsistencyDecoder(torch.nn.Module):
	# From https://github.com/openai/consistencydecoder
	def __init__(self):
		super().__init__()
		self.model = ConvUNetVAE()
		self.n_distilled_steps = 64

		sigma_data = 0.5
		betas = betas_for_alpha_bar(
			1024, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
		)
		alphas = 1.0 - betas
		alphas_cumprod = torch.cumprod(alphas, dim=0)
		self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
		sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
		sigmas = torch.sqrt(1.0 / alphas_cumprod - 1)
		self.c_skip = (
			sqrt_recip_alphas_cumprod
			* sigma_data**2
			/ (sigmas**2 + sigma_data**2)
		)
		self.c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5
		self.c_in = sqrt_recip_alphas_cumprod / (sigmas**2 + sigma_data**2) ** 0.5

	@staticmethod
	def round_timesteps(timesteps, total_timesteps, n_distilled_steps, truncate_start=True):
		with torch.no_grad():
			space = torch.div(total_timesteps, n_distilled_steps, rounding_mode="floor")
			rounded_timesteps = (
				torch.div(timesteps, space, rounding_mode="floor") + 1
			) * space
			if truncate_start:
				rounded_timesteps[rounded_timesteps == total_timesteps] -= space
			else:
				rounded_timesteps[rounded_timesteps == total_timesteps] -= space
				rounded_timesteps[rounded_timesteps == 0] += space
			return rounded_timesteps

	@staticmethod
	def ldm_transform_latent(z, extra_scale_factor=1):
		channel_means = [0.38862467, 0.02253063, 0.07381133, -0.0171294]
		channel_stds = [0.9654121, 1.0440036, 0.76147926, 0.77022034]

		if len(z.shape) != 4:
			raise ValueError()

		z = z * 0.18215
		channels = [z[:, i] for i in range(z.shape[1])]

		channels = [
			extra_scale_factor * (c - channel_means[i]) / channel_stds[i]
			for i, c in enumerate(channels)
		]
		return torch.stack(channels, dim=1)

	@torch.no_grad()
	def decode(self, features: torch.Tensor, schedule=[1.0, 0.5]):
		features = self.ldm_transform_latent(features)
		ts = self.round_timesteps(
			torch.arange(0, 1024),
			1024,
			self.n_distilled_steps,
			truncate_start=False,
		)
		shape = (
			features.size(0),
			3,
			8 * features.size(2),
			8 * features.size(3),
		)
		x_start = torch.zeros(shape, device=features.device, dtype=features.dtype)
		schedule_timesteps = [int((1024 - 1) * s) for s in schedule]
		for i in schedule_timesteps:
			t = ts[i].item()
			t_ = torch.tensor([t] * features.shape[0], device=features.device)
			noise = torch.randn_like(x_start, device=features.device)
			x_start = (
				_extract_into_tensor(self.sqrt_alphas_cumprod, t_, x_start.shape)
				* x_start
				+ _extract_into_tensor(
					self.sqrt_one_minus_alphas_cumprod, t_, x_start.shape
				)
				* noise
			)
			c_in = _extract_into_tensor(self.c_in, t_, x_start.shape)
			model_output = self.model((c_in * x_start).to(features.dtype), t_, features=features)
			B, C = x_start.shape[:2]
			model_output, _ = torch.split(model_output, C, dim=1)
			pred_xstart = (
				_extract_into_tensor(self.c_out, t_, x_start.shape) * model_output
				+ _extract_into_tensor(self.c_skip, t_, x_start.shape) * x_start
			).clamp(-1, 1)
			x_start = pred_xstart
		return x_start

	def encode(self, *args, **kwargs):
		raise NotImplementedError("ConsistencyDecoder can't be used for encoding!")

"""
Model definitions ported from:
https://gist.github.com/madebyollin/865fa6a18d9099351ddbdfbe7299ccbf
https://gist.github.com/mrsteyk/74ad3ec2f6f823111ae4c90e168505ac.
"""

class TimestepEmbedding(nn.Module):
	def __init__(self, n_time=1024, n_emb=320, n_out=1280) -> None:
		super().__init__()
		self.emb = nn.Embedding(n_time, n_emb)
		self.f_1 = nn.Linear(n_emb, n_out)
		self.f_2 = nn.Linear(n_out, n_out)

	def forward(self, x) -> torch.Tensor:
		x = self.emb(x)
		x = self.f_1(x)
		x = F.silu(x)
		return self.f_2(x)


class ImageEmbedding(nn.Module):
	def __init__(self, in_channels=7, out_channels=320) -> None:
		super().__init__()
		self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

	def forward(self, x) -> torch.Tensor:
		return self.f(x)


class ImageUnembedding(nn.Module):
	def __init__(self, in_channels=320, out_channels=6) -> None:
		super().__init__()
		self.gn = nn.GroupNorm(32, in_channels)
		self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

	def forward(self, x) -> torch.Tensor:
		return self.f(F.silu(self.gn(x)))


class ConvResblock(nn.Module):
	def __init__(self, in_features=320, out_features=320) -> None:
		super().__init__()
		self.f_t = nn.Linear(1280, out_features * 2)

		self.gn_1 = nn.GroupNorm(32, in_features)
		self.f_1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

		self.gn_2 = nn.GroupNorm(32, out_features)
		self.f_2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)

		skip_conv = in_features != out_features
		self.f_s = (
			nn.Conv2d(in_features, out_features, kernel_size=1, padding=0)
			if skip_conv
			else nn.Identity()
		)

	def forward(self, x, t):
		x_skip = x
		t = self.f_t(F.silu(t))
		t = t.chunk(2, dim=1)
		t_1 = t[0].unsqueeze(dim=2).unsqueeze(dim=3) + 1
		t_2 = t[1].unsqueeze(dim=2).unsqueeze(dim=3)

		gn_1 = F.silu(self.gn_1(x))
		f_1 = self.f_1(gn_1)

		gn_2 = self.gn_2(f_1)

		return self.f_s(x_skip) + self.f_2(F.silu(gn_2 * t_1 + t_2))


# Also ConvResblock
class Downsample(nn.Module):
	def __init__(self, in_channels=320) -> None:
		super().__init__()
		self.f_t = nn.Linear(1280, in_channels * 2)

		self.gn_1 = nn.GroupNorm(32, in_channels)
		self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
		self.gn_2 = nn.GroupNorm(32, in_channels)

		self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

	def forward(self, x, t) -> torch.Tensor:
		x_skip = x

		t = self.f_t(F.silu(t))
		t_1, t_2 = t.chunk(2, dim=1)
		t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
		t_2 = t_2.unsqueeze(2).unsqueeze(3)

		gn_1 = F.silu(self.gn_1(x))
		avg_pool2d = F.avg_pool2d(gn_1, kernel_size=(2, 2), stride=None)
		f_1 = self.f_1(avg_pool2d)
		gn_2 = self.gn_2(f_1)

		f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

		return f_2 + F.avg_pool2d(x_skip, kernel_size=(2, 2), stride=None)


# Also ConvResblock
class Upsample(nn.Module):
	def __init__(self, in_channels=1024) -> None:
		super().__init__()
		self.f_t = nn.Linear(1280, in_channels * 2)

		self.gn_1 = nn.GroupNorm(32, in_channels)
		self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
		self.gn_2 = nn.GroupNorm(32, in_channels)

		self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

	def forward(self, x, t) -> torch.Tensor:
		x_skip = x

		t = self.f_t(F.silu(t))
		t_1, t_2 = t.chunk(2, dim=1)
		t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
		t_2 = t_2.unsqueeze(2).unsqueeze(3)

		gn_1 = F.silu(self.gn_1(x))
		upsample = F.interpolate(gn_1.float(), scale_factor=2, mode="nearest").to(gn_1.dtype)
		
		f_1 = self.f_1(upsample)
		gn_2 = self.gn_2(f_1)

		f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

		return f_2 + F.interpolate(x_skip.float(), scale_factor=2, mode="nearest").to(x_skip.dtype)


class ConvUNetVAE(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.embed_image = ImageEmbedding()
		self.embed_time = TimestepEmbedding()

		down_0 = nn.ModuleList(
			[
				ConvResblock(320, 320),
				ConvResblock(320, 320),
				ConvResblock(320, 320),
				Downsample(320),
			]
		)
		down_1 = nn.ModuleList(
			[
				ConvResblock(320, 640),
				ConvResblock(640, 640),
				ConvResblock(640, 640),
				Downsample(640),
			]
		)
		down_2 = nn.ModuleList(
			[
				ConvResblock(640, 1024),
				ConvResblock(1024, 1024),
				ConvResblock(1024, 1024),
				Downsample(1024),
			]
		)
		down_3 = nn.ModuleList(
			[
				ConvResblock(1024, 1024),
				ConvResblock(1024, 1024),
				ConvResblock(1024, 1024),
			]
		)
		self.down = nn.ModuleList(
			[
				down_0,
				down_1,
				down_2,
				down_3,
			]
		)

		self.mid = nn.ModuleList(
			[
				ConvResblock(1024, 1024),
				ConvResblock(1024, 1024),
			]
		)

		up_3 = nn.ModuleList(
			[
				ConvResblock(1024 * 2, 1024),
				ConvResblock(1024 * 2, 1024),
				ConvResblock(1024 * 2, 1024),
				ConvResblock(1024 * 2, 1024),
				Upsample(1024),
			]
		)
		up_2 = nn.ModuleList(
			[
				ConvResblock(1024 * 2, 1024),
				ConvResblock(1024 * 2, 1024),
				ConvResblock(1024 * 2, 1024),
				ConvResblock(1024 + 640, 1024),
				Upsample(1024),
			]
		)
		up_1 = nn.ModuleList(
			[
				ConvResblock(1024 + 640, 640),
				ConvResblock(640 * 2, 640),
				ConvResblock(640 * 2, 640),
				ConvResblock(320 + 640, 640),
				Upsample(640),
			]
		)
		up_0 = nn.ModuleList(
			[
				ConvResblock(320 + 640, 320),
				ConvResblock(320 * 2, 320),
				ConvResblock(320 * 2, 320),
				ConvResblock(320 * 2, 320),
			]
		)
		self.up = nn.ModuleList(
			[
				up_0,
				up_1,
				up_2,
				up_3,
			]
		)

		self.output = ImageUnembedding()

	def forward(self, x, t, features) -> torch.Tensor:
		x = torch.cat([x, F.interpolate(features.float(),scale_factor=8,mode="nearest").to(features.dtype)], dim=1)
		t = self.embed_time(t)
		x = self.embed_image(x)

		skips = [x]
		for down in self.down:
			for block in down:
				x = block(x, t)
				skips.append(x)

		for i in range(2):
			x = self.mid[i](x, t)

		for up in self.up[::-1]:
			for block in up:
				if isinstance(block, ConvResblock):
					x = torch.concat([x, skips.pop()], dim=1)
				x = block(x, t)

		return self.output(x)
