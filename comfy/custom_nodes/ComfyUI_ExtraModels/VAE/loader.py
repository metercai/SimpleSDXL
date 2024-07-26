import torch
import comfy.sd
import comfy.utils
from comfy import model_management
from comfy import diffusers_convert

class EXVAE(comfy.sd.VAE):
	def __init__(self, model_path, model_conf, dtype=torch.float32):
		self.latent_dim   = model_conf["embed_dim"]
		self.latent_scale = model_conf["embed_scale"]
		self.device = model_management.vae_device()
		self.offload_device = model_management.vae_offload_device()
		self.vae_dtype = dtype

		sd = comfy.utils.load_torch_file(model_path)
		model = None
		if model_conf["type"] == "AutoencoderKL":
			from .models.kl import AutoencoderKL
			model = AutoencoderKL(config=model_conf)
			if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys():
				sd = diffusers_convert.convert_vae_state_dict(sd)
		elif model_conf["type"] == "AutoencoderKL-VideoDecoder":
			from .models.temporal_ae import AutoencoderKL
			model = AutoencoderKL(config=model_conf)
		elif model_conf["type"] == "VQModel":
			from .models.vq import VQModel
			model = VQModel(config=model_conf)
		elif model_conf["type"] == "ConsistencyDecoder":
			from .models.consistencydecoder import ConsistencyDecoder
			model = ConsistencyDecoder()
			sd = {f"model.{k}":v for k,v in sd.items()}
		elif model_conf["type"] == "MoVQ3":
			from .models.movq3 import MoVQ
			model = MoVQ(model_conf)
		else:
			raise NotImplementedError(f"Unknown VAE type '{model_conf['type']}'")

		self.first_stage_model = model.eval()
		m, u = self.first_stage_model.load_state_dict(sd, strict=False)
		if len(m) > 0: print("Missing VAE keys", m)
		if len(u) > 0: print("Leftover VAE keys", u)

		self.first_stage_model.to(self.vae_dtype).to(self.offload_device)

	### Encode/Decode functions below needed due to source repo having 4 VAE channels and a scale factor of 8 hardcoded
	def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap = 16):
		steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
		steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
		steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
		pbar = comfy.utils.ProgressBar(steps)

		decode_fn = lambda a: (self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)) + 1.0).float()
		output = torch.clamp((
			(comfy.utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = self.latent_scale, pbar = pbar) +
			comfy.utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = self.latent_scale, pbar = pbar) +
			 comfy.utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount = self.latent_scale, pbar = pbar))
			/ 3.0) / 2.0, min=0.0, max=1.0)
		return output

	def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap = 64):
		steps = pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
		steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
		steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)
		pbar = comfy.utils.ProgressBar(steps)

		encode_fn = lambda a: self.first_stage_model.encode((2. * a - 1.).to(self.vae_dtype).to(self.device)).float()
		samples = comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount = (1/self.latent_scale), out_channels=self.latent_dim, pbar=pbar)
		samples += comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = (1/self.latent_scale), out_channels=self.latent_dim, pbar=pbar)
		samples += comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = (1/self.latent_scale), out_channels=self.latent_dim, pbar=pbar)
		samples /= 3.0
		return samples

	def decode(self, samples_in):
		self.first_stage_model = self.first_stage_model.to(self.device)
		try:
			memory_used = (2562 * samples_in.shape[2] * samples_in.shape[3] * 64) * 1.7
			model_management.free_memory(memory_used, self.device)
			free_memory = model_management.get_free_memory(self.device)
			batch_number = int(free_memory / memory_used)
			batch_number = max(1, batch_number)

			pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * self.latent_scale), round(samples_in.shape[3] * self.latent_scale)), device="cpu")
			for x in range(0, samples_in.shape[0], batch_number):
				samples = samples_in[x:x+batch_number].to(self.vae_dtype).to(self.device)
				pixel_samples[x:x+batch_number] = torch.clamp((self.first_stage_model.decode(samples).cpu().float() + 1.0) / 2.0, min=0.0, max=1.0)
		except model_management.OOM_EXCEPTION as e:
			print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
			pixel_samples = self.decode_tiled_(samples_in)

		self.first_stage_model = self.first_stage_model.to(self.offload_device)
		pixel_samples = pixel_samples.cpu().movedim(1,-1)
		return pixel_samples

	def encode(self, pixel_samples):
		self.first_stage_model = self.first_stage_model.to(self.device)
		pixel_samples = pixel_samples.movedim(-1,1)
		try:
			memory_used = (2078 * pixel_samples.shape[2] * pixel_samples.shape[3]) * 1.7 #NOTE: this constant along with the one in the decode above are estimated from the mem usage for the VAE and could change.
			model_management.free_memory(memory_used, self.device)
			free_memory = model_management.get_free_memory(self.device)
			batch_number = int(free_memory / memory_used)
			batch_number = max(1, batch_number)
			samples = torch.empty((pixel_samples.shape[0], self.latent_dim, round(pixel_samples.shape[2] // self.latent_scale), round(pixel_samples.shape[3] // self.latent_scale)), device="cpu")
			for x in range(0, pixel_samples.shape[0], batch_number):
				pixels_in = (2. * pixel_samples[x:x+batch_number] - 1.).to(self.vae_dtype).to(self.device)
				samples[x:x+batch_number] = self.first_stage_model.encode(pixels_in).cpu().float()

		except model_management.OOM_EXCEPTION as e:
			print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
			samples = self.encode_tiled_(pixel_samples)

		self.first_stage_model = self.first_stage_model.to(self.offload_device)
		return samples
