import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import comfy.conds
import torch
import math 
from comfy import model_management
from .diffusers_convert import convert_state_dict

class EXM_PixArt(comfy.supported_models_base.BASE):
	unet_config = {}
	unet_extra_config = {}
	latent_format = comfy.latent_formats.SD15

	def __init__(self, model_conf):
		self.model_target = model_conf.get("target")
		self.unet_config = model_conf.get("unet_config", {})
		self.sampling_settings = model_conf.get("sampling_settings", {})
		self.latent_format = self.latent_format()
		# UNET is handled by extension
		self.unet_config["disable_unet_model_creation"] = True

	def model_type(self, state_dict, prefix=""):
		return comfy.model_base.ModelType.EPS

class EXM_PixArt_Model(comfy.model_base.BaseModel):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def extra_conds(self, **kwargs):
		out = super().extra_conds(**kwargs)

		img_hw = kwargs.get("img_hw", None)
		if img_hw is not None:
			out["img_hw"] = comfy.conds.CONDRegular(torch.tensor(img_hw))
		
		aspect_ratio = kwargs.get("aspect_ratio", None)
		if aspect_ratio is not None:
			out["aspect_ratio"] = comfy.conds.CONDRegular(torch.tensor(aspect_ratio))

		cn_hint = kwargs.get("cn_hint", None)
		if cn_hint is not None:
			out["cn_hint"] = comfy.conds.CONDRegular(cn_hint)

		return out

def load_pixart(model_path, model_conf=None):
	state_dict = comfy.utils.load_torch_file(model_path)
	state_dict = state_dict.get("model", state_dict)

	# prefix
	for prefix in ["model.diffusion_model.",]:
		if any(True for x in state_dict if x.startswith(prefix)):
			state_dict = {k[len(prefix):]:v for k,v in state_dict.items()}

	# diffusers
	if "adaln_single.linear.weight" in state_dict:
		state_dict = convert_state_dict(state_dict) # Diffusers

	# guess auto config
	if model_conf is None:
		model_conf = guess_pixart_config(state_dict)

	parameters = comfy.utils.calculate_parameters(state_dict)
	unet_dtype = model_management.unet_dtype(model_params=parameters)
	load_device = comfy.model_management.get_torch_device()
	offload_device = comfy.model_management.unet_offload_device()

	# ignore fp8/etc and use directly for now
	manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)
	if manual_cast_dtype:
		print(f"PixArt: falling back to {manual_cast_dtype}")
		unet_dtype = manual_cast_dtype

	model_conf = EXM_PixArt(model_conf) # convert to object
	model = EXM_PixArt_Model( # same as comfy.model_base.BaseModel
		model_conf,
		model_type=comfy.model_base.ModelType.EPS,
		device=model_management.get_torch_device()
	)

	if model_conf.model_target == "PixArtMS":
		from .models.PixArtMS import PixArtMS
		model.diffusion_model = PixArtMS(**model_conf.unet_config)
	elif model_conf.model_target == "PixArt":
		from .models.PixArt import PixArt
		model.diffusion_model = PixArt(**model_conf.unet_config)
	elif model_conf.model_target == "PixArtMSSigma":
		from .models.PixArtMS import PixArtMS
		model.diffusion_model = PixArtMS(**model_conf.unet_config)
		model.latent_format = comfy.latent_formats.SDXL()
	elif model_conf.model_target == "ControlPixArtMSHalf":
		from .models.PixArtMS import PixArtMS
		from .models.pixart_controlnet import ControlPixArtMSHalf
		model.diffusion_model = PixArtMS(**model_conf.unet_config)
		model.diffusion_model = ControlPixArtMSHalf(model.diffusion_model)
	elif model_conf.model_target == "ControlPixArtHalf":
		from .models.PixArt import PixArt
		from .models.pixart_controlnet import ControlPixArtHalf
		model.diffusion_model = PixArt(**model_conf.unet_config)
		model.diffusion_model = ControlPixArtHalf(model.diffusion_model)
	else:
		raise NotImplementedError(f"Unknown model target '{model_conf.model_target}'")

	m, u = model.diffusion_model.load_state_dict(state_dict, strict=False)
	if len(m) > 0: print("Missing UNET keys", m)
	if len(u) > 0: print("Leftover UNET keys", u)
	model.diffusion_model.dtype = unet_dtype
	model.diffusion_model.eval()
	model.diffusion_model.to(unet_dtype)

	model_patcher = comfy.model_patcher.ModelPatcher(
		model,
		load_device = load_device,
		offload_device = offload_device,
		current_device = "cpu",
	)
	return model_patcher

def guess_pixart_config(sd):
	"""
	Guess config based on converted state dict.
	"""
	# Shared settings based on DiT_XL_2 - could be enumerated
	config = {
		"num_heads"   :   16, # get from attention
		"patch_size"  :    2, # final layer I guess?
		"hidden_size" : 1152, # pos_embed.shape[2]
	}
	config["depth"] = sum([key.endswith(".attn.proj.weight") for key in sd.keys()]) or 28

	try:
		# this is not present in the diffusers version for sigma?
		config["model_max_length"] = sd["y_embedder.y_embedding"].shape[0]
	except KeyError:
		# need better logic to guess this
		config["model_max_length"] = 300

	if "pos_embed" in sd:
		config["input_size"] = int(math.sqrt(sd["pos_embed"].shape[1])) * config["patch_size"]
		config["pe_interpolation"] = config["input_size"] // (512//8) # dumb guess

	target_arch = "PixArtMS"
	if config["model_max_length"] == 300:
		# Sigma
		target_arch = "PixArtMSSigma"
		config["micro_condition"] = False
		if "input_size" not in config:
			# The diffusers weights for 1K/2K are exactly the same...?
			# replace patch embed logic with HyDiT?
			print(f"PixArt: diffusers weights - 2K model will be broken, use manual loading!")
			config["input_size"] = 1024//8
	else:
		# Alpha
		if "csize_embedder.mlp.0.weight" in sd:
			# MS (microconds)
			target_arch = "PixArtMS"
			config["micro_condition"] = True
			if "input_size" not in config:
				config["input_size"] = 1024//8
				config["pe_interpolation"] = 2
		else:
			# PixArt
			target_arch = "PixArt"
			if "input_size" not in config:
				config["input_size"] = 512//8
				config["pe_interpolation"] = 1

	print("PixArt guessed config:", target_arch, config)
	return {
		"target": target_arch,
		"unet_config": config,
		"sampling_settings": {
			"beta_schedule" : "sqrt_linear",
			"linear_start"  : 0.0001,
			"linear_end"    : 0.02,
			"timesteps"     : 1000,
		}	
	}
