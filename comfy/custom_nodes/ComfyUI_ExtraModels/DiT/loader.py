import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import torch
from comfy import model_management

class EXM_DiT(comfy.supported_models_base.BASE):
	unet_config = {}
	unet_extra_config = {}
	latent_format = comfy.latent_formats.SD15

	def __init__(self, model_conf):
		self.unet_config = model_conf.get("unet_config", {})
		self.sampling_settings = model_conf.get("sampling_settings", {})
		self.latent_format = self.latent_format()
		# UNET is handled by extension
		self.unet_config["disable_unet_model_creation"] = True

	def model_type(self, state_dict, prefix=""):
		return comfy.model_base.ModelType.EPS

def load_dit(model_path, model_conf):
	state_dict = comfy.utils.load_torch_file(model_path)
	state_dict = state_dict.get("model", state_dict)
	parameters = comfy.utils.calculate_parameters(state_dict)
	unet_dtype = model_management.unet_dtype(model_params=parameters)
	load_device = comfy.model_management.get_torch_device()
	offload_device = comfy.model_management.unet_offload_device()

	# ignore fp8/etc and use directly for now
	manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)
	if manual_cast_dtype:
		print(f"DiT: falling back to {manual_cast_dtype}")
		unet_dtype = manual_cast_dtype

	model_conf["unet_config"]["num_classes"] = state_dict["y_embedder.embedding_table.weight"].shape[0] - 1 # adj. for empty

	model_conf = EXM_DiT(model_conf)
	model = comfy.model_base.BaseModel(
		model_conf,
		model_type=comfy.model_base.ModelType.EPS,
		device=model_management.get_torch_device()
	)

	from .model import DiT
	model.diffusion_model = DiT(**model_conf.unet_config)

	model.diffusion_model.load_state_dict(state_dict)
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
