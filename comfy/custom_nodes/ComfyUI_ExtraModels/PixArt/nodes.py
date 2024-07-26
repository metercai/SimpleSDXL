import os
import json
import torch
import folder_paths

from comfy import utils
from .conf import pixart_conf, pixart_res
from .lora import load_pixart_lora
from .loader import load_pixart

class PixArtCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
				"model": (list(pixart_conf.keys()),),
			}
		}
	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt Checkpoint Loader"

	def load_checkpoint(self, ckpt_name, model):
		ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
		model_conf = pixart_conf[model]
		model = load_pixart(
			model_path = ckpt_path,
			model_conf = model_conf,
		)
		return (model,)

class PixArtCheckpointLoaderSimple(PixArtCheckpointLoader):
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
			}
		}
	TITLE = "PixArt Checkpoint Loader (auto)"

	def load_checkpoint(self, ckpt_name):
		ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
		model = load_pixart(model_path=ckpt_path)
		return (model,)

class PixArtResolutionSelect():
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": (list(pixart_res.keys()),),
				# keys are the same for both
				"ratio": (list(pixart_res["PixArtMS_XL_2"].keys()),{"default":"1.00"}),
			}
		}
	RETURN_TYPES = ("INT","INT")
	RETURN_NAMES = ("width","height")
	FUNCTION = "get_res"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt Resolution Select"

	def get_res(self, model, ratio):
		width, height = pixart_res[model][ratio]
		return (width,height)

class PixArtLoraLoader:
	def __init__(self):
		self.loaded_lora = None

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("MODEL",),
				"lora_name": (folder_paths.get_filename_list("loras"), ),
				"strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
			}
		}
	RETURN_TYPES = ("MODEL",)
	FUNCTION = "load_lora"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt Load LoRA"

	def load_lora(self, model, lora_name, strength,):
		if strength == 0:
			return (model)

		lora_path = folder_paths.get_full_path("loras", lora_name)
		lora = None
		if self.loaded_lora is not None:
			if self.loaded_lora[0] == lora_path:
				lora = self.loaded_lora[1]
			else:
				temp = self.loaded_lora
				self.loaded_lora = None
				del temp

		if lora is None:
			lora = utils.load_torch_file(lora_path, safe_load=True)
			self.loaded_lora = (lora_path, lora)

		model_lora = load_pixart_lora(model, lora, lora_path, strength,)
		return (model_lora,)

class PixArtResolutionCond:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"cond": ("CONDITIONING", ),
				"width": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
				"height": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	RETURN_NAMES = ("cond",)
	FUNCTION = "add_cond"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt Resolution Conditioning"
	
	def add_cond(self, cond, width, height):
		for c in range(len(cond)):
			cond[c][1].update({
				"img_hw": [[height, width]],
				"aspect_ratio": [[height/width]],
			})
		return (cond,)

class PixArtControlNetCond:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"cond": ("CONDITIONING",),
				"latent": ("LATENT",),
				# "image": ("IMAGE",),
				# "vae": ("VAE",),
				# "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	RETURN_NAMES = ("cond",)
	FUNCTION = "add_cond"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt ControlNet Conditioning"

	def add_cond(self, cond, latent):
		for c in range(len(cond)):
			cond[c][1]["cn_hint"] = latent["samples"] * 0.18215
		return (cond,)

class PixArtT5TextEncode:
	"""
	Reference code, mostly to verify compatibility.
	 Once everything works, this should instead inherit from the
	 T5 text encode node and simply add the extra conds (res/ar).
	"""
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"text": ("STRING", {"multiline": True}),
				"T5": ("T5",),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	FUNCTION = "encode"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt T5 Text Encode [Reference]"

	def mask_feature(self, emb, mask):
		if emb.shape[0] == 1:
			keep_index = mask.sum().item()
			return emb[:, :, :keep_index, :], keep_index
		else:
			masked_feature = emb * mask[:, None, :, None]
			return masked_feature, emb.shape[2]

	def encode(self, text, T5):
		text = text.lower().strip()
		tokenizer_out = T5.tokenizer.tokenizer(
			text,
			max_length            = 120,
			padding               = 'max_length',
			truncation            = True,
			return_attention_mask = True,
			add_special_tokens    = True,
			return_tensors        = 'pt'
		)
		tokens = tokenizer_out["input_ids"]
		mask = tokenizer_out["attention_mask"]
		embs = T5.cond_stage_model.transformer(
			input_ids      = tokens.to(T5.load_device),
			attention_mask = mask.to(T5.load_device),
		)['last_hidden_state'].float()[:, None]
		masked_embs, keep_index = self.mask_feature(
			embs.detach().to("cpu"),
			mask.detach().to("cpu")
		)
		masked_embs = masked_embs.squeeze(0) # match CLIP/internal
		print("Encoded T5:", masked_embs.shape)
		return ([[masked_embs, {}]], )

class PixArtT5FromSD3CLIP:
	"""
	Split the T5 text encoder away from SD3
	"""
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"sd3_clip": ("CLIP",),
				"padding": ("INT", {"default": 1, "min": 1, "max": 300}),
			}
		}

	RETURN_TYPES = ("CLIP",)
	RETURN_NAMES = ("t5",)
	FUNCTION = "split"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt T5 from SD3 CLIP"

	def split(self, sd3_clip, padding):
		try:
			from comfy.text_encoders.sd3_clip import SD3Tokenizer, SD3ClipModel
		except ImportError:
			# fallback for older ComfyUI versions
			from comfy.sd3_clip import SD3Tokenizer, SD3ClipModel
		import copy
	
		clip = sd3_clip.clone()
		assert clip.cond_stage_model.t5xxl is not None, "CLIP must have T5 loaded!"

		# remove transformer
		transformer = clip.cond_stage_model.t5xxl.transformer
		clip.cond_stage_model.t5xxl.transformer = None

		# clone object
		tmp = SD3ClipModel(clip_l=False, clip_g=False, t5=False)
		tmp.t5xxl = copy.deepcopy(clip.cond_stage_model.t5xxl)
		# put transformer back
		clip.cond_stage_model.t5xxl.transformer = transformer
		tmp.t5xxl.transformer = transformer

		# override special tokens
		tmp.t5xxl.special_tokens = copy.deepcopy(clip.cond_stage_model.t5xxl.special_tokens)
		tmp.t5xxl.special_tokens.pop("end") # make sure empty tokens match

		# tokenizer
		tok = SD3Tokenizer()
		tok.t5xxl.min_length = padding
		
		clip.cond_stage_model = tmp
		clip.tokenizer = tok

		return (clip, )

NODE_CLASS_MAPPINGS = {
	"PixArtCheckpointLoader" : PixArtCheckpointLoader,
	"PixArtCheckpointLoaderSimple" : PixArtCheckpointLoaderSimple,
	"PixArtResolutionSelect" : PixArtResolutionSelect,
	"PixArtLoraLoader" : PixArtLoraLoader,
	"PixArtT5TextEncode" : PixArtT5TextEncode,
	"PixArtResolutionCond" : PixArtResolutionCond,
	"PixArtControlNetCond" : PixArtControlNetCond,
	"PixArtT5FromSD3CLIP": PixArtT5FromSD3CLIP,
}
