import os
import torch
import comfy.utils
import comfy.model_patcher
from comfy import model_management
import folder_paths

from .t5v11 import T5v11Model, T5v11Tokenizer

class EXM_T5v11:
	def __init__(self, textmodel_ver="xxl", embedding_directory=None, textmodel_path=None, no_init=False, device="cpu", dtype=None):
		if no_init:
			return

		if device == "auto":
			size = 0
			self.load_device = model_management.text_encoder_device()
			self.offload_device = model_management.text_encoder_offload_device()
			self.init_device = "cpu"
		elif dtype == "bnb8bit":
			# BNB doesn't support size enum
			size = 12.4 * (1024**3)
			# Or moving between devices
			self.load_device = model_management.get_torch_device()
			self.offload_device = self.load_device
			self.init_device = self.load_device
		elif dtype == "bnb4bit":
			# This seems to use the same VRAM as 8bit on Pascal?
			size = 6.2 * (1024**3)
			self.load_device = model_management.get_torch_device()
			self.offload_device = self.load_device
			self.init_device = self.load_device
		elif device == "cpu":
			size = 0
			self.load_device = "cpu"
			self.offload_device = "cpu"
			self.init_device="cpu"
		elif device.startswith("cuda"):
			print("Direct CUDA device override!\nVRAM will not be freed by default.")
			size = 0
			self.load_device = device
			self.offload_device = device
			self.init_device = device
		else:
			size = 0
			self.load_device = model_management.get_torch_device()
			self.offload_device = "cpu"
			self.init_device="cpu"

		self.cond_stage_model = T5v11Model(
			textmodel_ver  = textmodel_ver,
			textmodel_path = textmodel_path,
			device         = device,
			dtype          = dtype,
		)
		self.tokenizer = T5v11Tokenizer(embedding_directory=embedding_directory)
		self.patcher = comfy.model_patcher.ModelPatcher(
			self.cond_stage_model,
			load_device    = self.load_device,
			offload_device = self.offload_device,
			current_device = self.load_device,
			size           = size,
		)

	def clone(self):
		n = T5(no_init=True)
		n.patcher = self.patcher.clone()
		n.cond_stage_model = self.cond_stage_model
		n.tokenizer = self.tokenizer
		return n

	def tokenize(self, text, return_word_ids=False):
		return self.tokenizer.tokenize_with_weights(text, return_word_ids)

	def encode_from_tokens(self, tokens):
		self.load_model()
		return self.cond_stage_model.encode_token_weights(tokens)

	def encode(self, text):
		tokens = self.tokenize(text)
		return self.encode_from_tokens(tokens)

	def load_sd(self, sd):
		return self.cond_stage_model.load_sd(sd)

	def get_sd(self):
		return self.cond_stage_model.state_dict()

	def load_model(self):
		if self.load_device != "cpu":
			model_management.load_model_gpu(self.patcher)
		return self.patcher

	def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
		return self.patcher.add_patches(patches, strength_patch, strength_model)

	def get_key_patches(self):
		return self.patcher.get_key_patches()


def load_t5(model_type, model_ver, model_path, path_type="file", device="cpu", dtype=None):
	assert model_type in ["t5v11"] # Only supported model for now
	model_args = {
		"textmodel_ver" : model_ver,
		"device" : device,
		"dtype"  : dtype,
	}

	if path_type == "folder":
		# pass directly to transformers and initialize there
		# this is to avoid having to handle multi-file state dict loading for now.
		model_args["textmodel_path"] = os.path.dirname(model_path)
		return EXM_T5v11(**model_args)
	else:
		# for some reason this returns garbage with torch.int8 weights, or just OOMs
		model = EXM_T5v11(**model_args)
		sd = comfy.utils.load_torch_file(model_path)
		model.load_sd(sd)
		return model
