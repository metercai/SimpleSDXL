import os
import json
import torch
import folder_paths

from .conf import dit_conf
from .loader import load_dit

class DitCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
				"model": (list(dit_conf.keys()),),
				"image_size": ([256, 512],),
				# "num_classes": ("INT", {"default": 1000, "min": 0,}),
			}
		}
	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "ExtraModels/DiT"
	TITLE = "DitCheckpointLoader"

	def load_checkpoint(self, ckpt_name, model, image_size):
		ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
		model_conf = dit_conf[model]
		model_conf["unet_config"]["input_size"]  = image_size // 8
		# model_conf["unet_config"]["num_classes"] = num_classes
		dit = load_dit(
			model_path = ckpt_path,
			model_conf = model_conf,
		)
		return (dit,)

# todo: this needs frontend code to display properly
def get_label_data(label_file="labels/imagenet1000.json"):
	label_path = os.path.join(
		os.path.dirname(os.path.realpath(__file__)),
		label_file,
	)
	label_data = {0: "None"}
	with open(label_path, "r") as f:
		label_data = json.loads(f.read())
	return label_data
label_data = get_label_data()

class DiTCondLabelSelect:
	@classmethod
	def INPUT_TYPES(s):
		global label_data
		return {
			"required": {
				"model" : ("MODEL",),
				"label_name": (list(label_data.values()),),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	RETURN_NAMES = ("class",)
	FUNCTION = "cond_label"
	CATEGORY = "ExtraModels/DiT"
	TITLE = "DiTCondLabelSelect"

	def cond_label(self, model, label_name):
		global label_data
		class_labels = [int(k) for k,v in label_data.items() if v == label_name]
		y = torch.tensor([[class_labels[0]]]).to(torch.int)
		return ([[y, {}]], )

class DiTCondLabelEmpty:
	@classmethod
	def INPUT_TYPES(s):
		global label_data
		return {
			"required": {
				"model" : ("MODEL",),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	RETURN_NAMES = ("empty",)
	FUNCTION = "cond_empty"
	CATEGORY = "ExtraModels/DiT"
	TITLE = "DiTCondLabelEmpty"

	def cond_empty(self, model):
		# [ID of last class + 1] == [num_classes]
		y_null = model.model.model_config.unet_config["num_classes"]
		y = torch.tensor([[y_null]]).to(torch.int)
		return ([[y, {}]], )

NODE_CLASS_MAPPINGS = {
	"DitCheckpointLoader" : DitCheckpointLoader,
	"DiTCondLabelSelect"  : DiTCondLabelSelect,
	"DiTCondLabelEmpty"   : DiTCondLabelEmpty,
}
