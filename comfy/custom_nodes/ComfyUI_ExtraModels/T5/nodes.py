import os
import json
import torch
import folder_paths

from .loader import load_t5
from ..utils.dtype import string_to_dtype

# initialize custom folder path
os.makedirs(
	os.path.join(folder_paths.models_dir,"t5"),
	exist_ok = True,
)
folder_paths.folder_names_and_paths["t5"] = (
	[
		os.path.join(folder_paths.models_dir,"t5"),
		*folder_paths.folder_names_and_paths.get("t5", [[],set()])[0]
	],
	folder_paths.supported_pt_extensions
)

dtypes = [
	"default",
	"auto (comfy)",
	"FP32",
	"FP16",
	# Note: remove these at some point
	"bnb8bit",
	"bnb4bit",
]
try: torch.float8_e5m2
except AttributeError: print("Torch version too old for FP8")
else: dtypes += ["FP8 E4M3", "FP8 E5M2"]

class T5v11Loader:
	@classmethod
	def INPUT_TYPES(s):
		devices = ["auto", "cpu", "gpu"]
		# hack for using second GPU as offload
		for k in range(1, torch.cuda.device_count()):
			devices.append(f"cuda:{k}")
		return {
			"required": {
				"t5v11_name": (folder_paths.get_filename_list("t5"),),
				"t5v11_ver": (["xxl"],),
				"path_type": (["folder", "file"],),
				"device": (devices, {"default":"cpu"}),
				"dtype": (dtypes,),
			}
		}
	RETURN_TYPES = ("T5",)
	FUNCTION = "load_model"
	CATEGORY = "ExtraModels/T5"
	TITLE = "T5v1.1 Loader"

	def load_model(self, t5v11_name, t5v11_ver, path_type, device, dtype):
		if "bnb" in dtype:
			assert device == "gpu" or device.startswith("cuda"), "BitsAndBytes only works on CUDA! Set device to 'gpu'."
		dtype = string_to_dtype(dtype, "text_encoder")
		if device == "cpu":
			assert dtype in [None, torch.float32], f"Can't use dtype '{dtype}' with CPU! Set dtype to 'default'."

		return (load_t5(
			model_type = "t5v11",
			model_ver  = t5v11_ver,
			model_path = folder_paths.get_full_path("t5", t5v11_name),
			path_type  = path_type,
			device     = device,
			dtype      = dtype,
		),)

class T5TextEncode:
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
	CATEGORY = "ExtraModels/T5"
	TITLE = "T5 Text Encode"

	def encode(self, text, T5=None):
		tokens = T5.tokenize(text)
		cond = T5.encode_from_tokens(tokens)
		return ([[cond, {}]], )

NODE_CLASS_MAPPINGS = {
	"T5v11Loader"  : T5v11Loader,
	"T5TextEncode" : T5TextEncode,
}
