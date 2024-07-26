import os
import copy
import json
import torch
import comfy.lora
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from .diffusers_convert import convert_lora_state_dict

class EXM_PixArt_ModelPatcher(ModelPatcher):
	def calculate_weight(self, patches, weight, key):
		"""
		This is almost the same as the comfy function, but stripped down to just the LoRA patch code.
		The problem with the original code is the q/k/v keys being combined into one for the attention.
		In the diffusers code, they're treated as separate keys, but in the reference code they're recombined (q+kv|qkv).
		This means, for example, that the [1152,1152] weights become [3456,1152] in the state dict.
		The issue with this is that the LoRA weights are [128,1152],[1152,128] and become [384,1162],[3456,128] instead.
		
		This is the best thing I could think of that would fix that, but it's very fragile.
		 - Check key shape to determine if it needs the fallback logic
		 - Cut the input into parts based on the shape (undoing the torch.cat)
		 - Do the matrix multiplication logic
		 - Recombine them to match the expected shape
		"""
		for p in patches:
			alpha = p[0]
			v = p[1]
			strength_model = p[2]
			if strength_model != 1.0:
				weight *= strength_model
			
			if isinstance(v, list):
				v = (self.calculate_weight(v[1:], v[0].clone(), key), )

			if len(v) == 2:
				patch_type = v[0]
				v = v[1]

			if patch_type == "lora":
				mat1 = comfy.model_management.cast_to_device(v[0], weight.device, torch.float32)
				mat2 = comfy.model_management.cast_to_device(v[1], weight.device, torch.float32)
				if v[2] is not None:
					alpha *= v[2] / mat2.shape[0]
				try:
					mat1 = mat1.flatten(start_dim=1)
					mat2 = mat2.flatten(start_dim=1)

					ch1 = mat1.shape[0] // mat2.shape[1]
					ch2 = mat2.shape[0] // mat1.shape[1]
					### Fallback logic for shape mismatch ###
					if mat1.shape[0] != mat2.shape[1] and ch1 == ch2 and (mat1.shape[0]/mat2.shape[1])%1 == 0:
						mat1 = mat1.chunk(ch1, dim=0)
						mat2 = mat2.chunk(ch1, dim=0)
						weight += torch.cat(
							[alpha * torch.mm(mat1[x], mat2[x]) for x in range(ch1)],
							dim=0,
						).reshape(weight.shape).type(weight.dtype)
					else:
						weight += (alpha * torch.mm(mat1, mat2)).reshape(weight.shape).type(weight.dtype)
				except Exception as e:
					print("ERROR", key, e)
		return weight

	def clone(self):
		n = EXM_PixArt_ModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, weight_inplace_update=self.weight_inplace_update)
		n.patches = {}
		for k in self.patches:
			n.patches[k] = self.patches[k][:]

		n.object_patches = self.object_patches.copy()
		n.model_options = copy.deepcopy(self.model_options)
		n.model_keys = self.model_keys
		return n

def replace_model_patcher(model):
	n = EXM_PixArt_ModelPatcher(
		model                 = model.model,
		size                  = model.size,
		load_device           = model.load_device,
		offload_device        = model.offload_device,
		current_device        = model.current_device,
		weight_inplace_update = model.weight_inplace_update,
	)
	n.patches = {}
	for k in model.patches:
		n.patches[k] = model.patches[k][:]

	n.object_patches = model.object_patches.copy()
	n.model_options = copy.deepcopy(model.model_options)
	return n

def find_peft_alpha(path):
	def load_json(json_path):
		with open(json_path) as f:
			data = json.load(f)
		alpha = data.get("lora_alpha")
		alpha = alpha or data.get("alpha")
		if not alpha:
			print(" Found config but `lora_alpha` is missing!")
		else:
			print(f" Found config at {json_path} [alpha:{alpha}]")
		return alpha

	# For some weird reason peft doesn't include the alpha in the actual model
	print("PixArt: Warning! This is a PEFT LoRA. Trying to find config...")
	files = [
		f"{os.path.splitext(path)[0]}.json",
		f"{os.path.splitext(path)[0]}.config.json",
		os.path.join(os.path.dirname(path),"adapter_config.json"),
	]
	for file in files:
		if os.path.isfile(file):
			return load_json(file)

	print(" Missing config/alpha! assuming alpha of 8. Consider converting it/adding a config json to it.")
	return 8.0

def load_pixart_lora(model, lora, lora_path, strength):
	k_back = lambda x: x.replace(".lora_up.weight", "")
	# need to convert the actual weights for this to work.
	if any(True for x in lora.keys() if x.endswith("adaln_single.linear.lora_A.weight")):
		lora = convert_lora_state_dict(lora, peft=True)
		alpha = find_peft_alpha(lora_path)
		lora.update({f"{k_back(x)}.alpha":torch.tensor(alpha) for x in lora.keys() if "lora_up" in x})
	else: # OneTrainer
		lora = convert_lora_state_dict(lora, peft=False)

	key_map = {k_back(x):f"diffusion_model.{k_back(x)}.weight" for x in lora.keys() if "lora_up" in x} # fake

	loaded = comfy.lora.load_lora(lora, key_map)
	if model is not None:
		# switch to custom model patcher when using LoRAs
		if isinstance(model, EXM_PixArt_ModelPatcher):
			new_modelpatcher = model.clone()
		else:
			new_modelpatcher = replace_model_patcher(model)
		k = new_modelpatcher.add_patches(loaded, strength)
	else:
		k = ()
		new_modelpatcher = None

	k = set(k)
	for x in loaded:
		if (x not in k):
			print("NOT LOADED", x)

	return new_modelpatcher
