
import torch
import torch.nn as nn
import os
import folder_paths
from comfy.utils import load_torch_file, ProgressBar
from tqdm import tqdm
from comfy import model_management as mm
script_directory = os.path.dirname(os.path.abspath(__file__))
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
from transformers import AutoTokenizer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from accelerate import init_empty_weights
from ..utils import set_module_tensor_to_device, log

from .system_prompt import SYSTEM_PROMPT_MAP
SYSTEM_PROMPT_KEYS = [item["label"] for item in SYSTEM_PROMPT_MAP]

config_3b ={
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 32768,
  "max_window_layers": 70,
  "model_type": "qwen2",
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": True,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": True,
  "use_sliding_window": False,
  "vocab_size": 151936
}

config_7b ={
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": True,
  "use_sliding_window": False,
  "vocab_size": 152064
}


class QwenLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (folder_paths.get_filename_list("text_encoders"), ),
            "load_device": (["main_device", "offload_device"], {"advanced": True}),
            "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
        },
    }
    RETURN_TYPES = ("QWENMODEL",)
    FUNCTION = "load"
    CATEGORY = "WanVideoWrapper"

    def load(self, model, load_device, precision):
        transformer_load_device = device if load_device == "main_device" else offload_device
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[precision]


        sd = load_torch_file(folder_paths.get_full_path("text_encoders", model))
        tokenizer_path = os.path.join(script_directory, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        hf_config = Qwen2Config(**config_3b if "3b" in model.lower() else config_7b)
        
        # Fix vocab size to match actual tokenizer
        actual_vocab_size = len(tokenizer)
        if hf_config.vocab_size != actual_vocab_size:
            log.warning(f"Adjusting vocab_size from {hf_config.vocab_size} to {actual_vocab_size} to match tokenizer")
            hf_config.vocab_size = actual_vocab_size
            
        with init_empty_weights():
            hf_model = Qwen2ForCausalLM(hf_config)
        log.info("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in hf_model.named_parameters())
        pbar = ProgressBar(param_count)
        for name, param in tqdm(hf_model.named_parameters(),
                desc=f"Loading transformer parameters to {transformer_load_device}",
                total=param_count,
                leave=True):
            if name not in sd:
                log.warning(f"Parameter {name} not found in state dict, skipping.")
                continue
         
            set_module_tensor_to_device(hf_model, name, device=transformer_load_device, dtype=base_dtype, value=sd[name])
            pbar.update(1)

        hf_model.lm_head = nn.Linear(hf_model.config.hidden_size, hf_model.config.vocab_size, bias=False)
        
        if hf_config.tie_word_embeddings:
            hf_model.lm_head.weight = hf_model.get_input_embeddings().weight
        else:
            if "lm_head.weight" in sd:
                set_module_tensor_to_device(hf_model, "lm_head.weight", device=transformer_load_device, dtype=base_dtype, value=sd["lm_head.weight"])
            else:
                hf_model.lm_head.weight = hf_model.get_input_embeddings().weight
                
        hf_model.lm_head.to(hf_model.device, dtype=base_dtype)

        class EmptyObj:
            pass
        qwen = EmptyObj()
        qwen.model = hf_model
        qwen.tokenizer = tokenizer
        return (qwen,)

class WanVideoPromptExtender:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "qwen": ("QWENMODEL", ),
            "prompt": ("STRING", {"multiline": True}),
            "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1, "tooltip": "Maximum number of new tokens to generate."}),
            "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the model on. Default uses the main device."}),
            "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Force offload the model to the offload device after generation. Useful for large models."})
        },
        "optional": {
            "system_prompt": (SYSTEM_PROMPT_KEYS, {"tooltip": "System prompt to use for the model."}),
            "custom_system_prompt": ("STRING", {"default": "", "forceInput": True, "tooltip": "Custom system prompt to use instead of the predefined ones."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "WanVideoWrapper"

    def generate(self, qwen, prompt, device, force_offload, max_new_tokens, system_prompt=None, custom_system_prompt=None, seed=0):
        if device == "gpu":
            device = mm.get_torch_device()
        elif device == "cpu":
            device = torch.device("cpu")

        if custom_system_prompt is None:
            sys_prompt = next((item["prompt"] for item in SYSTEM_PROMPT_MAP if item["label"] == system_prompt), "")
        else:
            sys_prompt = custom_system_prompt

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        text = qwen.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = qwen.tokenizer([text], return_tensors="pt").to(device)
        torch.manual_seed(seed)
        qwen.model.to(device)
        generated_ids = qwen.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
        )
        if force_offload:
            qwen.model.to(offload_device)
            mm.soft_empty_cache()

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = qwen.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return (response,)
    
class WanVideoPromptExtenderSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (folder_paths.get_filename_list("text_encoders"), ),
            "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1, "tooltip": "Maximum number of new tokens to generate."}),
            "system_prompt": (SYSTEM_PROMPT_KEYS, {"tooltip": "System prompt to use for the model."}),
        },
        "optional": {
            "custom_system_prompt": ("STRING", {"default": "", "forceInput": True, "tooltip": "Custom system prompt to use instead of the predefined ones."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }
        }
    RETURN_TYPES = ("WANVIDEOPROMPTEXTENDER_ARGS",)
    RETURN_NAMES = ("extender_args",)
    FUNCTION = "set"
    CATEGORY = "WanVideoWrapper"

    def set(self, model, system_prompt, max_new_tokens, custom_system_prompt=None, seed=0):

        if custom_system_prompt is None:
            sys_prompt = next((item["prompt"] for item in SYSTEM_PROMPT_MAP if item["label"] == system_prompt), "")
        else:
            sys_prompt = custom_system_prompt
        
        extender_settings = {
            "model": model,
            "system_prompt": sys_prompt,
            "max_new_tokens": max_new_tokens,
            "device": "gpu",
            "force_offload": True,
            "seed": seed
        }

        return (extender_settings,)
    
NODE_CLASS_MAPPINGS = {
    "QwenLoader": QwenLoader,
    "WanVideoPromptExtender": WanVideoPromptExtender,
    "WanVideoPromptExtenderSelect": WanVideoPromptExtenderSelect
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenLoader": "Qwen Loader",
    "WanVideoPromptExtender": "Wan Video Prompt Extender",
    "WanVideoPromptExtenderSelect": "Wan Video Prompt Extender Select"
    }
