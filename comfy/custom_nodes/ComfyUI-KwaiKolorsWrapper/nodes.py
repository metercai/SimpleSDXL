import torch
import os
import random
import re
import gc
import json
import psutil
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

folder_paths.add_model_folder_path("llms", os.path.join(folder_paths.models_dir, "llms", "checkpoints"))

from .kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from .kolors.models.modeling_chatglm import ChatGLMModel, ChatGLMConfig
from .kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel
from diffusers import (DPMSolverMultistepScheduler, 
        EulerDiscreteScheduler, 
        EulerAncestralDiscreteScheduler, 
        DEISMultistepScheduler, 
        UniPCMultistepScheduler
)

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    pass
from comfy.utils import ProgressBar

class DownloadAndLoadKolorsModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'Kwai-Kolors/Kolors',
                    ],
                    ),
            "precision": ([ 'fp16'],
                    {
                    "default": 'fp16'
                    }),
            },
        }

    RETURN_TYPES = ("KOLORSMODEL",)
    RETURN_NAMES = ("kolors_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "KwaiKolorsWrapper"

    def loadmodel(self, model, precision):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        pbar = ProgressBar(4)

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
      
        if not os.path.exists(model_path):
            print(f"Downloading Kolor model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            allow_patterns=['*fp16.safetensors*', '*.json'],
                            ignore_patterns=['vae/*', 'text_encoder/*', 'tokenizer/*'],
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
        pbar.update(1)

        ram_rss_start = psutil.Process().memory_info().rss
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        
        print("Load UNET...")
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder= 'unet', variant="fp16", revision=None, low_cpu_mem_usage=True).to(dtype).eval()      
        ram_rss_end = psutil.Process().memory_info().rss
        print(f'Kolors-unet: RAM allocated = {(ram_rss_end-ram_rss_start)/(1024*1024*1024):.3f}GB')

        pipeline = StableDiffusionXLPipeline(
                unet=unet,
                scheduler=scheduler,
                )
    
        kolors_model = {
            'pipeline': pipeline, 
            'dtype': dtype
            }

        return (kolors_model,)

class LoadChatGLM3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "chatglm3_checkpoint": (folder_paths.get_filename_list("llms"),),
            },
        }

    RETURN_TYPES = ("CHATGLM3MODEL",)
    RETURN_NAMES = ("chatglm3_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "KwaiKolorsWrapper"

    def loadmodel(self, chatglm3_checkpoint):
        device=mm.get_torch_device()
        offload_device=mm.unet_offload_device()
        print(f'chatglm3: device={device}, offload_device={offload_device}')

        pbar = ProgressBar(2)
        chatglm3_path = folder_paths.get_full_path("llms", chatglm3_checkpoint)
        print("Load TEXT_ENCODER...")
        text_encoder_config = os.path.join(script_directory, 'configs', 'text_encoder_config.json')
        with open(text_encoder_config, 'r') as file:
            config = json.load(file)

        text_encoder_config = ChatGLMConfig(**config)
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            text_encoder = ChatGLMModel(text_encoder_config)
            if '4bit' in chatglm3_checkpoint:
                text_encoder.quantize(4)
            elif '8bit' in chatglm3_checkpoint:
                text_encoder.quantize(8)

        text_encoder_sd = load_torch_file(chatglm3_path)

        if is_accelerate_available:
            for key in text_encoder_sd:
                set_module_tensor_to_device(text_encoder, key, device=offload_device, value=text_encoder_sd[key])
        else:
            text_encoder.load_state_dict()
       
        tokenizer_path = os.path.join(script_directory,'configs',"tokenizer")
        tokenizer = ChatGLMTokenizer.from_pretrained(tokenizer_path)
        pbar.update(1)
    
        chatglm3_model = {
            'text_encoder': text_encoder, 
            'tokenizer': tokenizer
            }

        return (chatglm3_model,)

class DownloadAndLoadChatGLM3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "precision": ([ 'fp16', 'quant4', 'quant8'],
                    {
                    "default": 'fp16'
                    }),
            },
        }

    RETURN_TYPES = ("CHATGLM3MODEL",)
    RETURN_NAMES = ("chatglm3_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "KwaiKolorsWrapper"

    def loadmodel(self, precision):

        pbar = ProgressBar(2)
        model = "Kwai-Kolors/Kolors"
        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
        text_encoder_path = os.path.join(model_path, "text_encoder")
      
        if not os.path.exists(text_encoder_path):
            print(f"Downloading ChatGLM3 to: {text_encoder_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            allow_patterns=['text_encoder/*'],
                            ignore_patterns=['*.py', '*.pyc'],
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
        pbar.update(1)

        ram_rss_start = psutil.Process().memory_info().rss
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        print(f"Load TEXT_ENCODER..., {precision}, {offload_device}")

        text_encoder = ChatGLMModel.from_pretrained(
            text_encoder_path,
            torch_dtype=torch.float16
            ).to(offload_device)
        if precision == 'quant8':
            text_encoder.quantize(8)
        elif precision == 'quant4':
            text_encoder.quantize(4)
       
        tokenizer = ChatGLMTokenizer.from_pretrained(text_encoder_path)
        pbar.update(1)
    
        chatglm3_model = {
            'text_encoder': text_encoder, 
            'tokenizer': tokenizer
            }
        ram_rss_end = psutil.Process().memory_info().rss
        print(f'chatglm3: RAM allocated = {(ram_rss_end-ram_rss_start)/(1024*1024*1024):.3f}GB')

        return (chatglm3_model,)
        
class KolorsTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatglm3_model": ("CHATGLM3MODEL", ),
                "prompt": ("STRING", {"multiline": True, "default": "",}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",}),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("KOLORS_EMBEDS",)
    RETURN_NAMES =("kolors_embeds",)
    FUNCTION = "encode"
    CATEGORY = "KwaiKolorsWrapper"

    def encode(self, chatglm3_model, prompt, negative_prompt, num_images_per_prompt):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
         # Function to randomly select an option from the brackets
        def choose_random_option(match):
            options = match.group(1).split('|')
            return random.choice(options)

        # Randomly choose between options in brackets for prompt and negative_prompt
        prompt = re.sub(r'\{([^{}]*)\}', choose_random_option, prompt)
        negative_prompt = re.sub(r'\{([^{}]*)\}', choose_random_option, negative_prompt)

        if "|" in prompt:
            prompt = prompt.split("|")
            negative_prompt = [negative_prompt] * len(prompt)  # Replicate negative_prompt to match length of prompt list


        print(prompt)
        do_classifier_free_guidance = True

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        # Define tokenizers and text encoders
        tokenizer = chatglm3_model['tokenizer']
        text_encoder = chatglm3_model['text_encoder']

        text_encoder.to(device)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        output = text_encoder(
                input_ids=text_inputs['input_ids'] ,
                attention_mask=text_inputs['attention_mask'],
                position_ids=text_inputs['position_ids'],
                output_hidden_states=True)
        
        prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
        text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)


        if do_classifier_free_guidance:
            uncond_tokens = []
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
     

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            output = text_encoder(
                    input_ids=uncond_input['input_ids'] ,
                    attention_mask=uncond_input['attention_mask'],
                    position_ids=uncond_input['position_ids'],
                    output_hidden_states=True)
            negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
            negative_text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

        bs_embed = text_proj.shape[0]
        text_proj = text_proj.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        negative_text_proj = negative_text_proj.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        text_encoder.to(offload_device)
        mm.soft_empty_cache()
        gc.collect()
        kolors_embeds = {
            'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds,
            'pooled_prompt_embeds': text_proj,
            'negative_pooled_prompt_embeds': negative_text_proj
        }
        
        return (kolors_embeds,)


class KolorsSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kolors_model": ("KOLORSMODEL", ),
                "kolors_embeds": ("KOLORS_EMBEDS", ),
              
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.01}),

                "scheduler": (
                    [ 
                        "EulerDiscreteScheduler",
                        "EulerAncestralDiscreteScheduler",
                        "DPMSolverMultistepScheduler",
                        "DPMSolverMultistepScheduler_SDE_karras",
                        "UniPCMultistepScheduler",
                        "DEISMultistepScheduler",
                    ],
                      {
                    "default": 'EulerDiscreteScheduler'
                    }
                    ),
                },
                "optional": {
                      "latent": ("LATENT", ),
                      "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES =("latent",)
    FUNCTION = "process"
    CATEGORY = "KwaiKolorsWrapper"

    def process(self, kolors_model, kolors_embeds, width, height, seed, steps, cfg, scheduler, latent=None, denoise_strength=1.0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        vae_scaling_factor = 0.13025 #SDXL scaling factor

        mm.soft_empty_cache()
        gc.collect()

        pipeline = kolors_model['pipeline']

        scheduler_config = {
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.014,
            "dynamic_thresholding_ratio": 0.995,
            "num_train_timesteps": 1100,
            "prediction_type": "epsilon",
            "rescale_betas_zero_snr": False,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "trained_betas": None,
        }
        if scheduler == "DPMSolverMultistepScheduler":
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == "DPMSolverMultistepScheduler_SDE_karras":
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == "DEISMultistepScheduler":
            scheduler_config.pop("rescale_betas_zero_snr")
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == "EulerDiscreteScheduler":
            scheduler_config.update({"interpolation_type": "linear"})
            scheduler_config.pop("dynamic_thresholding_ratio")
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == "EulerAncestralDiscreteScheduler":
            scheduler_config.pop("dynamic_thresholding_ratio")
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        elif scheduler == "UniPCMultistepScheduler":
            scheduler_config.pop("rescale_betas_zero_snr")
            noise_scheduler = UniPCMultistepScheduler(**scheduler_config)

        pipeline.scheduler = noise_scheduler

        generator= torch.Generator(device).manual_seed(seed)

        pipeline.unet.to(device)

        if latent is not None:
            samples_in = latent['samples']
            samples_in = samples_in * vae_scaling_factor
            samples_in = samples_in.to(pipeline.unet.dtype).to(device)

        latent_out = pipeline(
            prompt=None,
            latents=samples_in if latent is not None else None,
            prompt_embeds = kolors_embeds['prompt_embeds'],
            pooled_prompt_embeds = kolors_embeds['pooled_prompt_embeds'],
            negative_prompt_embeds = kolors_embeds['negative_prompt_embeds'],
            negative_pooled_prompt_embeds = kolors_embeds['negative_pooled_prompt_embeds'],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=1,
            generator= generator,
            strength=denoise_strength,
            ).images

        pipeline.unet.to(offload_device)
        
        latent_out = latent_out / vae_scaling_factor

        return ({'samples': latent_out},)   
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadKolorsModel": DownloadAndLoadKolorsModel,
    "DownloadAndLoadChatGLM3": DownloadAndLoadChatGLM3,
    "KolorsSampler": KolorsSampler,
    "KolorsTextEncode": KolorsTextEncode,
    "LoadChatGLM3": LoadChatGLM3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadKolorsModel": "(Down)load Kolors Model",
    "DownloadAndLoadChatGLM3": "(Down)load ChatGLM3 Model",
    "KolorsSampler": "Kolors Sampler",
    "KolorsTextEncode": "Kolors Text Encode",
    "LoadChatGLM3": "Load ChatGLM3 Model"
}
