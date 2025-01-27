import os
import gc
import torch
import shared
import threading
import modules.config as config
import enhanced.translator as translator
import enhanced.superprompter as superprompter
import ldm_patched.modules.model_management
import modules.default_pipeline as pipeline
import enhanced.all_parameters as ads
import logging
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

from PIL import Image
from transformers import AutoTokenizer, AutoModel
from modules.model_loader import download_diffusers_model
from modules.util import HWC3, resize_image, is_chinese
from enhanced.simpleai import comfyd

class MiniCPM:  
    model = "MiniCPMv2_6-prompt-generator" # "MiniCPM-V-2_6-int4"
    prompt_i2t = "A descriptive caption for this image"
    output_chinese = "and output it in Chinese"
    prompt_extend = "Expand the following description to obtain a descriptive caption with more details in image: "
    prompt_translator = "Translate the following text into English, remind you only need respons the translation itself and no other information:"
    prompt_translator_cn = "Translate the following text into Chinese, remind you only need respons the translation itself and no other information:"
    model_file = os.path.join(model, "pytorch_model-00001-of-00002.bin")  # "model-00001-of-00002.safetensors")
    model_url = "https://huggingface.co/metercai/SimpleSDXL2/resolve/main/models_minicpm_v2.6_prompt_simpleai_1224.zip"
    
    lock = threading.Lock()
    model_v26 = None
    tokenizer = None
    enable = ads.get_admin_default('minicpm_checkbox')
    bf16_support = ( torch.cuda.is_available() and torch.cuda.get_device_capability(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))[0] >= 8 )

    def __init__(self):
        pass

    @classmethod
    def set_enable(cls, flag):
        with cls.lock:
            cls.enable = flag
    
    @classmethod
    def get_enable(cls):
        return cls.enable

    def load_model(self, download=False):
        if not shared.modelsinfo.exists_model(catalog="llms", model_path=MiniCPM.model_file):
            if download:
                download_diffusers_model('llms', MiniCPM.model, 2, MiniCPM.model_url)
            else:
                return
        MODEL_PATH = os.path.join(config.paths_llms[0], MiniCPM.model)
        tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, trust_remote_code=True, low_cpu_mem_usage=True)
        text_model = AutoModel.from_pretrained(
                MODEL_PATH, trust_remote_code=True, low_cpu_mem_usage=True,
                attn_implementation="sdpa", torch_dtype=torch.bfloat16 if MiniCPM.bf16_support else torch.float16)
        text_model.eval()
        with MiniCPM.lock:
            MiniCPM.model_v26 = text_model
            MiniCPM.tokenizer = tokenizer
        ldm_patched.modules.model_management.print_memory_info("after load minicpm model")
        return

    def free_model(self):
        if MiniCPM.model_v26 is None and MiniCPM.tokenizer is None:
            return
        with MiniCPM.lock:
            del MiniCPM.model_v26
            del MiniCPM.tokenizer
            MiniCPM.model_v26 = None
            MiniCPM.tokenizer = None
        translator.free_translator_model()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        ldm_patched.modules.model_management.print_memory_info("after free minicpm model")
    
    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, image, prompt, max_tokens=2048, temperature=0.7, top_p=0.8, top_k=100, repetition_penalty=1.05, seed=-1):
        comfyd.stop()
        pipeline.free_everything()
        ldm_patched.modules.model_management.print_vram_info_by_nvml("before minicpm inference")
        if MiniCPM.model_v26 is None or MiniCPM.tokenizer is None:
            self.load_model(download=True)
        image = image if image is None else Image.fromarray(resize_image(image, min_side=768, resize_mode=3))
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        
        res = MiniCPM.model_v26.chat(
            image=None,
            msgs=msgs,
            tokenizer=MiniCPM.tokenizer,
            sampling=True,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed
        )
        
        generated_text = res
        logger.info(f'The generated text:{generated_text}')
        ldm_patched.modules.model_management.print_memory_info("after minicpm inference")
        return generated_text

    def interrogate(self, image, output_chinese=False, prompt=None, additional_prompt=None):
        if prompt is not None:
            logger.info(f'The prompt of image: {prompt}')
            return self.inference(image, prompt)
        prompt = MiniCPM.prompt_i2t
        if additional_prompt:
            prompt = f'{prompt}, {additional_prompt}'
        if output_chinese:
            prompt = f'{prompt}, {MiniCPM.output_chinese}'
        logger.info(f'The prompt of image: {prompt}')
        return self.inference(image, prompt)

    def extended_prompt(self, input_text, prompt, translation_methods='Third APIs'):
        if not MiniCPM.get_enable() or not shared.modelsinfo.exists_model(catalog="llms", model_path=MiniCPM.model_file):
            return superprompter.answer(input_text=translator.convert(f'{prompt}{input_text}', translation_methods))
        else:
            return self.inference(None, prompt=f'{MiniCPM.prompt_extend}{input_text}')

    def translate(self, input_text, method=None):
        if not is_chinese(input_text):
            return input_text
        if MiniCPM.get_enable() and shared.modelsinfo.exists_model(catalog="llms", model_path=MiniCPM.model_file) and method in [None, 'Big Model']:
            return self.inference(None, prompt=f'{MiniCPM.prompt_translator}{input_text}')
        else:
            return translator.convert(input_text, method)

    def translate_cn(self, input_text_cn):
        if MiniCPM.get_enable() and shared.modelsinfo.exists_model(catalog="llms", model_path=MiniCPM.model_file):
            return self.inference(None, prompt=f'{MiniCPM.prompt_translator_cn}{input_text_cn}')
        else:
            return input_text_cn

minicpm = MiniCPM()
default_interrogator = minicpm.interrogate
