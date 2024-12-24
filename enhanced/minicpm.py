import os
import torch
import shared
import threading
import modules.config as config
import enhanced.translator as translator
import enhanced.superprompter as superprompter

from PIL import Image
from transformers import AutoTokenizer, AutoModel
from modules.model_loader import download_diffusers_model

class MiniCPM:
    model = "MiniCPMv2_6-prompt-generator" # "MiniCPM-V-2_6-int4"
    prompt_i2t = "A descriptive caption for this image"
    prompt_i2t_chinese = "A descriptive caption for this image, and output it in Chinese"
    prompt_extend = "Expand the following description to obtain a descriptive caption with more details in image: "
    prompt_translator = "Translate the following Chinese text into English, remind you only need respons the translation itself and no other information:"
    model_file = os.path.join(model, "pytorch_model-00001-of-00002.bin")  # "model-00001-of-00002.safetensors")
    model_url = "https://huggingface.co/metercai/SimpleSDXL2/resolve/main/models_minicpm_v2.6_prompt_simpleai_1224.zip"
    
    lock = threading.Lock()
    model_v26 = None
    tokenizer = None
    enable = False

    def __init__(self):
        with MiniCPM.lock:
            MiniCPM.model_v26 = None
            MiniCPM.tokenizer = None
            MiniCPM.enable = False

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
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        text_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
        text_model.eval()
        with MiniCPM.lock:
            MiniCPM.model_v26 = text_model
            MiniCPM.tokenizer = tokenizer
        return

    def free_model(self):
        with MiniCPM.lock:
            MiniCPM.model_v26 = None
            MiniCPM.tokenizer = None
    
    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, image, prompt, max_tokens=2048, temperature=0.7):
        if MiniCPM.model_v26 is None or MiniCPM.tokenizer is None:
            self.load_model(download=True)
        image = image if image is None else Image.fromarray(image)
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        res = MiniCPM.model_v26.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        res = MiniCPM.model_v26.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=False,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generated_text = ""
        for new_text in res:
            generated_text += new_text
        print(f'MiniCPMv26 generated_text:{generated_text}')
        return generated_text

    def interrogate(self, image, output_chinese=False):
        if output_chinese:
            return self.inference(image, MiniCPM.prompt_i2t_chinese)
        else:
            return self.inference(image, MiniCPM.prompt_i2t)

    def extended_prompt(self, input_text, prompt, translation_methods='Third APIs'):
        if not MiniCPM.get_enable() or not shared.modelsinfo.exists_model(catalog="llms", model_path=MiniCPM.model_file):
            return superprompter.answer(input_text=translator.convert(f'{prompt}{input_text}', translation_methods))
        else:
            return self.inference(None, prompt=f'{MiniCPM.prompt_extend}{input_text}')

    def translate(self, input_text, translation_methods='Third APIs'):
        if not is_chinese(input_text):
            return input_text
        if not MiniCPM.get_enable() or not shared.modelsinfo.exists_model(catalog="llms", model_path=MiniCPM.model_file):
            return translator.convert(input_text, translation_methods)
        else:
            return self.inference(None, prompt=f'{MiniCPM.prompt_translator}{input_text}')

is_chinese = lambda x: sum([1 if (u'\u4e00' <= i <= u'\u9fa5') or (u'\u3000' <= i <= u'\u303F') or (u'\uFF00' <= i <= u'\uFFEF') else 0 for i in x]) > 0

minicpm = MiniCPM()
default_interrogator = minicpm.interrogate
