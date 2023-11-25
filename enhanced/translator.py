import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from modules.config import path_translator

models_dir = os.path.join(path_translator, 'nllb-200-distilled-600M')

def is_chinese_char(_char):
    if u'\u4e00' <= _char <= u'\u9fa5':
        return True
    return False

def translate_en(model, tokenizer, text_zh):
    inputs = tokenizer(text_zh, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].lower()

def convert(text: str) -> str:
    is_chinese= lambda x='ddd':sum([1 if u'\u4e00' <= i <= u'\u9fa5' else 0 for i in x])>0
    if is_chinese(text):
        print(f'[Translator] load model form : {models_dir}')
        tokenizer = AutoTokenizer.from_pretrained(models_dir, src_lang="zho_Hans")
        model = AutoModelForSeq2SeqLM.from_pretrained(models_dir)

        text_eng = ""
        text_zh = ""
        for _char in iter(text):
            if is_chinese_char(_char):
                text_zh += _char
            else:
                if len(text_zh) > 0:
                    text_eng += translate_en(model, tokenizer, text_zh)  
                    text_zh = ""
                text_eng += _char
        if len(text_zh) > 0:
            text_eng += translate_en(model, tokenizer, text_zh)
        print(f'[Translator] translate "{text}" to "{text_eng}"')
        return text_eng
    return text


