import os
import random
import torch
import tarfile
import time
import translators as ts

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from modules.config import paths_llms
from modules.model_loader import load_file_from_url
from download import download
from functools import lru_cache


Q_punct = '｀～！＠＃＄％＾＆＊（）＿＋＝－｛｝［］：＂；｜＜＞？，．／。　１２３４５６７８９０'
B_punct = '`~!@#$%^&*()_+=-{}[]:";|<>?,./. 1234567890'
Q_alphabet = 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
B_alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

translator_org = ['baidu', 'alibaba', 'sogou', 'caiyun']
translator_default = translator_org[random.randint(1,2)]
translator_path = os.path.join(paths_llms[0], 'nllb-200-distilled-600M')
translator_slim_path = os.path.join(paths_llms[0], 'Helsinki-NLP/opus-mt-zh-en')
is_chinese = lambda x: sum([1 if u'\u4e00' <= i <= u'\u9fa5' else 0 for i in x]) > 0

translator_path_old = os.path.join(paths_llms[0], '../translator')
if os.path.exists(translator_path_old) and not os.path.exists(paths_llms[0]):
    os.rename(translator_path_old, paths_llms[0])


g_tokenizer = ''
g_model = ''
g_model_type = ''

def Q2B_number_punctuation(text):
    global Q_punct, B_punct

    texts = list(text)
    Bpunct = list(B_punct)
    for i in range(0,len(texts)):
        j = Q_punct.find(texts[i])
        if j >= 0:
            texts[i] = Bpunct[j]
    return ''.join(texts)

def Q2B_alphabet(text):
    global Q_alphabet, B_alphabet

    texts = list(text)
    Balphabet = list(B_alphabet)
    for i in range(0,len(texts)):
        j = Q_alphabet.find(texts[i])
        if j >= 0:
            texts[i] = Balphabet[j]
    return ''.join(texts)


def translate2en_model(model, tokenizer, text_zh):
    inputs = tokenizer(text_zh, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=60
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].lower()

def translate2zh_model(model, tokenizer, text_en):
    inputs = tokenizer(text_en, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"), max_length=60
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].lower()


@lru_cache(maxsize=32, typed=False)
def translate2en_apis(text):
    global translator_default
    if not text:
        return text
    try:
        return ts.translate_text(text, translator=translator_default, to_language='en')
    except Exception as e:
        try:
            print(f'[Translator] Change another translator because of {e}')
            translator_default = translator_org[random.randint(1,2)]
            return ts.translate_text(text, translator=translator_default, to_language='en')
        except Exception as e:
            print(f'[Translator] Error during translation of APIs methods: {e}')
            return text

def init_or_load_translator_model(method='Slim Model'):
    global g_tokenizer, g_model, g_model_type

    print(f'init_or_load_translator_model: {method}')
    if method != g_model_type or g_tokenizer is None or g_model is None:
        if method == "Big Model":
            if not os.path.exists(translator_path):
                os.makedirs(translator_path)
                url = 'https://gitee.com/metercai/SimpleSDXL/releases/download/win64/nllb_200_distilled_600m.tar.gz'
                cached_file = os.path.join(translator_path, 'nllb_200_distilled_600m.tar.gz')
                download(url, cached_file, progressbar=True)
                with tarfile.open(cached_file, 'r:gz') as tarf:
                    tarf.extractall(translator_path)
                os.remove(cached_file)
            if not os.path.exists(os.path.join(translator_path, 'pytorch_model.bin')):
                load_file_from_url(
                    url='https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin',
                    model_dir=translator_path,
                    file_name='pytorch_model.bin')
            print(f'[Translator] load model form : {translator_path}')
            g_tokenizer = AutoTokenizer.from_pretrained(translator_path, src_lang="zho_Hans")
            g_model = AutoModelForSeq2SeqLM.from_pretrained(translator_path)
        else:
            if not os.path.exists(translator_slim_path):
                os.makedirs(translator_slim_path)
                url = 'https://gitee.com/metercai/SimpleSDXL/releases/download/win64/opus_mt_zh_en.tar.gz'
                cached_file = os.path.join(translator_slim_path, 'opus_mt_zh_en.tar.gz')
                download(url, cached_file, progressbar=True)
                with tarfile.open(cached_file, 'r:gz') as tarf:
                    tarf.extractall(translator_slim_path)
                os.remove(cached_file)
            if not os.path.exists(os.path.join(translator_slim_path, 'pytorch_model.bin')):
                load_file_from_url(
                    url='https://huggingface.co/Helsinki-NLP/opus-mt-zh-en/resolve/main/pytorch_model.bin',
                    model_dir=translator_slim_path,
                    file_name='pytorch_model.bin')
            print(f'[Translator] load slim model form : {translator_slim_path}')
            g_tokenizer = AutoTokenizer.from_pretrained(translator_slim_path)
            g_model = AutoModelForSeq2SeqLM.from_pretrained(translator_slim_path).eval()
        g_model_type = method
    return g_tokenizer, g_model

def free_translator_model():
    global g_tokenizer, g_model
    del g_tokenizer
    del g_model
    return

def toggle(text: str, method: str = 'Slim Model') -> str:
    is_chinese_ext = lambda x: (Q_alphabet + B_punct).find(x) < -1
    if is_chinese(text):
        return convert(text, method)
    else:
        return convert(text, method, 'cn')


def convert(text: str, method: str = 'Slim Model', lang: str = 'en' ) -> str:
    global Q_alphabet, B_puncti, is_chinese

    start = time.perf_counter()

    if lang=='cn':
        tokenizer, model = init_or_load_translator_model('Big Model')
        text_zh = translate2zh_model(model, tokenizer, text)
        stop = time.perf_counter()
        print(f'[Translator] Translate by "Big Model" in {(stop-start):.2f}s: "{text}" to "{text_zh}"')
        return text_zh
    is_chinese_ext = lambda x: (Q_alphabet + B_punct).find(x) < -1 
    #text = Q2B_number_punctuation(text)
    if is_chinese(text):
        if method == 'Third APIs':
            print(f'[Translator] Using an online translation APIs.')
        else:
            tokenizer, model = init_or_load_translator_model(method)


        def T_ZH2EN(text_zh):
            if method=="Slim Model":
                encoded = tokenizer([text_zh], return_tensors="pt")
                sequences = model.generate(**encoded)
                return 'Slim Model', tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
            elif method=="Big Model":
                inputs = tokenizer(text_zh, return_tensors="pt")
                translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=60)
                return 'Big Model', tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].lower()
            else:
                return translator_default, translate2en_apis(text_zh)


        text_eng = ""
        text_zh = ""
        for _char in iter(text):
            if is_chinese(_char):
                text_zh += _char
            else:
                if len(text_zh) > 0:
                    if is_chinese_ext(_char):
                        text_zh += _char
                        continue
                    else:
                        #text_zh = Q2B_alphabet(text_zh)
                        ts_methods, text_en=T_ZH2EN(text_zh)
                        #print(f'translate: {text_zh} -> {text_en}')
                        text_eng += text_en  
                        text_zh = ""
                text_eng += _char
        if len(text_zh) > 0:
            ts_methods, text_en=T_ZH2EN(text_zh)
            text_eng += text_en
        text_eng = Q2B_number_punctuation(text_eng)
        text_eng = Q2B_alphabet(text_eng)
        stop = time.perf_counter()
        print(f'[Translator] Translate by "{ts_methods}" in {(stop-start):.2f}s: "{text}" to "{text_eng}"')
        return text_eng
    return text


