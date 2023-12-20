import os
import torch
import tarfile
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from modules.config import path_translator
from modules.model_loader import load_file_from_url
from download import download

translator_path = os.path.join(path_translator, 'nllb-200-distilled-600M')

Q_punct = '｀～！＠＃＄％＾＆＊（）＿＋＝－｛｝［］：＂；｜＜＞？，．／。　１２３４５６７８９０'
B_punct = '`~!@#$%^&*()_+=-{}[]:";|<>?,./. 1234567890'
Q_alphabet = 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
B_alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def Q2B_number_punctuation(text):
    global Q_punct, B_punct

    texts = list(text)
    Bpunct = list(B_punct)
    for i in range(0,len(texts)):
        j = Q_punct.find(texts[i])
        if j >= 0:
            texts[i] = Bpunct[j]
    #print(f'punctuation: text1={text} \n text2={texts}')
    return ''.join(texts)

def Q2B_alphabet(text):
    global Q_alphabet, B_alphabet

    texts = list(text)
    Balphabet = list(B_alphabet)
    for i in range(0,len(texts)):
        j = Q_alphabet.find(texts[i])
        if j >= 0:
            texts[i] = Balphabet[j]
    #print(f'alphabet: text1={text} \n text2={texts}')
    return ''.join(texts)

def translate_en(model, tokenizer, text_zh):
    inputs = tokenizer(text_zh, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].lower()

def convert(text: str) -> str:
    global Q_alphabet, B_punct

    is_chinese = lambda x: sum([1 if u'\u4e00' <= i <= u'\u9fa5' else 0 for i in x]) > 0
    is_chinese_ext = lambda x: (Q_alphabet + B_punct).find(x) < -1 


    #text = Q2B_number_punctuation(text)
    if is_chinese(text):
        if not os.path.exists(translator_path):
            os.makedirs(translator_path)
            url = 'https://gitee.com/metercai/SimpleSDXL/releases/download/win64/nllb_200_distilled_600m.tar.gz'
            cached_file = os.path.abspath(os.path.join(translator_path, 'nllb_200_distilled_600m.tar.gz'))
            download(url, cached_file, progressbar=True)
            with tarfile.open(cached_file, 'r:gz') as tarf:
                tarf.extractall(translator_path)
            os.remove(cached_file)
        load_file_from_url(
                url='https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin',
                model_dir=translator_path,
                file_name='pytorch_model.bin')
        print(f'[Translator] load model form : {translator_path}')
        tokenizer = AutoTokenizer.from_pretrained(translator_path, src_lang="zho_Hans")
        model = AutoModelForSeq2SeqLM.from_pretrained(translator_path)

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
                        text_en = translate_en(model, tokenizer, text_zh)
                        #print(f'translate: {text_zh} -> {text_en}')
                        text_eng += text_en  
                        text_zh = ""
                text_eng += _char
        if len(text_zh) > 0:
            text_eng += translate_en(model, tokenizer, text_zh)
        text_eng = Q2B_number_punctuation(text_eng)
        text_eng = Q2B_alphabet(text_eng)
        print(f'[Translator] Translate "{text}" to "{text_eng}"')
        return text_eng
    return text


