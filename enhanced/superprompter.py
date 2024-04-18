import os
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import modules.config as config

tokenizer = None
model = None
modelDir = os.path.join(config.path_llms, "superprompt-v1" )


def answer(input_text="", max_new_tokens=512, repetition_penalty=1.2, temperature=0.5, top_p=1, top_k = 1 , seed=-1):
    global tokenizer, model

    if tokenizer is None or model is None:
        if not all(os.path.exists(os.path.join(modelDir, "model.safetensors")) for file in modelDir):
            config.downloading_superprompter_model()
            print("[SuperPrompt] Downloaded the model file for superprompter. \n")

        tokenizer = T5Tokenizer.from_pretrained(modelDir)
        model = T5ForConditionalGeneration.from_pretrained(modelDir, torch_dtype=torch.float16)

    if seed == -1:
        seed = random.randint(1, 1000000)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    if torch.cuda.is_available():
        model.to('cuda')

    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                            do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)

    dirty_text = tokenizer.decode(outputs[0])
    text = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
    
    return text
