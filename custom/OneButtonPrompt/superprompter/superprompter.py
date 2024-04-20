import os
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import modules.config as top_config
import shared
import shutil
from enhanced.superprompter import *


def load_models():
    global tokenizer, model, modelDir

    if tokenizer is None or model is None:
        if not os.path.exists(modelDir):
            org_modelDir = os.path.join(shared.root, "models/llms/superprompt-v1")
            shutil.copytree(org_modelDir, modelDir)
        if not os.path.exists(os.path.join(modelDir, "model.safetensors")):
            top_config.downloading_superprompter_model()
            print("[SuperPrompt] Downloaded the model file for superprompter. \n")
        tokenizer = T5Tokenizer.from_pretrained(modelDir)
        model = T5ForConditionalGeneration.from_pretrained(modelDir, torch_dtype=torch.float16).to(shared.torch_device)

