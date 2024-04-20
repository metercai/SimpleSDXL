import os
import sys
from shared import *
import modules.config as config

paths_checkpoints = config.paths_checkpoints
paths_loras = config.paths_loras
path_embeddings = config.path_embeddings
path_vae_approx = config.path_vae_approx
path_upscale_models = config.path_upscale_models
path_inpaint = config.path_inpaint
path_controlnet = config.path_controlnet
path_clip_vision = config.path_clip_vision
path_fooocus_expansion = config.path_fooocus_expansion
path_llms = config.path_llms
path_outputs = config.path_outputs

path_root = root

def init_module(file_path):
    module_root = os.path.dirname(file_path)
    sys.path.append(module_root)
    module_name = os.path.relpath(module_root, os.path.dirname(os.path.abspath(__file__)))
    print(f'[{module_name}] The customized module:{module_name} is initializing ...') 
    return module_name, module_root


