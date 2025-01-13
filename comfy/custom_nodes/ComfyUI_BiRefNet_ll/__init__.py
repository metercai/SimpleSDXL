import os
import sys

import folder_paths

# 获取当前目录的父目录的父目录
parent_dir = os.path.dirname(os.path.abspath(__file__))

# 添加父目录的父目录到系统路径
sys.path.insert(0, parent_dir)

models_dir_key = "birefnet"
models_dir_default = os.path.join(folder_paths.models_dir, "rembg")
if models_dir_key not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[models_dir_key] = (
        [models_dir_default], folder_paths.supported_pt_extensions)
else:
    if not os.path.exists(models_dir_default):
        os.makedirs(models_dir_default, exist_ok=True)
    folder_paths.add_model_folder_path(models_dir_key, models_dir_default)

from . import birefnetNode

NODE_CLASS_MAPPINGS = {**birefnetNode.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**birefnetNode.NODE_DISPLAY_NAME_MAPPINGS}
