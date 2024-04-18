import os
from transformers import CLIPTokenizer
from modules.config import path_clip_vision

gradio_root = None

tokenizer = CLIPTokenizer.from_pretrained(os.path.join(path_clip_vision, "clip-vit-large-patch14"))

