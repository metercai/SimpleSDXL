import os
from transformers import CLIPTokenizer

gradio_root = None

clip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/clip_vision')
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(clip_path, "clip-vit-large-patch14"))

