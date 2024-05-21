"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 Searge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os

import folder_paths

from .modules._legacy import LEGACY_CLASS_MAPPINGS, LEGACY_DISPLAY_NAME_MAPPINGS

from .searge_sdxl import SEARGE_CLASS_MAPPINGS, SEARGE_DISPLAY_NAME_MAPPINGS

folder_paths.add_model_folder_path("annotators", os.path.join(folder_paths.models_dir, "annotators"))

NODE_CLASS_MAPPINGS = SEARGE_CLASS_MAPPINGS | LEGACY_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = SEARGE_DISPLAY_NAME_MAPPINGS | LEGACY_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
