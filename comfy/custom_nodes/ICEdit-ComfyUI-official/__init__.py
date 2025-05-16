# -*- coding: utf-8 -*-
from .ICEF_node import (InContextEditInstruction, DiptychCreate, ICEFConditioning)

NODE_MAPPINGS = {
    'InContextEditInstruction': ('InContextEditInstruction~', InContextEditInstruction),
    'DiptychCreate': ('DiptychCreate~', DiptychCreate),
    'ICEFConditioning': ('ICEFConditioning~', ICEFConditioning)
}

NODE_CLASS_MAPPINGS = {k: v[1] for k, v in NODE_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k: v[0] for k, v in NODE_MAPPINGS.items()}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
