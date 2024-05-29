import os
import sys
import custom.shared as shared
module_name, module_root = shared.init_module(os.path.abspath(__file__))

from .modules.path import PathManager
path_manager = PathManager()

state = {"preview_image": None, "ctrls_name": [], "ctrls_obj": [], "pipeline": None}

def add_ctrl(name, obj):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]

