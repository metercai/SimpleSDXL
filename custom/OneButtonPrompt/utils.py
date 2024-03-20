import os
import sys
import custom.shared
module_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_root)

relative_path = os.path.relpath(module_root, custom.shared.path_root)
relative_path_n = os.path.relpath(custom.shared.path_root, module_root)

def path_fixed(file_path):
    fixed_path = os.path.join(relative_path, file_path)
    return fixed_path

def root_path_fixed(file_path):
    fixed_path = os.path.join(relative_path_n, file_path)
    return fixed_path

