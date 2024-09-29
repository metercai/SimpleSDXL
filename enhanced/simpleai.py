import os
import sys
import gradio as gr
import shared
from simpleai_base import simpleai_base, utils, comfyd, models_hub_host, torch_version, xformers_version, cuda_version, comfyclient_pipeline
from simpleai_base.params_mapper import ComfyTaskParams
from simpleai_base.models_info import ModelsInfo, sync_model_info
from build_launcher import is_win32_standalone_build

#utils.echo_off = False
args_comfyd = [[]]
modelsinfo_filename = 'models_info.json'

def init_modelsinfo(models_root, path_map):
    global modelsinfo_filename
    models_info_path = os.path.abspath(os.path.join(models_root, modelsinfo_filename))
    if not shared.modelsinfo:
        shared.modelsinfo = ModelsInfo(models_info_path, path_map)
    return shared.modelsinfo

def reset_simpleai_args():
    global args_comfyd
    shared.sysinfo.update(dict(
        torch_version=torch_version,
        xformers_version=xformers_version,
        cuda_version=cuda_version))
    comfyclient_pipeline.COMFYUI_ENDPOINT_PORT = shared.sysinfo["loopback_port"]
    smart_memory = [] if shared.sysinfo['gpu_memory']<8180 else [['--disable-smart-memory']]
    windows_standalone = [["--windows-standalone-build"]] if is_win32_standalone_build else []
    args_comfyd = comfyd.args_mapping(sys.argv) + [["--listen"], ["--port", f'{shared.sysinfo["loopback_port"]}']] + smart_memory + windows_standalone
    args_comfyd += [["--cuda-malloc"]] if not shared.args.disable_async_cuda_allocation and not shared.args.async_cuda_allocation else []
    #args_comfyd += [["--fast"]] if 'RTX 40' in shared.sysinfo['gpu_name'] else []
    comfyd.comfyd_args = args_comfyd
    return



identity_note = '将数字身份与手机号绑定，既可以用固定密钥存储和管理个人配置信息，避免数据丢失，又可以参与创意分享等互助服务。'

from ldm_patched.modules.model_management import unload_all_models, soft_empty_cache

def get_vcode(nick, tele, state):
    return state

def bind_identity(nick, tele, vcode, state):
    return state

def confirm_identity(phrase, state):

    return state

def toggle_identity_dialog(state):
    if 'confirm_dialog' in state:
        flag = state['confirm_dialog']
    else:
        state['confirm_dialog'] = False
        flag = False
    state['confirm_dialog'] = not flag
    return gr.update(visible=not flag)

