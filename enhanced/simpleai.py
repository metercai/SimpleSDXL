import os
import sys
import json
import gradio as gr
import shared
from simpleai_base import simpleai_base, comfyd, models_hub_host, config, torch_version, xformers_version, cuda_version, comfyclient_pipeline
from simpleai_base.params_mapper import ComfyTaskParams
from simpleai_base.models_info import get_models_info, get_modelsinfo, set_scan_models_hash, refresh_models_info_from_path, models_info_path, sync_model_info
from build_launcher import is_win32_standalone_build

simpleai_config = config
#comfyd.echo_off = False
args_comfyd = [[]]
modelsinfo = None #get_modelsinfo()
models_info, models_info_muid, models_info_file = get_models_info()

def refresh_models_info():
    global modelsinfo, models_info, models_info_muid, models_info_file
    #refresh_models_info_from_path()
    modelsinfo = get_modelsinfo()
    models_info, models_info_muid, models_info_file = get_models_info()
    return

def reset_simpleai_args():
    global args_comfyd
    shared.sysinfo.update(dict(
        torch_version=torch_version,
        xformers_version=xformers_version,
        cuda_version=cuda_version))
    comfyclient_pipeline.COMFYUI_ENDPOINT_PORT = shared.sysinfo["loopback_port"]
    args_comfyd = comfyd.args_mapping(sys.argv) + [["--listen"], ["--port", f'{shared.sysinfo["loopback_port"]}', '--disable-smart-memory']] + ([["--windows-standalone-build"]] if is_win32_standalone_build else [])
    args_comfyd += [["--cuda-malloc"]] if not shared.args.disable_async_cuda_allocation and not shared.args.async_cuda_allocation else []
    args_comfyd += [["--fast"]] if 'RTX 40' in shared.sysinfo['gpu_name'] else []
    comfyd.comfyd_args = args_comfyd
    return

#set_scan_models_hash(True)


identity_note = '将身份ID与手机号绑定，可以固定本地密钥。既可以在本地存储和管理个人配置信息，又可以参与创意分享等会员服务。详情可见说明文档。'

from ldm_patched.modules.model_management import unload_all_models, soft_empty_cache

def get_vcode(nick, tele, state):
    unload_all_models()
    return state

def bind_identity(nick, tele, vcode, state):
    soft_empty_cache(True)
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

