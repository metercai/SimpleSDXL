import os
import sys
import json
import gradio as gr
from simpleai_base import simpleai_base, comfyd, models_hub_host, config, torch_version, xformers_version, cuda_version, comfyclient_pipeline
from simpleai_base.params_mapper import ComfyTaskParams
from simpleai_base.models_info import get_models_info, refresh_models_info_from_path, models_info_path, sync_model_info

simpleai_config = config
token = None
sysinfo = {"location": "CN"}
#comfyd.echo_off = False
args_comfyd = [[]]

models_info, models_info_muid, models_info_file = get_models_info()

def refresh_models_info():
    global models_info, models_info_muid, models_info_file
    refresh_models_info_from_path()
    models_info, models_info_muid, models_info_file = get_models_info()

def reset_simpleai_args(launch_token, launch_sysinfo):
    global token, sysinfo, args_comfyd
    token = launch_token
    sysinfo = launch_sysinfo
    sysinfo.update(dict(
        torch_version=torch_version,
        xformers_version=xformers_version,
        cuda_version=cuda_version))
    comfyclient_pipeline.COMFYUI_ENDPOINT_PORT = sysinfo["loopback_port"]
    args_comfyd = comfyd.args_mapping(sys.argv) + [["--listen"], ["--port", f'{sysinfo["loopback_port"]}']]

identity_note = '将手机号与身份私钥绑定，获得固定的可信数字身份标识，就可以存储、找回和分享非公开资源，也可以具备云端登录和算力共享的能力，详情可见。'

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

