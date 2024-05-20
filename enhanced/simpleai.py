import json
import gradio as gr
from simpleai_base import simpleai_base, models_hub_host, config, torch_version, xformers_version, cuda_version, comfyclient_pipeline
from simpleai_base.params_mapper import ComfyTaskParams
from simpleai_base.models_info import init_models_info, models_info, models_info_muid, refresh_models_info_from_path, sync_model_info

simpleai_config = config
token = simpleai_base.init_local(f'SimpleSDXL_User')
sysinfo = json.loads(token.get_sysinfo().to_json())
sysinfo.update(dict(
    did=token.get_did(),
    torch_version=torch_version,
    xformers_version=xformers_version,
    cuda_version=cuda_version))


#identity_note = '将手机号与身份私钥绑定后，即获得固定可信的数字身份标识，具备了个人数据的管理和流转能力：<br>1，包括系统配置、出图目录和日志等私有数据跟随数字身份标识而分隔存储，实现同设备私人数据的隔离。2，可以接收和转发非公开资源，实现私人数据间的安全流转。3，可以管理多设备相同身份的数据，实现私人数据的分布式共享。<br>SimpleAI的去中心化数字身份，由本地存储的身份私钥、身份口令、设备私钥及系列加解密算法组成，自主可控，安全可靠。'

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

