import os
import sys
import shutil
import re
import gradio as gr
import shared
import cv2
import modules.util as util
from simpleai_base import simpleai_base, utils, comfyd, models_hub_host, torch_version, xformers_version, cuda_version, comfyclient_pipeline
from simpleai_base.params_mapper import ComfyTaskParams
from simpleai_base.models_info import ModelsInfo, sync_model_info
from simpleai_base.simpleai_base import export_identity_qrcode_svg, import_identity_qrcode
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
    comfyd_images_path = shared.path_outputs #os.path.join(shared.path_outputs, shared.token.get_guest_did())
    comfyd_output = os.path.join(comfyd_images_path, 'comfyd_outputs')
    comfyd_intput = os.path.join(comfyd_images_path, 'comfyd_inputs')
    if not os.path.exists(comfyd_output):
        os.makedirs(comfyd_output)
    if not os.path.exists(comfyd_intput):
        os.makedirs(comfyd_intput)
    comfyd_intput_default_image = os.path.join(comfyd_intput, 'welcome.png')
    if not os.path.exists(comfyd_intput_default_image):
        default_image_path = os.path.join(shared.root, 'enhanced/attached/welcome.png')
        shutil.copy(default_image_path, comfyd_intput)
    args_comfyd += [["--output-directory", comfyd_output], ["--temp-directory", shared.temp_path], ["--input-directory", comfyd_intput]]
    #args_comfyd += [["--fast"]] if 'RTX 40' in shared.sysinfo['gpu_name'] else []
    comfyd.comfyd_args = args_comfyd
    return

def get_path_in_user_dir(user_did, filename, catalog=None):
    if user_did and filename:
        path = catalog if catalog else filename
        if shared.token.is_guest(user_did):
            user_did = 'guest_user'
        path_file = shared.token.get_path_in_user_dir(user_did, path)
        #print(f'get_path_in_user_dir: {path_file}')
        if catalog: 
            path_file = os.path.join(path_file, filename)
        path_file = os.path.abspath(path_file)
        if not os.path.exists(path_file):
            if os.path.isdir(path_file):
                os.makedirs(path_file)
            else:
                directory = os.path.dirname(path_file)
                if not os.path.exists(directory):
                    os.makedirs(directory)
        return path_file
    return None

def start_fast_comfyd(fast):
    if fast:
        comfyd.start(args_patch=[["--fast"]], force=True)
    else:
        comfyd.start(args_patch=[[]], force=True)

identity_note = '使用绑定手机号的可信数字身份，既可用固定密钥存储和管理个人配置，避免数据丢失，又可参与创意分享等互助服务。'
identity_note_1 = '当前浏览器已绑定身份访问。若要更换身份或在别处绑定，需先"解除绑定"。同一时刻同一身份只能在一处绑定。若需要完全离线使用身份，可导出身份二维码后妥善保存，在身份信息输入界面按提示导入。'
note1_0 = '请按提示输入创建身份时预设的身份口令，确认身份后完成绑定。'
note1_1 = '本地未匹配到数字身份，由云端根节点参与验证，请注意查收手机短信的身份验证码，用其找回加密副本或创建新身份。'
note1_2 = f'已匹配到本地的数字身份，{note1_0}'
note1_3 = f'已匹配到云端加密副本, {note1_0}'
note1_4 = '身份信息格式检验不通过。昵称最少4个字符或2个汉字，国内手机号11位数字。请重新输入身份信息，再次绑定。'
note1_5 = '与云端根节点连接异常，无法找回加密副本或验证身份。请检查软件环境，重新输入身份信息，再次绑定。'

note2_0 = '最少8位含大写、小写字母及数字的组合，每种字符至少1个。\n**<span style="color: red;">特别提醒</span>**<span style="color: red;">: 身份口令是唯一解锁数字身份的密钥，无法找回，遗失将导致已存储的配置信息和数据丢失，需妥善保存!!!</span>'
note2_1 = f'身份已验证，请按提示预设个人身份口令，{note2_0}'
note2_2 = f'身份口令遗失无法找回，请重复输入预设的身份口令，{note2_0}'
note2_3 = '身份验码证格式不对，请正确输入短信里的身份验证码，重新进行"身份验证"。'
note2_4 = '身份验码证未通过，请正确输入短信里的身份验证码，重新进行"身份验证"。'
note2_5 = f'设置的身份口令格式不对，请重新预设个人身份口令，{note2_0}'
note2_6 = f'身份口令设置失败，请重新预设个人的身份口令，{note2_0}'
note2_7 = f'身份口令与上次不一致，请重新预设个人身份口令，{note2_0}'
note2_8 = f'已匹配到本地的数字身份，请按提示预设个人身份口令，{note2_0}'

note3 = f'绑定成功! {identity_note_1}'
note3_1 = '身份绑定不成功，请重新输入个人身份口令，再次确认身份。'
note3_2 = '输入的身份口令格式不对，最少8位的大写、小写字母及数字的组合，每种字符至少1个。请重新输入个人身份口令。'

note4 = '身份已成功解绑，当前浏览器和应用服务回退到游客模式。'
note4_1 = '身份解绑不成功，请重新输入个人身份口令，再次确认身份。'

identity_help = '''
  我们的身份系统采用独创的分布式数字身份系统，其即具有安全性和隐私保护特性，也具有统一可流动性:
  **安全性**
  **隐私保护**
  **可用性**
  **一致性**
'''



theme_color = {
    "dark": "aqua",
    "light": "blue",
    }

current_id_info = lambda x,y,z,t: f'<b>当前用户信息</b><br>身份昵称: <span style="color: {theme_color[t]};">' + f'{x}' + f'</span><br>身份标识: <span style="color: {theme_color[t]};">{y}</span><br>系统标识: <span style="color: {theme_color[t]};">{z}</span>'

#[identity_note_info, input_id_info, identity_qr, identity_nick_input, identity_tele_input, identity_bind_button, identity_reset_button, identity_vcode_input, identity_verify_button, identity_phrase_input, identity_phrases_confirm_button, identity_phrases_set_button, identity_confirm_button, identity_unbind_button]
# [identity_note_info, input_identity, input_id_display, identity_vcode_input, identity_verify_button, identity_phrase_input, identity_phrases_set_button, identity_phrases_confirm_button, identity_confirm_button, identity_unbind_button]
# [identity_nick_input, identity_tele_input, identity_qr]

def trigger_input_identity(img):
    image = util.HWC3(img)
    qr_code_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_code_detector.detectAndDecode(image)
    if bbox is not None:
        try:
            user_did, nickname, telephone = import_identity_qrcode(data)
        except Exception as e:
            print("qrcode parse error")
            user_did, nickname, telephone = '', '', ''
    else:
        user_did, nickname, telephone = '', '', ''
    return  bind_identity(nickname, telephone)


def bind_identity(nick, tele):
    if check_input(nick, tele):
        where = shared.token.check_local_user_token(nick, tele)
        if where == 'local': # 本地密钥, 输入身份口令
            result = [note1_2] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=False)]*2 +[gr.update(visible=True)] + [gr.update(visible=False)]
        elif where == 'remote': # 远程找回, 输入验证码
            result = [note1_1] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=True)]*2 + [gr.update(visible=False)]*5
        elif where == 'immature': # 本地遗留密钥,重设身份口令
            result = [note2_8] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=True)]
        else:  # 过程出错, 重新输入绑定信息,再来
            result = [note1_5] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
    else: # 身份信息不合规, 重新输入
        result = [note1_4] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
    return result + [f'{nick}, {tele}']

def change_identity():
    return [identity_note] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7 + ['', '', None]


def verify_identity(input_id_info, state, vcode):
    if check_vcode(vcode):
        inputs = input_id_info.split(',')
        nick, tele = inputs[0].strip(), inputs[1].strip()
        next_cmd = shared.token.check_user_verify_code(nick, tele, vcode)
        if next_cmd == 'create':  # 验证成功, 创建新身份, 开始设置口令
            result = [note2_1] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=True)] + [gr.update(visible=False)]*3
        elif next_cmd == 'recall': # 验证并找回身份, 要求直接输入口令
            result = [note1_3] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=False)]*2 +[gr.update(visible=True)] + [gr.update(visible=False)]
        else:  # 验证失败, 重新输入
            if 'error:' in next_cmd:
                count = next_cmd.split(':')[1]
                note = note2_4 + f'还剩<span style="color: {theme_color[state["__theme"]]};">{count}</span>次机会。'
            else:
                note = note2_4
            result = [note] + [gr.update(visible=False)] + [gr.update(visible=True)]*3 + [gr.update(visible=False)]*5
    else: # 验证码格式错误, 重新输入
        result = [note2_3] + [gr.update(visible=False)] + [gr.update(visible=True)]*3 + [gr.update(visible=False)]*5
    return result

def set_phrases(input_id_info, state, phrase, steps):
    if steps == 'set':
        if check_phrase(phrase):  # 第一次设置, 要求二次确认
            state["user_phrase"] = phrase
            result = [note2_2] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2
        else: # 口令格式不对, 重新设置
            result = [note2_5] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=True)] + [gr.update(visible=False)]*3
    else:
        if state["user_phrase"] == phrase:
            inputs = input_id_info.split(',')
            nick, tele = inputs[0].strip(), inputs[1].strip()
            context = shared.token.set_phrase_and_get_context(nick, tele, phrase)
            if not shared.token.is_guest(context.get_did()):
                state["user_name"] = context.get_nickname()
                state["user_did"] = context.get_did()
                state["sys_did"] = context.get_sys_did()
                state["sstoken"] = shared.token.get_user_sstoken(context.get_did(), state["ua_hash"])
                note = f'身份口令设置成功，完成身份绑定。请牢记身份口令: `{phrase}` ，解除绑定或再次绑定都需要，建议抄写到私人笔记，仅限自己可见。'
                result = [note] + [gr.update(visible=False)]*4 + [gr.update(visible=True, value="")] + [gr.update(visible=False)]*3 + [gr.update(visible=True)]
            else: # 设置身份口令失败, 重新设置
                result = [note2_6] + [gr.update(visible=False)]  + [gr.update(visible=True)]*2 + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=True)] + [gr.update(visible=False)]*3
        else: # 口令两次不一致, 重新设置
            result = [note2_7] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')]+ [gr.update(visible=True)] + [gr.update(visible=False)]*3
        state["user_phrase"] = ''
    id_info = current_id_info(state["user_name"], state["user_did"], state["sys_did"], state["__theme"])
    return result + [id_info, gr.update(visible=not shared.token.is_guest(state["user_did"]))]

def confirm_identity(input_id_info, state, phrase):
    if check_phrase(phrase):
        inputs = input_id_info.split(',')
        nick, tele = inputs[0].strip(), inputs[1].strip()
        context = shared.token.get_user_context_with_phrase(nick, tele, phrase)
        if shared.token.is_guest(context.get_did()): # 口令不对, 绑定失败, 重新输入口令, 再次绑定
            result = [note3_1] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=False)]*2 +[gr.update(visible=True)] + [gr.update(visible=False)]
        else: # 绑定成功, 转解绑输入
            state["user_name"] = context.get_nickname()
            state["user_did"] = context.get_did()
            state["sys_did"] = context.get_sys_did()
            state["sstoken"] = shared.token.get_user_sstoken(context.get_did(), state["ua_hash"])
            result = [note3] + [gr.update(visible=False)]*4 + [gr.update(visible=True, value="")] + [gr.update(visible=False)]*3 + [gr.update(visible=True)]
    else: # 口令格式不对, 重新输入口令, 再次绑定
        result = [note3_2] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=False)]*2 +[gr.update(visible=True)] + [gr.update(visible=False)]
    id_info = current_id_info(state["user_name"], state["user_did"], state["sys_did"], state["__theme"])
    return result + [id_info, gr.update(visible=not shared.token.is_guest(state["user_did"]))]

def unbind_identity(input_id_info, state, phrase):
    if check_phrase(phrase):
        inputs = input_id_info.split(',')
        nick, tele = inputs[0].strip(), inputs[1].strip()
        context = shared.token.unbind_and_return_guest(nick, tele, phrase)
        if shared.token.is_guest(context.get_did()):
            state["user_name"] = context.get_nickname()
            state["user_did"] = context.get_did()
            state["sys_did"] = context.get_sys_did()
            state["sstoken"] = shared.token.get_user_sstoken(context.get_did(), state["ua_hash"])
            state["preset_store"] = False
            result = [note4, gr.update(visible=True)] + [gr.update(visible=False)]*8 + ['', '', None]
        else: # 口令不对, 解绑失败, 重新输入口令, 再次解绑
            result = [note4_1] + [gr.update(visible=False)]*4 + [gr.update(visible=True, value="")] + [gr.update(visible=False)]*3 + [gr.update(visible=True)] + ['', '', None]
    else: # 口令格式不对, 重新输入口令, 再次解绑
        result = [note3_2] + [gr.update(visible=False)]*4 + [gr.update(visible=True, value="")] + [gr.update(visible=False)]*3 +[gr.update(visible=True)] + ['', '', None]
    id_info = current_id_info(state["user_name"], state["user_did"], state["sys_did"], state["__theme"])
    return result + [id_info, gr.update(visible=not shared.token.is_guest(state["user_did"]))]


# [identity_dialog, current_id_info, identity_export_btn]
# [identity_note_info, input_identity, input_id_display, identity_vcode_input, identity_verify_button, identity_phrase_input, identity_phrases_set_button, identity_phrases_confirm_button, identity_confirm_button, identity_unbind_button]
# [identity_nick_input, identity_tele_input, identity_qr]
def toggle_identity_dialog(state):
    if 'identity_dialog' in state:
        flag = state['identity_dialog']
    else:
        state['identity_dialog'] = False
        flag = False
    state['identity_dialog'] = not flag
    is_guest = shared.token.is_guest(state["user_did"])
    result = [identity_note if is_guest else identity_note_1] + [gr.update(visible=is_guest)] + [gr.update(visible=False)]*3 + [gr.update(visible=not is_guest)] + [gr.update(visible=False)]*3 + [gr.update(visible=not is_guest)] + ['', '', None]
    result = [gr.update(visible=not flag), current_id_info(state["user_name"], state["user_did"], state["sys_did"], state["__theme"]), gr.update(visible=not is_guest)]
    return result


is_chinese = lambda x: sum([1 if u'\u4e00' <= i <= u'\u9fa5' else 0 for i in x]) > 0
def check_input(nick, tele):
    length = 0
    for n in nick:
        length += 1
        if is_chinese(n):
            length += 1
    if length < 4 or length > 24:
        return False
    
    if len(tele)<8 or len(tele)>15 or not tele.isdigit() or tele[0] == '0':
        return False

    if shared.sysinfo["location"]=='CN':
        if len(tele)>11 or not tele.isdigit() or tele[0] != '1':
            return False
    
    return True

def check_phrase(phrase):
    if len(phrase) < 8 or len(phrase) > 16:
        return False
    if not re.match(r'^[a-zA-Z0-9]+$', phrase):
        return False
    if not re.search(r'[a-z]', phrase) or not re.search(r'[A-Z]', phrase) or not re.search(r'[0-9]', phrase):
        return False
    
    return True

def check_vcode(vcode):
    if len(vcode)<4 or len(vcode)>6:
        return False
    return True
