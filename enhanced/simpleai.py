import os
import sys
import shutil
import re
import gradio as gr
import shared
import cv2
import modules.util as util
import enhanced.all_parameters as ads
from simpleai_base import simpleai_base, utils, comfyd, torch_version, xformers_version, comfyclient_pipeline
from simpleai_base.params_mapper import ComfyTaskParams
from simpleai_base.models_info import ModelsInfo, sync_model_info
from simpleai_base.simpleai_base import export_identity_qrcode_svg, import_identity_qrcode
from build_launcher import is_win32_standalone_build
import logging
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

utils.echo_off = not ads.get_admin_default('advanced_logs')
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
        xformers_version=xformers_version ))
    comfyclient_pipeline.COMFYUI_ENDPOINT_PORT = shared.sysinfo["loopback_port"]
    reserve_vram_value = ads.get_admin_default('reserved_vram')
    reserve_vram = [['--reserve-vram', f'{reserve_vram_value}']] if reserve_vram_value and reserve_vram_value>0 else [] 
    smart_memory = [] if shared.sysinfo['gpu_memory']<8180 else [['--disable-smart-memory']]
    windows_standalone = [["--windows-standalone-build"]] if is_win32_standalone_build else []
    fast_mode = [["--fast"]] if ads.get_admin_default('fast_comfyd_checkbox') else []
    args_comfyd = comfyd.args_mapping(sys.argv) + [["--listen"], ["--port", f'{shared.sysinfo["loopback_port"]}']] + smart_memory + windows_standalone + reserve_vram + fast_mode
    args_comfyd += [["--cuda-malloc"]] if not shared.args.disable_async_cuda_allocation and not shared.args.async_cuda_allocation else []
    comfyd_images_path = os.path.join(shared.path_userhome, 'guest_user')
    comfyd_output = os.path.join(comfyd_images_path, 'comfyd_outputs')
    comfyd_intput = os.path.join(comfyd_images_path, 'comfyd_inputs')
    if not os.path.exists(comfyd_output):
        os.makedirs(comfyd_output)
    if not os.path.exists(comfyd_intput):
        os.makedirs(comfyd_intput)
    comfyd_intput_default_image = os.path.join(comfyd_intput, 'welcome.png')
    if not os.path.exists(comfyd_intput_default_image):
        default_image_path = os.path.join(shared.root, 'presets/welcome/welcome.png')
        shutil.copy(default_image_path, comfyd_intput)
    args_comfyd += [["--output-directory", comfyd_output], ["--temp-directory", shared.temp_path], ["--input-directory", comfyd_intput]]
    #args_comfyd += [["--fast"]] if 'RTX 40' in shared.sysinfo['gpu_name'] else []
    comfyd.comfyd_args = args_comfyd
    return


def get_path_in_user_dir(filename, user_did=None, catalog=None):
    if user_did is None:
        user_did = shared.token.get_guest_did()
    if filename:
        path = catalog if catalog else filename
        path_file = shared.token.get_path_in_user_dir(user_did, path)
        if not os.path.exists(os.path.dirname(path_file)):
            for cata in ["presets", "workflows", "styles", "wildcards"]:
                os.makedirs(os.path.join(os.path.dirname(path_file), cata))
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

def change_advanced_logs(advanced_logs):
    if advanced_logs:
        utils.echo_off = False
    else:
        utils.echo_off = True

def get_echo_off():
    return utils.echo_off


identity_note = '您的昵称+手机号组成您的可信身份，昵称支持汉字。绑定身份即激活"我的预置"导航等高级服务。用身份二维码导入既安全又快捷。首个绑定身份者为系统管理员。'
identity_note_1 = '您的身份已绑定当前浏览器和本机节点。若要更换其他身份，需先"解除绑定"。导出身份二维码可方便再次绑定，适用离线和漫游场景，导出后请妥善保存。'
identity_note_0 = '孤岛节点只有游客和本机管理员身份。用"导出身份"可直接导出本地管理员身份二维码，身份口令见后台日志。'

upstream_tooltip = "请查看后台日志，排除错误后" if shared.upstream_did else "无法连接上游节点，请检查网络环境"
upstream_status_text = "<b>On</b>" if shared.upstream_did else "<b>Off</b>"
is_export_qr = lambda x: not shared.token.is_guest(x) or shared.token.get_node_mode()!='online'

note1_0 = '请按提示输入创建身份时预设的身份口令，确认身份后完成绑定。'
note1_1 = '未匹配到数字身份，请注意查收手机短信验证码，用其认证新身份。若十分钟未收到短信验证码，请点击"更换身份信息"，重新输入，再次绑定。'
note1_2 = f'已匹配到数字身份，{note1_0}'
note1_3 = f'已匹配到云端加密身份副本, {note1_0}'
note1_4 = '身份信息格式不对，昵称最少4个字符或2个汉字，国内手机号11位数字。请重新输入身份信息，再次绑定。'
note1_5 = '该手机号绑定身份数量超过上限，无法绑定该身份，请更换身份信息，再次申请绑定。'
note1_6 = '短时间内重复提交相同身份信息，请稍后再重新输入身份信息，再次申请绑定。'
note1_7 = '已提交过但未验证通过的身份，已清除遗留数据，需重新输入身份信息，再次绑定。'
note1_8 = f'找回加密副本或验证身份出错。{upstream_tooltip}，重启软件，再次绑定。'
note1_9 = '当前系统为孤岛节点，只能绑定本地管理员身份。请导入本地管理员身份二维码绑定。'

note2_0 = lambda x: f'口令最少8位字符，必须包含大写、小写字母和数字，不能有特殊字符。\n**<span style="color: {x};">特别提醒</span>**<span style="color: {x};">: 身份口令是唯一解锁数字身份的密钥，无法找回，遗失将导致已存储的配置信息和数据丢失，需妥善保存!!!</span>'
note2_1 = f'身份已验证，请按提示预设个人身份口令。{note2_0("lightseagreen")}'
note2_2 = f'口令遗失无法找回，请再次输入一致的口令。{note2_0("darkorange")}'
note2_3 = '身份验证码格式不对，请正确输入短信里的身份验证码，重新进行"身份验证"。'
note2_4 = '身份验证码不正确，请正确输入短信里的身份验证码，重新进行"身份验证"。'
note2_5 = f'设置的身份口令格式不对，请重新预设个人身份口令。{note2_0("lightseagreen")}'
note2_6 = f'身份口令设置异常，请重新输入身份信息进行绑定。'
note2_7 = f'身份口令与上次不一致，请重新预设个人身份口令。{note2_0("lightseagreen")}'
note2_8 = f'已匹配到本地数字身份，请按提示预设个人身份口令。{note2_0("lightseagreen")}'

note3 = f'绑定成功! {identity_note_1}'
note3_1 = '身份绑定不成功，请重新输入个人身份口令，再次确认身份。'
note3_2 = '输入的身份口令格式不对，最少8位字符，必须包含大写字母、小写字母和数字，不能有特殊字符。请重新输入个人身份口令。'

note4 = '身份已成功解绑，当前节点的服务已回退到游客模式。可以更换其他身份再次绑定。'
note4_1 = '身份解绑不成功，请重新输入个人身份口令，再次确认身份。'


theme_color = {
    "dark": "aqua",
    "light": "blue",
    }

id_info_css = lambda x: f'style="color: {theme_color[x]};"'
#lambda x: f'style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: inline-block; max-width: 150px; color: {theme_color[x]};"'
guest_inc = "身份为游客" if shared.token.get_node_mode()=='online' else "为孤岛节点游客"
user_inc = "绑定身份为" if shared.token.get_node_mode()=='online' else "为孤岛节点管理员"
current_id_info = lambda x,y,z,t: f'<b>当前{guest_inc if shared.token.is_guest(y) else user_inc}</b><br>身份昵称: <span {id_info_css(t)}>' + f'{x}' + f'</span><br>身份标识: <span {id_info_css(t)}>{y}</span><br>节点标识: <span {id_info_css(t)}>{z}</span>'

# [identity_note_info, input_identity, input_id_display, identity_vcode_input, identity_verify_button, identity_phrase_input, identity_phrases_set_button, identity_phrases_confirm_button, identity_confirm_button, identity_unbind_button]
# [identity_nick_input, identity_tele_input, identity_qr]

def trigger_input_identity(img):
    image = util.HWC3(img)
    qr_code_detector = cv2.QRCodeDetector()
    try:
        data, bbox, data_bytes = qr_code_detector.detectAndDecode(image)
        if bbox is not None:
            try:
                user_did, nickname, telephone = import_identity_qrcode(data)
            except Exception as e:
                logger.debug("qrcode parse error")
                user_did, nickname, telephone = '', '', ''
        else:
            user_did, nickname, telephone = '', '', ''
    except UnicodeDecodeError as e:
        logger.debug(f'bbox:{bbox}, data_bytes:{data_bytes}')
    return  bind_identity_sub(nickname, telephone, user_did)

def bind_identity(nick, areacode, tele):
    areacode = areacode.split('-')[0]
    tele = f'{areacode}{tele}'
    return bind_identity_sub(nick, tele)

def bind_identity_sub(nick, tele, user_did=None):
    logger.debug(f'nickename={nick}, telephone={tele}')
    if check_input(nick, tele):
        where = shared.token.check_local_user_token(nick, tele)
        logger.info(f'check_local_user_token:{where}')
        if where in ['local', 'recall']: # 本地或远程有身份, 输入身份口令
            result = [note1_2] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=False)]*2 +[gr.update(visible=True)] + [gr.update(visible=False)]
        elif where == 'create': # 新身份, 输入验证码
            result = [note1_1] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=True, value='')] + [gr.update(visible=True)] + [gr.update(visible=False)]*5
        elif where == 'immature': # 本地遗留密钥,重设身份口令
            result = [note2_8] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=True)] + [gr.update(visible=False)]*3
        elif where == 'unknown_exceeded': # 手机号绑定身份过多
            result = [note1_5] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
        elif where == 'unknown_repeat': # 重复提交
            result = [note1_6] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
        elif where == 're_input': # 之前提交过失败,需重新提交
            result = [note1_7] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
        elif where == 'isolated': # 孤岛节点, 提示导出管理员身份二维码
            result = [note1_9] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
        else:  # 过程出错, 重新输入绑定信息,再来
            result = [note1_8] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
    else: # 身份信息不合规, 重新输入
        result = [note1_4] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
    return result + [f'{nick}, {tele}, {user_did if user_did else ""}']

def change_identity():
    return [identity_note] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7 + ['', '86-CN-中国', '', None]


def verify_identity(input_id_info, state, vcode):
    if check_vcode(vcode):
        inputs = input_id_info.split(',')
        nick, tele = inputs[0].strip(), inputs[1].strip()
        next_cmd = shared.token.check_user_verify_code(nick, tele, vcode)
        logger.debug(f'check_user_verify_code:{next_cmd}')
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
            result = [note] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=True, value='')] + [gr.update(visible=True)] + [gr.update(visible=False)]*5
    else: # 验证码格式错误, 重新输入
        result = [note2_3] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=True, value='')] + [gr.update(visible=True)] + [gr.update(visible=False)]*5
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
                state["user"] = context
                state["sys_did"] = context.get_sys_did()
                state["sstoken"] = shared.token.get_user_sstoken(context.get_did(), state["ua_hash"])
                state["__session"] = state["sstoken"]
                note = f'身份口令设置成功，完成身份绑定。请牢记身份口令: `{phrase}` ，解除绑定或再次绑定都需要，建议抄写到私人笔记，仅限自己可见。及时导出身份二维码，方便再次绑定，导出后妥善保存。'
                result = [note] + [gr.update(visible=False)]*4 + [gr.update(visible=True, value="")] + [gr.update(visible=False)]*3 + [gr.update(visible=True)]
            else: # 设置身份口令失败, 重新设置
                result = [note2_6] + [gr.update(visible=True)] + [gr.update(visible=False)] + [gr.update(visible=False)]*7
        else: # 口令两次不一致, 重新设置
            result = [note2_7] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')]+ [gr.update(visible=True)] + [gr.update(visible=False)]*3
        state["user_phrase"] = ''
    id_info = current_id_info(state["user"].get_nickname(), state["user"].get_did(), state["sys_did"], state["__theme"])
    upstream_status = gr.update(visible=not is_export_qr(state["user"].get_did()), value=upstream_status_text)
    export_qr = gr.update(visible=is_export_qr(state["user"].get_did()))
    return result + [id_info, upstream_status, export_qr]

def confirm_identity(input_id_info, state, phrase):
    if check_phrase(phrase):
        inputs = input_id_info.split(',')
        nick, tele, user_did = inputs[0].strip(), inputs[1].strip(), inputs[2].strip()
        context = shared.token.get_user_context_with_phrase(nick, tele, user_did, phrase)
        if shared.token.is_guest(context.get_did()): # 口令不对, 绑定失败, 重新输入口令, 再次绑定
            result = [note3_1] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=False)]*2 +[gr.update(visible=True)] + [gr.update(visible=False)]
        else: # 绑定成功, 转解绑输入
            state["user"] = context
            state["sys_did"] = context.get_sys_did()
            state["sstoken"] = shared.token.get_user_sstoken(context.get_did(), state["ua_hash"])
            state["__session"] = state["sstoken"]
            result = [note3] + [gr.update(visible=False)]*4 + [gr.update(visible=True, value="")] + [gr.update(visible=False)]*3 + [gr.update(visible=True)]
    else: # 口令格式不对, 重新输入口令, 再次绑定
        result = [note3_2] + [gr.update(visible=False)] + [gr.update(visible=True)] + [gr.update(visible=False)]*2 + [gr.update(visible=True, value='')] + [gr.update(visible=False)]*2 +[gr.update(visible=True)] + [gr.update(visible=False)]
    id_info = current_id_info(state["user"].get_nickname(), state["user"].get_did(), state["sys_did"], state["__theme"])
    upstream_status = gr.update(visible=not is_export_qr(state["user"].get_did()), value=upstream_status_text)
    export_qr = gr.update(visible=is_export_qr(state["user"].get_did()))
    return result + [id_info, upstream_status, export_qr]

def unbind_identity(input_id_info, state, phrase):
    if check_phrase(phrase):
        context = shared.token.unbind_and_return_guest(state["user"].get_did(), phrase)
        if shared.token.is_guest(context.get_did()):
            state["user"] = context
            state["sys_did"] = context.get_sys_did()
            state["sstoken"] = shared.token.get_user_sstoken(context.get_did(), state["ua_hash"])
            state["__session"] = state["sstoken"]
            state["preset_store"] = False
            result = [note4, gr.update(visible=True)] + [gr.update(visible=False)]*8 + ['', '86-CN-中国', '', None]
        else: # 口令不对, 解绑失败, 重新输入口令, 再次解绑
            result = [note4_1] + [gr.update(visible=False)]*4 + [gr.update(visible=True, value="")] + [gr.update(visible=False)]*3 + [gr.update(visible=True)] + ['', '86-CN-中国', '', None]
    else: # 口令格式不对, 重新输入口令, 再次解绑
        result = [note3_2] + [gr.update(visible=False)]*4 + [gr.update(visible=True, value="")] + [gr.update(visible=False)]*3 +[gr.update(visible=True)] + ['', '86-CN-中国', '', None]
    id_info = current_id_info(state["user"].get_nickname(), state["user"].get_did(), state["sys_did"], state["__theme"])
    upstream_status = gr.update(visible=not is_export_qr(state["user"].get_did()), value=upstream_status_text)
    export_qr = gr.update(visible=is_export_qr(state["user"].get_did()))
    return result + [id_info, upstream_status, export_qr]


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
    is_guest = shared.token.is_guest(state["user"].get_did())
    result = [identity_note_0 if shared.token.get_node_mode()!='online' else identity_note if is_guest else identity_note_1] + [gr.update(visible=is_guest)] + [gr.update(visible=False)]*3 + [gr.update(visible=not is_guest)] + [gr.update(visible=False)]*3 + [gr.update(visible=not is_guest)] + ['', '86-CN-中国', '', None]
    upstream_status = gr.update(visible=not is_export_qr(state["user"].get_did()), value=upstream_status_text)
    export_qr = gr.update(visible=is_export_qr(state["user"].get_did()))
    result = [gr.update(visible=not flag), current_id_info(state["user"].get_nickname(), state["user"].get_did(), state["sys_did"], state["__theme"]), upstream_status, export_qr] + result
    return result

def check_input(nick, tele):
    length = 0
    for n in nick:
        length += 1
        if util.is_chinese(n):
            length += 1
    if length < 4 or length > 24:
        return False
    
    if len(tele)<8 or len(tele)>15 or not tele.isdigit() or tele[0] == '0':
        return False

    if tele.startswith('86') and tele!='8610000000001':
        if len(tele)!=13 or not tele.isdigit() or tele[2] != '1' or tele[3] in ['0', '1', '2']:
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

self_contact = '''
软件特点:<br>
<b>SimpAI</b>，开源的中文 AI 创意生图平台，简洁、高效而稳定。<br>
<b>SimpAI</b>，开放架构，支持SDXL/SD1.5/Flux/可图等多种开源模型，丰富的控图方式，真正“所想即所得，创意无边界”。<br>
<b>SimpAI</b>，程序和模型本地部署，保障数据安全与可控；按需加载不同的场景预置包，贴合实际需求，操作简洁、高效。<br>
<b>SimpAI</b>，引入分布式身份管理机制，让本地部署的节点具有多用户服务和多节点组网能力，具备了企业级服务的特性。<br>
<br>
使用技巧:<br>
Wiki: <a target= "_blank" href="http://simpai.cn">http://simpai.cn</a><br>
开源代码:<br>
Github: <a target= "_blank" href="https://github.com/metercai/SimpleSDXL">https://github.com/metercai/SimpleSDXL</a><br>
学习交流:<br>
QQ群: 938075852<br>
商务合作:<br>
邮箱: 925457@qq.com
'''
