import os
import json
import gradio as gr
import modules.util as util
import modules.config as config
import modules.flags
import modules.sdxl_styles
import numbers
import copy
import re
import args_manager
import enhanced.gallery as gallery_util
from enhanced.models_info import models_info, models_info_muid
from modules.model_loader import load_file_from_url, load_file_from_muid


css = '''
.top_nav{
    height: 18px;
    position: relative;
}

.top_nav_left {
    position:absolute;
    left: 0;
    margin: 0;
    padding: 0;
    list-style: none;
}
.top_nav_right {
    position:absolute;
    right:0;
    margin: 0;
    padding: 0;
    list-style: none;
}

.top_nav_preset {
    display: inline-block;
    text-align: center;
    width: 60px;
    cursor:pointer;
    border-right: 1px solid rgb(161, 158, 158);
}
.top_nav_preset:hover {
    background-color: rgb(161, 158, 158);
}
.top_nav_preset:last-child{
     border-right: 0;
}

.top_nav_theme {
    display: inline-block;
    text-align: center;
    width: 40px;
    cursor:pointer;
}

.systemMsg {
    position: absolute;
    z-index: 1002;
    top: 0%;
    left: 5%;
    right: 5%;
    height: 45px;
    overflow: auto;
    user-select: none;
    -webkit-user-select: none;
}

.systemMsgBox {
    position: absolute;
    left: 24px;
    right: 24px;
    top: 2px;
    font-size: 12px;
}
.systemMsgBox::-webkit-scrollbar {
    border: 1px !important;
}

.systemMsgClose {
    position: absolute;
    top: 0px;
    right: 5px;
}

iframe::-webkit-scrollbar {
    display: none;
}
'''


nav_html = '''
<div class="top_nav">
    <ul class="top_nav_left" id="top_preset">
    *item_list*
    </ul>
    <ul class="top_nav_right" id="top_theme">
    <b>Theme:</b>
    <li class="top_nav_theme" id="theme_light" onmouseover="nav_mOver(this)" onmouseout="nav_mOut(this)" onclick="refresh_theme('light')">light</li>
    <li class="top_nav_theme" id="theme_dark" onmouseover="nav_mOver(this)" onmouseout="nav_mOut(this)" onclick="refresh_theme('dark')">dark</li>
    </ul>
</div>
'''


# app context
nav_id_list = ''
nav_preset_html = ''
system_message = ''
config_ext = {}


def make_html():
    global nav_id_list, nav_preset_html

    path_preset = os.path.abspath(f'./presets/')
    presets = util.get_files_from_folder(path_preset, ['.json'], None)
    del presets[presets.index('default.json')]
    file_times = [(f, os.path.getmtime(os.path.join(path_preset, f))) for f in presets]
    sorted_file_times = sorted(file_times, key=lambda x: x[1], reverse=True)
    sorted_files = [f[0] for f in sorted_file_times]
    sorted_files.insert(0, 'default.json')
    presets = sorted_files[:9]
    item_list = '<b>Preset:</b>'
    id_array = ''
    for i in range(len(presets)):
        item_list += f'<li class="top_nav_preset" id="preset_{presets[i][:-5]}" onmouseover="nav_mOver(this)" onmouseout="nav_mOut(this)" onclick="refresh_preset(\'{presets[i][:-5]}\')">{presets[i][:-5]}</li>'
        id_array += f'preset_{presets[i][:-5]},'
    id_array += 'theme_light,theme_dark'
    nav_id_list = id_array
    nav_preset_html = item_list
    return nav_html.replace('*item_list*', item_list)


def get_system_message():
    fooocus_log = os.path.abspath(f'./update_log.md')
    simplesdxl_log = os.path.abspath(f'./simplesdxl_log.md')
    enhanced_config = os.path.abspath(f'./enhanced/config.json')
    if os.path.exists(enhanced_config):
        with open(enhanced_config, "r", encoding="utf-8") as json_file:
            config_ext.update(json.load(json_file))
    else:
        config_ext.update({'fooocus_line': '# 2.1.852', 'simplesdxl_line': '# 2023-12-20'})
    update_msg_f = ''
    first_line_f = None
    if os.path.exists(fooocus_log):
        with open(fooocus_log, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            while line:
                if line == '\n':
                    line = log_file.readline()
                    continue
                if line.startswith("# ") and first_line_f is None:
                    first_line_f = line.strip()
                if line.strip() == config_ext['fooocus_line']:
                    break
                if first_line_f:
                    update_msg_f += line
                line = log_file.readline()
    update_msg_s = ''
    first_line_s = None
    if os.path.exists(simplesdxl_log):
        with open(simplesdxl_log, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            while line:
                if line == '\n':
                    line = log_file.readline()
                    continue
                if line.startswith("# ") and first_line_s is None:
                    first_line_s = line.strip()
                if line.strip() == config_ext['simplesdxl_line']:
                    break
                if first_line_s:
                    update_msg_s += line
                line = log_file.readline()
    update_msg_f = update_msg_f.replace("\n","  ")
    update_msg_s = update_msg_s.replace("\n","  ")
    
    f_log_path = os.path.abspath("./update_log.md")
    s_log_path = os.path.abspath("./simplesdxl_log.md")
    if len(update_msg_f)>0:
        body_f = f'<b id="update_f">[Fooocus最新更新]</b>: {update_msg_f}<a href="{args_manager.args.webroot}/file={f_log_path}">更多>></a>   '
    else:
        body_f = '<b id="update_f"> </b>'
    if len(update_msg_s)>0:
        body_s = f'<b id="update_s">[SimpleSDXL最新更新]</b>: {update_msg_s}<a href="{args_manager.args.webroot}/file={s_log_path}">更多>></a>'
    else:
         body_s = '<b id="update_s"> </b>'
    import mistune
    body = mistune.html(body_f+body_s)
    if first_line_f and first_line_s and (first_line_f != config_ext['fooocus_line'] or first_line_s != config_ext['simplesdxl_line']):
        config_ext['fooocus_line']=first_line_f
        config_ext['simplesdxl_line']=first_line_s
        with open(enhanced_config, "w", encoding="utf-8") as config_file:
            json.dump(config_ext, config_file)
    return body if body else ''


def preset_instruction(state_params):
    preset = state_params["__preset"]
    preset_url = state_params["preset_url"]
    head = "<div style='max-width:100%; max-height:128px; overflow:hidden'>"
    foot = "</div>"
    p_name = preset
    if p_name == 'default':
        p_name = '默认'
    body = f'"{p_name}"预置包:<span style="position: absolute;right: 0;"><a href="https://gitee.com/metercai/SimpleSDXL/blob/SimpleSDXL/presets/readme.md">\U0001F4DD 什么是预置包</a></span>'
    preset_url_str = ''
    if preset_url:
        append_str = f'__theme={config.theme}&__lang={args_manager.args.language}'
        preset_url_str = f'{preset_url}&{append_str}' if preset_url.count('?') else f'{preset_url}?{append_str}'
    body += f'<iframe src="{preset_url_str}" frameborder="0" scrolling="no" width="100%"></iframe>'
    return head + body + foot


def embeddings_model_split(prompt, prompt_negative):
    prompt_tags = re.findall(r'[\(](.*?)[)]', prompt_negative) + re.findall(r'[\(](.*?)[)]', prompt)
    embeddings = []
    for e in prompt_tags:
        embed = e.split(':')
        if len(embed)>2 and embed[0] == 'embedding':
            embeddings += [embed[1]]
    embeds = []
    for k in models_info.keys():
            if k.startswith('embeddings') and k[11:].split('.')[0] in embeddings:
                embeds += [k]
    return embeds


get_preset_params_js = '''
function(preset_params) {
    var preset=preset_params["__preset"];
    var theme=preset_params["__theme"];
    var nav_id_list=preset_params["__nav_id_list"];
    var nav_preset_html = preset_params["__nav_preset_html"];
    update_topbar("top_preset",nav_preset_html);
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    if (url_params["__preset"]!=null) {
        preset=url_params["__preset"];
    }
    if (url_params["__theme"]!=null) {
        theme=url_params["__theme"];
    }
    mark_position_for_topbar(nav_id_list,preset,theme);
    preset_params["__preset"]=preset;
    preset_params["__theme"]=theme;
    return preset_params;
}
'''


toggle_system_message_js = '''
function(system_params) {
    var message=system_params["__message"];
    if (message!=null && message.length>60) {
        showSysMsg(message);
    }
    return
}
'''


sync_generating_state_js = '''
function(system_params) {
    if (system_params["__generating_state"])
        c_generating_state = 1;
    else
        c_generating_state = 0;
    return
}
'''


def sync_generating_state_true(state_params):
    state_params.update({'__generating_state':1})
    return state_params, state_params


def sync_generating_state_false(state_params):
    state_params.update({'__generating_state':0})
    output_index = gallery_util.output_list[0].split('/')[0]
    gallery_util.refresh_images_list(output_index, True)
    gallery_util.parse_html_log(output_index, True)
    return state_params, state_params


def sync_message(state_params):
    state_params.update({"__message":system_message})
    return state_params

def reset_context(state_params):
    global system_message

    preset = state_params.get("__preset")
    theme = state_params.get("__theme")
    print(f'[Topbar] Reset_context: preset={config.preset}-->{preset}, theme={config.theme}-->{theme}')
    config_org = {}
    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    config_org = json.load(json_file)
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    if 'reference' in config_org.keys():
        preset_url = config_org["reference"]
    else:
        if 'reference' in config.config_dict.keys():
            config.config_dict.pop("reference")
        preset_inc_path = os.path.abspath(f'./presets/{preset}.inc.html')
        if os.path.exists(preset_inc_path):
            preset_url = f'{args_manager.args.webroot}/file={preset_inc_path}'
        else:
            preset_url = ''
    
    down_muid = {}
    for k in config_org["checkpoint_downloads"].keys():
        if config_org["checkpoint_downloads"][k].startswith('MUID:'):
            down_muid.update({"checkpoints/"+k: config_org["checkpoint_downloads"][k][5:]})
    for k in config_org["embeddings_downloads"].keys():
        if config_org["embeddings_downloads"][k].startswith('MUID:'):
            down_muid.update({"embeddings/"+k: config_org["embeddings_downloads"][k][5:]})
    for k in config_org["lora_downloads"].keys():
        if config_org["lora_downloads"][k].startswith('MUID:'):
            down_muid.update({"loras/"+k: config_org["lora_downloads"][k][5:]})

    embeddings = embeddings_model_split(config_org["default_prompt"], config_org["default_prompt_negative"])
    checklist = ["checkpoints/"+config_org["default_model"], "checkpoints/"+config_org["default_refiner"]] + ["loras/"+n for i, (n, v) in enumerate(config_org["default_loras"])]
    checklist += embeddings

    newlist = []
    downlist = []
    not_MUID = False
    for i in range(len(checklist)):
        f = checklist[i]
        filename = f
        if f and f != "checkpoints/None" and f != "loras/None":
            if f not in models_info.keys():
                if f in down_muid.keys() and down_muid[f] in models_info_muid.keys():
                    filename = models_info_muid[down_muid[f]]
                    print(f'[Topbar] The local file {filename.split("/")[1]} is the same as in preset {f.split("/")[1]}, replace it with the local file.')
                else:
                    downlist += [f]
            else:
                if not models_info[f]['muid']:
                    not_MUID = True
        newlist += [filename]

    if downlist:
        print(f'[Topbar] The model file in preset is not local, ready to download.')
        for f in downlist:
            if f in down_muid:
                if f.startswith("checkpoints/"):
                    file_path = os.path.join(config.path_checkpoints, f[12:])
                elif f.startswith("loras/"):
                    file_path = os.path.join(config.path_loras, f[6:])
                elif f.startswith("embeddings/"):
                    file_path = os.path.join(config.path_embeddings, f[11:])
                else:
                    file_path = os.path.abspath(f'./models/{f}')
                model_dir, filename = os.path.split(file_path)
                load_file_from_muid(filename, down_muid[f], model_dir)
            elif f[12:] in config_org["checkpoint_downloads"]:
                load_file_from_url(url=config_org["checkpoint_downloads"][f[12:]], model_dir=config.path_checkpoints, file_name=f[12:])
            elif f[6:] in config_org["lora_downloads"]:
                load_file_from_url(url=config_org["lora_downloads"][f[6:]], model_dir=config.path_loras, file_name=f[6:])
            elif f[11:] in config_org["embeddings_downloads"]:
                load_file_from_url(url=config_org["embeddings_downloads"][f[11:]], model_dir=config.path_embeddings, file_name=f[11:])
            else:
                print(f'[Topbar] The model file in preset is not local and cannot be download.')

    if not_MUID:
        print(f'[Topbar] The preset contains model file without MUID, need to sync model info for usability and transferability.')
    
    new_loras = []
    for i in range(len(newlist)):
        if newlist[i] != checklist[i]:
            if i==0:
                config_org["default_model"]=newlist[i][12:]
            elif i==1:
                config_org["default_refiner"]=newlist[i][12:]
            elif i>1 and i<7:
                new_loras += [[newlist[i][6:], config_org["default_loras"][i-2][1]]]
            else:
                embedding_new = newlist[i][11:].split('.')[0]
                embedding_old = checklist[i][11:].split('.')[0]
                config_org["default_prompt"].replace("(embedding:"+embedding_new+":", "(embedding:"+embedding_old+":")
                config_org["default_prompt_negative"].replace("(embedding:"+embedding_new+":", "(embedding:"+embedding_old+":")
        elif i>1 and i<7:
            new_loras += [[newlist[i][6:], config_org["default_loras"][i-2][1]]]
    config_org["default_loras"] = new_loras
    
    config.config_dict.update(config_org)
    reset_default_config()

    config.theme = theme
    config.preset = preset
    state_params.update({"preset_url":preset_url})

    results = []
    results += [gr.update(value=config.default_base_model_name), \
                gr.update(value=config.default_refiner_model_name), \
                gr.update(value=config.default_refiner_switch), \
                gr.update(value=config.default_cfg_scale), \
                gr.update(value=config.default_sample_sharpness), \
                gr.update(value=config.default_sampler), \
                gr.update(value=config.default_scheduler), \
                gr.update(value=config.default_performance), \
                gr.update(value=config.default_prompt), \
                gr.update(value=config.default_prompt_negative), \
                gr.update(value=copy.deepcopy(config.default_styles)), \
                gr.update(value=config.add_ratio(config.default_aspect_ratio))]
    for i, (n, v) in enumerate(config.default_loras):
        results += [gr.update(value=n),gr.update(value=v)]
    results += [gr.update(), gr.update(choices=gallery_util.output_list, value=None if len(gallery_util.output_list)==0 else gallery_util.output_list[0])]
    results += [gr.update(visible=True if preset_url else False, value=preset_instruction(state_params))]
    state_params.update({"__message":system_message})
    results += [state_params]
    system_message = 'system message was displayed!'

    return results


def reset_default_config():
    config.default_base_model_name = config.get_config_item_or_set_default(
        key='default_model',
        default_value='juggernautXL_version6Rundiffusion.safetensors',
        validator=lambda x: isinstance(x, str)
    )
    config.default_refiner_model_name = config.get_config_item_or_set_default(
        key='default_refiner',
        default_value='None',
        validator=lambda x: isinstance(x, str)
    )
    config.default_refiner_switch = config.get_config_item_or_set_default(
        key='default_refiner_switch',
        default_value=0.5,
        validator=lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1
    )
    config.default_loras = config.get_config_item_or_set_default(
        key='default_loras',
        default_value=[
            [
                "sd_xl_offset_example-lora_1.0.safetensors",
                0.1
            ],
            [
                "None",
                1.0
            ],
            [
                "None",
                1.0
            ],
            [
                "None",
                1.0
            ],
            [
                "None",
                1.0
            ]
        ],
        validator=lambda x: isinstance(x, list) and all(len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number) for y in x)
    )
    config.default_cfg_scale = config.get_config_item_or_set_default(
        key='default_cfg_scale',
        default_value=4.0,
        validator=lambda x: isinstance(x, numbers.Number)
    )
    config.default_sample_sharpness = config.get_config_item_or_set_default(
        key='default_sample_sharpness',
        default_value=2.0,
        validator=lambda x: isinstance(x, numbers.Number)
    )
    config.default_sampler = config.get_config_item_or_set_default(
        key='default_sampler',
        default_value='dpmpp_2m_sde_gpu',
        validator=lambda x: x in modules.flags.sampler_list
    )
    config.default_scheduler = config.get_config_item_or_set_default(
        key='default_scheduler',
        default_value='karras',
        validator=lambda x: x in modules.flags.scheduler_list
    )
    config.default_styles = config.get_config_item_or_set_default(
        key='default_styles',
        default_value=[
            "Fooocus V2",
            "Fooocus Enhance",
            "Fooocus Sharp"
        ],
        validator=lambda x: isinstance(x, list) and all(y in modules.sdxl_styles.legal_style_names for y in x)
    )
    config.default_prompt_negative = config.get_config_item_or_set_default(
        key='default_prompt_negative',
        default_value='',
        validator=lambda x: isinstance(x, str),
        disable_empty_as_none=True
    )
    config.default_prompt = config.get_config_item_or_set_default(
        key='default_prompt',
        default_value='',
        validator=lambda x: isinstance(x, str),
        disable_empty_as_none=True
    )
    config.default_performance = config.get_config_item_or_set_default(
        key='default_performance',
        default_value='Speed',
        validator=lambda x: x in modules.flags.performance_selections
    )
    config.default_advanced_checkbox = config.get_config_item_or_set_default(
        key='default_advanced_checkbox',
        default_value=True,
        validator=lambda x: isinstance(x, bool)
    )
    config.default_image_number = config.get_config_item_or_set_default(
        key='default_image_number',
        default_value=4,
        validator=lambda x: isinstance(x, int) and 1 <= x <= 32
    )
    config.checkpoint_downloads = config.get_config_item_or_set_default(
        key='checkpoint_downloads',
        default_value={
            "juggernautXL_version6Rundiffusion.safetensors": "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors"
        },
        validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
    )
    config.lora_downloads = config.get_config_item_or_set_default(
        key='lora_downloads',
        default_value={
            "sd_xl_offset_example-lora_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors"
        },
        validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
    )
    config.embeddings_downloads = config.get_config_item_or_set_default(
        key='embeddings_downloads',
        default_value={},
        validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
    )
    config.available_aspect_ratios = config.get_config_item_or_set_default(
        key='available_aspect_ratios',
        default_value=[
            '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
            '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
            '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
            '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
            '1664*576', '1728*576'
        ],
        validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1
    )
    config.default_aspect_ratio = config.get_config_item_or_set_default(
        key='default_aspect_ratio',
        default_value='1152*896' if '1152*896' in config.available_aspect_ratios else config.available_aspect_ratios[0],
        validator=lambda x: x in config.available_aspect_ratios
    )
    config.default_inpaint_engine_version = config.get_config_item_or_set_default(
        key='default_inpaint_engine_version',
        default_value='v2.6',
        validator=lambda x: x in modules.flags.inpaint_engine_versions
    )
    config.default_cfg_tsnr = config.get_config_item_or_set_default(
        key='default_cfg_tsnr',
        default_value=7.0,
        validator=lambda x: isinstance(x, numbers.Number)
    )
    config.default_overwrite_step = config.get_config_item_or_set_default(
        key='default_overwrite_step',
        default_value=-1,
        validator=lambda x: isinstance(x, int)
    )
    config.default_overwrite_switch = config.get_config_item_or_set_default(
        key='default_overwrite_switch',
        default_value=-1,
        validator=lambda x: isinstance(x, int)
    )
    config.example_inpaint_prompts = config.get_config_item_or_set_default(
        key='example_inpaint_prompts',
        default_value=[
            'highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes'
        ],
        validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x)
    )

    config.example_inpaint_prompts = [[x] for x in config.example_inpaint_prompts]

    config.config_dict["default_loras"] = config.default_loras = config.default_loras[:5] + [['None', 1.0] for _ in range(5 - len(config.default_loras))]

    return


system_message = get_system_message()
