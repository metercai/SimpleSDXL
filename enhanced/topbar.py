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
import random
import modules.constants as constants
import modules.advanced_parameters as ads
import modules.sdxl_styles as sdxl_styles
import modules.style_sorter as style_sorter
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

.preset_bar {
    text-align: center;
    padding: 0px;
}

.bar_title {
    width: 60px !important;
    padding: 0px;
    position: absolute;
    left: 5px;
    top: 2px
}

.bar_button {
    width: 80px !important;
    padding: 0px;
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
nav_name_list = ''
nav_id_list = ''
nav_preset_html = ''
system_message = ''
config_ext = {}


def make_html():
    global nav_name_list, nav_id_list, nav_preset_html

    path_preset = os.path.abspath(f'./presets/')
    presets = [p for p in util.get_files_from_folder(path_preset, ['.json'], None) if not p.startswith('.') and p!='default.json']
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
        nav_name_list += f'{presets[i][:-5]},'
    nav_name_list = nav_name_list[:-1]
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
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    if (url_params["__preset"]!=null) {
        preset=url_params["__preset"];
    }
    if (url_params["__theme"]!=null) {
        theme=url_params["__theme"];
    }
    preset_params["__preset"]=preset;
    preset_params["__theme"]=theme;
    return preset_params, preset_params;
}
'''


toggle_system_message_js = '''
function(system_params) {
    var nav_preset_html = system_params["__nav_preset_html"];
    update_topbar("top_preset",nav_preset_html);
    var preset=system_params["__preset"];
    var theme=system_params["__theme"];
    var nav_name_list_str = system_params["__nav_name_list"];
    var nav_name_list = new Array();
    nav_name_list = nav_name_list_str.split(",")
    console.log("nav_name_list_str:"+nav_name_list_str)
    for (var i=0;i<nav_name_list.length;i++) {
        var item_id = "bar"+i;
        var item_name = nav_name_list[i];
        var nav_item = gradioApp().getElementById(item_id);
         console.log("item_id:"+item_id+", item_name:"+item_name+", nav_item:"+nav_item)
        if (nav_item!=null) {
            nav_item.innerHTML = item_name;
            if (item_name != preset) {
                if (theme == "light") {
                    nav_item.style.color = 'var(--neutral-400)';
                    nav_item.style.backgroundColor= 'var(--neutral-50)';
                } else {
                    nav_item.style.color = 'var(--neutral-400)';
                    nav_item.style.backgroundColor= 'var(--neutral-950';
                }
            } else {
                if (c_theme_id == 'theme_light') {
                    nav_item.style.color = 'var(--neutral-800)';
                    nav_item.style.backgroundColor= 'var(--secondary-200)';
                } else {
                    nav_item.style.color = 'var(--neutral-50)';
                    nav_item.style.backgroundColor= 'var(--secondary-400';
                }
            }
        }
    }
    var message=system_params["__message"];
    if (message!=null && message.length>60) {
        showSysMsg(message);
    }
    var nav_id_list=system_params["__nav_id_list"];
    mark_position_for_topbar(nav_id_list,preset,theme);
    //var infobox=gradioApp().getElementById("infobox");
    //if (infobox!=null) {
        //css = infobox.getAttribute("class")
        //console.log("infobox.css="+css)
        //if (browser.device.is_mobile && css.indexOf("infobox_mobi")<0)
          //  infobox.setAttribute("class", css.replace("infobox", "infobox_mobi"));
    //}
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


def init_nav_bar(system_params, state_params):
    preset_name_list = system_params["__nav_name_list"].split(',')
    for i range(9-len(preset_name_list)):
        preset_name_list.append('')
    print(f'preset_name_list:{preset_name_list}')
    state_params = system_params
    results = []
    for name in preset_name_list:
        results += gr.update(value=name)
    return results + [state_params]


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


def reset_params_for_preset(bar_button, state_params):
    print(f'bar_button:{bar_button}')
    state_params.update({"__preset": bar_button})
    return reset_context(state_params)


def reset_context(state_params):
    global system_message, nav_id_list, nav_preset_html

    preset = state_params.get("__preset")
    theme = state_params.get("__theme")
    preset_url = state_params.get("preset_url", '')
    print(f'[Topbar] Reset_context: preset={config.preset}-->{preset}, theme={config.theme}-->{theme}')
    
    # load preset config
    config_preset = {}
    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    config_preset = json.load(json_file)
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    if 'reference' in config_preset.keys():
        preset_url = config_preset["reference"]
    else:
        if 'reference' in config.config_dict.keys():
            config.config_dict.pop("reference")
        preset_inc_path = os.path.abspath(f'./presets/{preset}.inc.html')
        if os.path.exists(preset_inc_path):
            preset_url = f'{args_manager.args.webroot}/file={preset_inc_path}'
        else:
            preset_url = ''
    
    info_preset = {}
    keys = config_preset.keys()
    info_preset.update({
        "Prompt": '' if 'default_prompt' not in keys else config_preset["default_prompt"],
        "Negative Prompt": '' if 'default_prompt_negativ' not in keys else config_preset["default_prompt_negative"],
        "Styles": "['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp']" if 'default_prompt_negativ' not in keys else config_preset["default_styles"],
        "Performance": 'Speed' if 'default_performance' not in keys else config_preset["default_performance"],
        "Sharpness": '2.0' if 'default_sample_sharpness' not in keys else config_preset["default_sample_sharpness"],
        "Guidance Scale": '4.0' if 'default_cfg_scale' not in keys else config_preset["default_cfg_scale"],
        "ADM Guidance": f'({ads.default["adm_scaler_positive"]}, {ads.default["adm_scaler_negative"]}, {ads.default["adm_scaler_end"]})',
        "Base Model": 'juggernautXL_version6Rundiffusion.safetensors' if 'default_model' not in keys else config_preset["default_model"],
        "Refiner Model": 'None' if 'default_refiner' not in keys else config_preset["default_refiner"],
        "Refiner Switch": '0.5' if 'default_refiner_switch' not in keys else config_preset["default_refiner_switch"], 
        "Sampler": f'{ads.default["sampler_name"]}' if 'default_sampler' not in keys else config_preset["default_sampler"],
        "Scheduler": f'{ads.default["scheduler_name"]}' if 'default_scheduler' not in keys else config_preset["default_scheduler"],
        "Seed": f'{random.randint(constants.MIN_SEED, constants.MAX_SEED)}' if 'default_seed' not in keys else config_preset["default_seed"]
        })
    if "default_aspect_ratio" in keys and config_preset["default_aspect_ratio"] in config.available_aspect_ratios:
        aspect_ratio = config_preset["default_aspect_ratio"].split('*')
        info_preset.update({'Resolution': f'({aspect_ratio[0]}, {aspect_ratio[1]})'})
    else:
        info_preset.update({"Resolution": '(1152, 896)'})
    if "default_loras" in keys:
        loras = [(n,v) for i, (n, v) in enumerate(config_preset["default_loras"]) if n!='None']
        for (n,v) in loras:
            info_preset.update({f'LoRA [{n}] weight': f'{v}'})
    if "checkpoint_downloads" in keys:
        info_preset.update({'checkpoint_downloads': config_preset["checkpoint_downloads"]})
    if "lora_downloads" in keys:
        info_preset.update({'lora_downloads': config_preset["lora_downloads"]})
    if "embeddings_downloads" in keys:
        info_preset.update({'embeddings_downloads': config_preset["embeddings_downloads"]})
    if "styles_description" in keys:
        info_preset.update({'styles_description': config_preset["styles_description"]})

    ads_params = {}
    if 'default_cfg_tsnr' in keys:
        ads_params.update({"adaptive_cfg": config_preset["default_cfg_tsnr"]})
    if 'default_overwrite_step' in keys:
        ads_params.update({"overwrite_step": config_preset["default_overwrite_step"]})
    if 'default_overwrite_switch' in keys:
        ads_params.update({"overwrite_switch": config_preset["default_overwrite_switch"]})
    if 'default_inpaint_engine' in keys:
        ads_params.update({"inpaint_engine": config_preset["default_inpaint_engine"]})
    if len(ads_params.keys())>0:
        info_preset.update({'Advanced_parameters': ads_params})

#other
#"available_aspect_ratios": []
#"default_advanced_checkbox": true,
#"default_max_image_number": 32,
#"default_image_number": 2,
#"example_inpaint_prompts":[]
    results = reset_params(check_prepare_for_reset(info_preset))[:26]

    config.theme = theme
    config.preset = preset
    state_params.update({"preset_url":preset_url})

    results += [gr.update(), gr.update(choices=gallery_util.output_list, value=None if len(gallery_util.output_list)==0 else gallery_util.output_list[0])]
    results += [gr.update(visible=True if preset_url else False, value=preset_instruction(state_params))]
    state_params.update({"__message": system_message})
    state_params.update({"__nav_id_list": nav_id_list})
    state_params.update({"__nav_preset_html": nav_preset_html})
    results += [state_params]
    system_message = 'system message was displayed!'
    return results


def check_prepare_for_reset(info):
    # download urls with MUID
    down_muid = {}
    if "checkpoint_downloads" in info.keys():
        for k in info["checkpoint_downloads"].keys():
            if info["checkpoint_downloads"][k].startswith('MUID:'):
                down_muid.update({"checkpoints/"+k: info["checkpoint_downloads"][k][5:]})
    if "lora_downloads" in info.keys():
        for k in info["lora_downloads"].keys():
            if info["lora_downloads"][k].startswith('MUID:'):
                down_muid.update({"loras/"+k: info["lora_downloads"][k][5:]})
    if "embeddings_downloads" in info.keys():
        for k in info["embeddings_downloads"].keys():
            if info["embeddings_downloads"][k].startswith('MUID:'):
                down_muid.update({"embeddings/"+k: info["embeddings_downloads"][k][5:]})

    # the models to be checked 
    info['Refiner Model'] = None if info['Refiner Model']=='' else info['Refiner Model']
    loras = [['None', 1.0], ['None', 1.0], ['None', 1.0], ['None', 1.0], ['None', 1.0]]
    for key in info:
        i=0
        if key.startswith('LoRA ['):
            loras.insert(i, [key[6:-8], float(info[key])])
    loras = loras[:5]
    embeddings = embeddings_model_split(info["Prompt"], info["Negative Prompt"])
    checklist = ["checkpoints/"+info["Base Model"], "checkpoints/"+info["Refiner Model"]] + ["loras/"+n for i, (n, v) in enumerate(loras)]
    checklist += embeddings

    # check if the models is in local model path
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
                    print(f'[Topbar] The local file {filename.split("/")[1]} is the same as in reset data {f.split("/")[1]}, replace it with the local file.')
                else:
                    downlist += [f]
            else:
                if not models_info[f]['muid']:
                    print(f'[Topbar] The file:{f} in local is missing MUID!')
                    not_MUID = True
        newlist += [filename]

    # download the missing model
    if downlist:
        print(f'[Topbar] The model is not local, ready to download.')
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
            elif f[12:] in info["checkpoint_downloads"]:
                load_file_from_url(url=info["checkpoint_downloads"][f[12:]], model_dir=config.path_checkpoints, file_name=f[12:])
            elif f[6:] in info["lora_downloads"]:
                load_file_from_url(url=info["lora_downloads"][f[6:]], model_dir=config.path_loras, file_name=f[6:])
            elif f[11:] in info["embeddings_downloads"]:
                load_file_from_url(url=info["embeddings_downloads"][f[11:]], model_dir=config.path_embeddings, file_name=f[11:])
            else:
                print(f'[Topbar] The model is not local and cannot be download.')

    if not_MUID:
        print(f'[Topbar] The reset request contains model file without MUID, need to sync model info for usability and transferability.')

    # replace to local model filename
    new_loras = []
    for i in range(len(newlist)):
        if newlist[i] != checklist[i]:
            if i==0:
                info["Base Model"]=newlist[i][12:]
            elif i==1:
                info["Refiner Model"]=newlist[i][12:]
            elif i>1 and i<7:
                new_loras += [[newlist[i][6:], loras[i-2][1]]]
            else:
                embedding_new = newlist[i][11:].split('.')[0]
                embedding_old = checklist[i][11:].split('.')[0]
                info["Prompt"].replace("(embedding:"+embedding_new+":", "(embedding:"+embedding_old+":")
                info["Negative Prompt"].replace("(embedding:"+embedding_new+":", "(embedding:"+embedding_old+":")
        elif i>1 and i<7:
            new_loras += [[newlist[i][6:], loras[i-2][1]]]
    loras = new_loras

    styles_update_flag = False
    if "styles_description" in info.keys():
        for key in info["styles_description"].keys():
            if key not in sdxl_styles.styles.keys():
                sdxl_styles.styles.update({key: info["styles_description"][key]})
                styles_update_flag = True
                print(f'[Topbar] New styles: {key} to be loaded in reset process!')
    info.update({'loras': loras})
    info.update({'styles_update_flag': styles_update_flag})
    return info

def reset_params(info):
    print(f'[Topbar] Ready to reset generation params session based.')
    aspect_ratios = info['Resolution'][1:-1].replace(', ', '*')
    adm_scaler_positive, adm_scaler_negative, adm_scaler_end = [float(f) for f in info['ADM Guidance'][1:-1].split(', ')]
    get_ads_value_or_default = lambda x: ads.default[x] if 'Advanced_parameters' not in info.keys() or x not in info['Advanced_parameters'].keys() else info['Advanced_parameters'][x]
    adaptive_cfg = get_ads_value_or_default('adaptive_cfg')
    overwrite_step = get_ads_value_or_default('overwrite_step')
    overwrite_switch = get_ads_value_or_default('overwrite_switch')
    inpaint_engine = get_ads_value_or_default('inpaint_engine')
    styles = [f[1:-1] for f in info['Styles'][1:-1].split(', ')]

    results = []
    results += [gr.update(value=info['Prompt']), gr.update(value=info['Negative Prompt'])]
    if 'styles_update_flag' in info.keys() and info['styles_update_flag']:
        keys_list = list(sdxl_styles.styles.keys())
        style_sorter.try_load_sorted_styles(
                    style_names=[sdxl_styles.fooocus_expansion] + keys_list,
                    default_selected=styles)
        results += [gr.update(value=copy.deepcopy(styles), choices=copy.deepcopy(style_sorter.all_styles))]
    else:
        results += [gr.update(value=copy.deepcopy(styles))]
    results += [gr.update(value=info['Performance']),  gr.update(value=config.add_ratio(aspect_ratios)), gr.update(value=float(info['Sharpness'])), \
            gr.update(value=float(info['Guidance Scale'])), gr.update(value=info['Base Model']), gr.update(value=info['Refiner Model']), \
            gr.update(value=float(info['Refiner Switch'])), gr.update(value=info['Sampler']), gr.update(value=info['Scheduler']), \
            gr.update(value=adaptive_cfg), gr.update(value=overwrite_step), gr.update(value=overwrite_switch), gr.update(value=inpaint_engine)]
    for i, (n, v) in enumerate(info['loras']):
        results += [gr.update(value=n),gr.update(value=v)]
    results += [gr.update(value=adm_scaler_positive), gr.update(value=adm_scaler_negative), gr.update(value=adm_scaler_end), gr.update(value=int(info['Seed']))]
    return results
                                                                                                                                            

system_message = get_system_message()
