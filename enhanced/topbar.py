import os
import json
import gradio as gr
import modules.util as util
import modules.config as config
import modules.flags
import modules.sdxl_styles
import numbers
import copy
import enhanced.gallery as gallery_util

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

get_preset_params_js = '''
function(preset_params) {
    var preset=preset_params["__preset"];
    var theme=preset_params["__theme"];
    var message=preset_params["__message"];
    var nav_id_list=preset_params["__nav_id_list"];
    var nav_preset_html = preset_params["__nav_preset_html"];
    update_topbar("top_preset",nav_preset_html)
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    if (url_params["__preset"]!=null) {
        preset=url_params["__preset"];
    }
    if (url_params["__theme"]!=null) {
        theme=url_params["__theme"];
    }
    if (message!=null && message.length>10) {
        showSysMsg(message);
    }
    mark_position_for_topbar(nav_id_list,preset,theme);
    preset_params["__preset"]=preset;
    preset_params["__theme"]=theme;
    return preset_params;
}
'''

preset_name = 'default'
preset_url = ''
nav_id_list = ''
nav_preset_html = ''

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
    id_array += 'theme_light,theme_dark,update_f,update_s'
    nav_id_list = id_array
    nav_preset_html = item_list
    return nav_html.replace('*item_list*', item_list)


config_ext = {}

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
            if first_line_f is None:
                first_line_f = line.strip()
            while line:
                if line == '\n':
                    line = log_file.readline()
                    continue
                if line.strip() == config_ext['fooocus_line']:
                    break
                update_msg_f += line
                line = log_file.readline()
    update_msg_s = ''
    first_line_s = None
    if os.path.exists(simplesdxl_log):
        with open(simplesdxl_log, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            if first_line_s is None:
                first_line_s = line.strip()
            while line:
                if line == '\n':
                    line = log_file.readline()
                    continue
                if line.strip() == config_ext['simplesdxl_line']:
                    break
                update_msg_s += line
                line = log_file.readline()
    update_msg_f = update_msg_f.replace("\n","  ")
    update_msg_s = update_msg_s.replace("\n","  ")
    body = ''
    import args_manager
    f_log_path = os.path.abspath("./update_log.md")
    s_log_path = os.path.abspath("./simplesdxl_log.md")
    if len(update_msg_f)>0:
        body += f'<b id="update_f">[Fooocus最新更新]</b>: {update_msg_f}<a href="{args_manager.args.webroot}/file={f_log_path}">更多>></a>   '
    if len(update_msg_s)>0:
        body += f'<b id="update_s">[SimpleSDXL最新更新]</b>: {update_msg_s}<a href="{args_manager.args.webroot}/file={s_log_path}">更多>></a>'
    import mistune
    body = mistune.html(body)
    if first_line_f and first_line_s and (first_line_f != config_ext['fooocus_line'] or first_line_s != config_ext['simplesdxl_line']):
        config_ext['fooocus_line']=first_line_f
        config_ext['simplesdxl_line']=first_line_s
        with open(enhanced_config, "w", encoding="utf-8") as config_file:
            json.dump(config_ext, config_file)
    return body if body else ''



def preset_instruction():
    global preset_name, preset_url

    head = "<div style='max-width:100%; height:120px; overflow:auto'>"
    foot = "</div>"
    p_name = config.preset
    if p_name == 'default':
        p_name = '默认'
    body = f'"{p_name}"包说明:<span style="position: absolute;right: 0;"><a href=>\U0001F4DD 什么是预置包</a></span>'
    preset_url_str = f'{preset_url}&__theme={config.theme}' if preset_url.count('?') else f'{preset_url}?__theme={config.theme}'
    body += f'<iframe src="{preset_url_str}" frameborder="0" scrolling="no" height="100" width="100%"></iframe>'
    return head + body + foot


def reset_context(preset_params):
    global preset_name, preset_url

    preset = preset_params.get("__preset", config.preset)
    theme = preset_params.get("__theme", config.theme)
    print(f'[Topbar] Reset_context: preset={config.preset}-->{preset}, theme={config.theme}-->{theme}')
    config.theme = theme

    results = []
    config.preset = preset
    config_org = {}
    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    config_org = json.load(json_file)
                    config.config_dict.update(config_org)
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    reset_default_config()
    preset_name = preset
    if 'reference' in config_org.keys():
        preset_url = config.config_dict["reference"]
    else:
        preset_url = ''
        if 'reference' in config.config_dict.keys():
            config.config_dict.pop("reference")
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
    return results

def reset_context_UI():
    results = [gr.update(), gr.update(choices=gallery_util.output_list, value=None if len(gallery_util.output_list)==0 else gallery_util.output_list[0])]
    results += [gr.update(visible=True if preset_url else False, value=preset_instruction())]
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


