import os
import json
import args_manager
import gradio as gr
import modules.util as util
import modules.config as config
import copy
import enhanced.gallery as gallery_util

nav_css = '''
.top_nav{
    height: 18px;
    position: relative;
}
.top_nav_left {
    position:absolute;
    left: 0;
    list-style: none;
    margin: auto;
}
.top_nav_right {
    position:absolute;
    right:0;
    list-style: none;
    margin: auto;
}

.top_nav_preset {
    color: rgb(224, 9, 9);
    display: inline-block;
    text-align: center;
    width: 60px;
    border-right: 1px solid rgb(161, 158, 158);
}
.top_nav_preset:hover {
    background-color: rgb(161, 158, 158);
}
.top_nav_preset:last-child{
     border-right: 0;
}

.top_nav_theme {
    color: rgb(224, 9, 9);
    display: inline-block;
    text-align: center;
    width: 40px;
}
.top_nav_theme:hover {
    background-color: rgb(161, 158, 158);
}
'''


nav_html = '''
<div class="top_nav">
    <ul class="top_nav_left" id="top_preset">
    Preset:
    *itemlist*
    </ul>
    <ul class="top_nav_right" id="top_theme">
    Theme:
    <li class="top_nav_theme" id="theme_light" onclick="refresh_theme('light')">light</li>
    <li class="top_nav_theme" id="theme_dark" onclick="refresh_theme('dark')">dark</li>
    </ul>
</div>
'''

get_preset_params_js = '''
function(preset_params) {
    var preset=preset_params["__preset"];
    var theme=preset_params["__theme"];
    const params = new URLSearchParams(window.location.search);
    preset_params = Object.fromEntries(params);
    if (preset_params["__preset"]!=null) {
        preset=preset_params["__preset"];
    }
    if (preset_params["__theme"]!=null) {
        theme=preset_params["__theme"];
    }
    mark_position_for_topbar(preset,theme);
    return preset_params;
}
'''

def make_html():
    path_preset = os.path.abspath(f'./presets/')
    presets = util.get_files_from_folder(path_preset, ['.json'], None)
    print(f'presets={presets}')
    itemlist = ''
    for i in range(len(presets)):
        itemlist += '<li class="top_nav_preset" id="preset_' + presets[i][:-5] + '" onclick="refresh_preset(\'' + presets[i][:-5] + '\')">' + presets[i][:-5] + '</li>'
    return nav_html.replace('*itemlist*', itemlist)


def reset_context(preset_params):
    preset = preset_params.get("__preset", config.preset)
    config.preset = preset
    print(f'preset={preset}')

    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    config.config_dict.update(json.load(json_file))
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    results = []
    results += [gr.update(value=config.config_dict.get("default_model", config.default_base_model_name)), \
                gr.update(value=config.config_dict.get("default_refiner",config.default_refiner_model_name)), \
                gr.update(value=config.config_dict.get("default_refiner_switch", config.default_refiner_switch)), \
                gr.update(value=config.config_dict.get("default_cfg_scale", config.default_cfg_scale)), \
                gr.update(value=config.config_dict.get("default_sample_sharpness", config.default_sample_sharpness)), \
                gr.update(value=config.config_dict.get("default_sampler", config.default_sampler)), \
                gr.update(value=config.config_dict.get("default_scheduler", config.default_scheduler)), \
                gr.update(value=config.config_dict.get("default_performance", config.default_performance)), \
                gr.update(value=config.config_dict.get("default_prompt", config.default_prompt)), \
                gr.update(value=config.config_dict.get("default_prompt_negative", config.default_prompt_negative)), \
                gr.update(value=copy.deepcopy(config.config_dict.get("default_styles", config.default_styles))), \
                gr.update(value=config.add_ratio(config.config_dict.get("default_aspect_ratio", config.default_aspect_ratio)))]
    for i, (n, v) in enumerate(config.config_dict.get("default_loras", config.default_loras)):
        results += [gr.update(value=n),gr.update(value=v)]
    gallery_util.refresh_output_list()
    results +=  [gr.update(value=gallery_util.get_images_from_gallery_index(None)), gr.update()]
    return results

def get_preset_params(preset_params, request: gr.Request):
    return preset_params



