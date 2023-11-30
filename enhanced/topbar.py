import os
import json
import args_manager
import gradio as gr
import modules.util as util
import modules.config as config
import modules.flags
import modules.sdxl_styles
import numbers
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
    display: inline-block;
    text-align: center;
    width: 40px;
}
.top_nav_theme:hover {
    background-color: rgb(161, 158, 158);
}

.toolbox {
    height: auto;
    position: absolute;
    top: 250px;
    left: 85%;
    width: 100px !important;
    z-index: 20;
    text-align: center;
}

.infobox {
    height: auto;
    position: absolute;
    top: -240px;
    left: 50%;
    transform: translateX(-50%);
    width: 400px !important;
    z-index: 20;
    text-align: left;
    opacity: 0.85;
    border-radius: 8px;
    padding: 6px;
    line-height: 120%;
    border: groove;
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

nav_html_new = '''
<div class="top_nav">
    <ul class="top_nav_left" id="top_preset">
    Preset:
    *itemlist*
    </ul>
    <ul class="top_nav_right" id="top_theme">
    <li class="top_nav_theme">
    <select id="top_theme_1">
    <option value="light">明亮</option>
    <option value="dark">夜黑</option>
    </select>
    </li>
    <li class="top_nav_theme">
    <select id="top_theme_2">
    <option value="cn">中文</option>
    <option value="en">En</option>
    </select>
    </li>
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
    #print(f'presets={presets}')
    itemlist = ''
    for i in range(len(presets)):
        itemlist += '<li class="top_nav_preset" id="preset_' + presets[i][:-5] + '" onclick="refresh_preset(\'' + presets[i][:-5] + '\')">' + presets[i][:-5] + '</li>'
    return nav_html.replace('*itemlist*', itemlist)


def reset_context(preset_params):
    preset = preset_params.get("__preset", config.preset)
    theme = preset_params.get("__theme", config.theme)
    print(f'[Topbar] Reset_context: preset={config.preset}-->{preset}, theme={config.theme}-->{theme}')
    config.theme = theme

    results = []
    if preset == config.preset:
        results = [gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]
        for i, (n, v) in enumerate(config.default_loras):
             results += [gr.update(), gr.update()]
    else:
        config.preset = preset
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
        reset_default_config()
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

    results +=  [gr.update(), gr.update(choices=gallery_util.output_list, value=gallery_util.output_list[0])]
    return results

def get_preset_params(preset_params, request: gr.Request):
    gallery_util.refresh_output_list()
    return preset_params

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


