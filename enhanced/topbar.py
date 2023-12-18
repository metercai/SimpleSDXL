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
import base64
import re
import hashlib
import requests
import time
import enhanced.gallery as gallery_util
import enhanced.token_did as token_did
from enhanced.models_hub_host import models_hub_host

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
    top: 230px;
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

.toolbox_note {
    height: auto;
    position: absolute;
    top: 160px;
    left: 50%;
    transform: translateX(-50%);
    width: 300px !important;
    z-index: 21;
    text-align: left;
    opacity: 1;
    border-radius: 8px;
    padding: 0px;
    border: groove;
}

.note_info {
    padding: 8px;
}

.preset_input textarea {
    width: 120px;
}

'''

toolbox_note_preset_title='Save a new preset for the current params and configuration.'
toolbox_note_regenerate_title='Extract parameters to backfill for regeneration. Please note that some parameters will be modified!'
toolbox_note_invalid_url='For usability and transferability of preset and embedinfo, please click "Sync model info" in the right model tab.'

nav_html = '''
<div class="top_nav">
    <ul class="top_nav_left" id="top_preset">
    <b>Preset:</b>
    *itemlist*
    </ul>
    <ul class="top_nav_right" id="top_theme">
    <b>Theme:</b>
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
    url_params = Object.fromEntries(params);
    if (url_params["__preset"]!=null) {
        preset=url_params["__preset"];
    }
    if (url_params["__theme"]!=null) {
        theme=url_params["__theme"];
    }
    mark_position_for_topbar(preset,theme,null);
    preset_params["__preset"]=preset;
    preset_params["__theme"]=theme;
    return preset_params;
}
'''

reset_preset_params_js = '''
function(preset_params) {
    var preset=preset_params["__preset"];
    var theme=preset_params["__theme"];
    var preset_pre=preset_params["__preset_pre"]
    mark_position_for_topbar(preset,theme,preset_pre);
    return 
}
'''


def reset_preset_params(preset_params):
    preset_params["__preset_pre"] = preset_params["__preset"]
    preset_params["__preset"]=config.preset_reset
    return preset_params


def reset_default_preset(preset_params):
    preset_params["__preset_pre"] = preset_params["__preset"]
    preset_params["__preset"] = 'default'
    return preset_params


def make_html():
    path_preset = os.path.abspath(f'./presets/')
    presets = util.get_files_from_folder(path_preset, ['.json'], None)
    del presets[presets.index('default.json')]
    file_times = [(f, os.path.getmtime(os.path.join(path_preset, f))) for f in presets]
    sorted_file_times = sorted(file_times, key=lambda x: x[1], reverse=True)
    sorted_files = [f[0] for f in sorted_file_times]
    sorted_files.insert(0, 'default.json')
    presets = sorted_files[:9]
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

    results +=  [gr.update(), gr.update(choices=gallery_util.output_list, value=None if len(gallery_util.output_list)==0 else gallery_util.output_list[0])]
    return results


def save_preset(prompt, negative_prompt, style_selections, performance_selection, aspect_ratios_selection, sharpness, guidance_scale, base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5, preset_name):

    if preset_name is not None and preset_name != '':
        preset = {}
        preset["default_model"] = base_model
        preset["default_refiner"] = refiner_model
        preset["default_refiner_switch"] = refiner_switch
        preset["default_loras"] = [[lora_model1, lora_weight1], [lora_model2, lora_weight2], [lora_model3, lora_weight3], [lora_model4, lora_weight4], [lora_model5, lora_weight5]]
        preset["default_cfg_scale"] = guidance_scale
        preset["default_sample_sharpness"] = sharpness
        preset["default_sampler"] = sampler_name
        preset["default_scheduler"] = scheduler_name
        preset["default_performance"] = performance_selection
        preset["default_prompt"] = prompt
        preset["default_prompt_negative"] = negative_prompt
        preset["default_styles"] = style_selections
        preset["default_aspect_ratio"] = aspect_ratios_selection.split(' ')[0].replace(u'\u00d7','*')

        preset["checkpoint_downloads"] = {base_model:models_info["checkpoints/"+base_model]["url"]}
        if refiner_model and refiner_model != 'None':
            preset["checkpoint_downloads"].update({refiner_model:models_info["checkpoints/"+refiner_model]["url"]})

        preset["embeddings_downloads"] = {}
        prompt_tags = re.findall(r'[\(](.*?)[)]', negative_prompt) + re.findall(r'[\(](.*?)[)]', prompt)
        embeddings = {}
        for e in prompt_tags:
            embed = e.split(':')
            if len(embed)>2 and embed[0] == 'embedding':
                embeddings.update({embed[1]:embed[2]})
        embeddings = embeddings.keys()
        for k in models_info.keys():
            if k.startswith('embeddings') and k[11:].split('.')[0] in embeddings:
                preset["embeddings_downloads"].update({k[11:]:models_info[k]["url"]})

        preset["lora_downloads"] = {}
        if lora_model1 and lora_model1 != 'None':
            preset["lora_downloads"].update({lora_model1:models_info["loras/"+lora_model1]["url"]})
        if lora_model2 and lora_model2 != 'None':
            preset["lora_downloads"].update({lora_model2:models_info["loras/"+lora_model2]["url"]})
        if lora_model3 and lora_model3 != 'None':
            preset["lora_downloads"].update({lora_model3:models_info["loras/"+lora_model3]["url"]})
        if lora_model4 and lora_model4 != 'None':
            preset["lora_downloads"].update({lora_model4:models_info["loras/"+lora_model4]["url"]})
        if lora_model5 and lora_model5 != 'None':
            preset["lora_downloads"].update({lora_model5:models_info["loras/"+lora_model5]["url"]})

        #print(f'preset:{preset}')
        save_path = 'presets/' + preset_name + '.json'
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(preset, json_file, indent=4)

        config.preset_reset = preset_name
        print(f'[Topbar] Saved the current params and config to {save_path}.')

    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=make_html())

models_info = {}
models_info_muid = {}
models_info_rsync = {}
models_info_file = ['models_info', 0]
models_info_path = os.path.abspath(f'./models/{models_info_file[0]}.json')

default_models_info = {
    "checkpoints/juggernautXL_version6Rundiffusion.safetensors": {
        "size": 7105348560,
        "hash": "H+bH7FTHhgQM2rx7TolyAGnZcJaSLiDQHxPndkQStH8=" },
    "checkpoints/sd_xl_base_1.0_0.9vae.safetensors": {
        "size": 6938078334,
        "hash": "5rueqFu/e/ZHinxtGLcSRvIuldQbzdgO1AqiEsM8/v8=" },
    "checkpoints/bluePencilXL_v050.safetensors": {
        "size": 6938040682,
        "hash": "jEYqgPS1KRsk/rYk1nJIvQn3H5B01wFNiteQGxIL3Mc=" },
    "checkpoints/sd_xl_refiner_1.0_0.9vae.safetensors": {
        "size": 6075981930,
        "hash": "jQzmwBYATL2s1Q+Tfa04HYw5ZijWIaf5cZFHBTJ4AWQ=" },
    "checkpoints/realisticStockPhoto_v10.safetensors": {
        "size": 6938053346,
        "hash": "LUTON43cySJ2nfTv/nTa4zcKiH6piXCT4fX73FC2egI=" },
    "checkpoints/DreamShaper_8_pruned.safetensors": {
        "size": 2132625894,
        "hash": "h521I8MNO5AXFD1WcFAV4VostWKHYsEdCG/tlTir1/0=" },
    "checkpoints/juggernautXL_v7Rundiffusion.safetensors": {
        "size": 7105352832,
        "hash": "ByRRjGtahV4XYEMvvwPIBblb0voHJy8PNq33rtlaAeM=" },
    "checkpoints/bluePencilXLV200.cXoH.safetensors": {
        "size": 6938040682,
        "hash": "1ZYnt0lbBYHe0rBIismPneuGtS1D6dkCkU8Pnc/9sWw=" },
    "checkpoints/dreamshaperXL_turboDpmppSDE.safetensors": {
        "size": 6939220250,
        "hash": "Z28NYMjoYBRtXooNgCWZyt0E58rfhcKD8Yn0HwHJ41k=" }
     }
    

def init_models_info():
    global models_info, models_info_muid, models_info_rsync, models_info_file, models_info_path

    if os.path.exists(models_info_path):
        file_mtime = time.localtime(os.path.getmtime(models_info_path)) 
        if (models_info is None or file_mtime != models_info_file[1]):
            try:
                with open(models_info_path, "r", encoding="utf-8") as json_file:
                    models_info.update(json.load(json_file))
                models_info_file[1] = file_mtime
            except Exception as e:
                print(f'[Topbar] Load model info file [{models_info_path}] failed!')
                print(e)
    
    model_filenames = config.get_model_filenames(config.path_checkpoints)
    lora_filenames = config.get_model_filenames(config.path_loras)
    embedding_filenames = config.get_model_filenames(config.path_embeddings)
    new_filenames = []
    for k in model_filenames:
        filename = 'checkpoints/'+k
        if filename not in models_info.keys():
            new_filenames.append(filename)
    for k in lora_filenames:
        filename = 'loras/'+k
        if filename not in models_info.keys():
            new_filenames.append(filename)
    for k in embedding_filenames:
        filename = 'embeddings/'+k
        if filename not in models_info.keys():
            new_filenames.append(filename)
    if len(new_filenames)>0:
        try:
            for filename in new_filenames:
                if filename.startswith('checkpoints'):
                    file_path = os.path.join(config.path_checkpoints, filename[12:])
                elif filename.startswith('loras'):
                    file_path = os.path.join(config.path_loras, filename[6:])
                elif filename.startswith('embeddings'):
                    file_path = os.path.join(config.path_embeddings, filename[11:])
                else:
                    file_path = os.path.abspath(f'./models/{filename}')
                size = os.path.getsize(file_path)
                if filename in default_models_info.keys() and size == default_models_info[filename]["size"]:
                    hash = default_models_info[filename]["hash"]
                else:
                    hash = ''
                models_info.update({filename:{'size': size, 'hash': hash, 'url': None, 'muid': None}})
            with open(models_info_path, "w", encoding="utf-8") as json_file:
                json.dump(models_info, json_file, indent=4)
            models_info_file[1] = time.localtime(os.path.getmtime(models_info_path))
        except Exception as e:
            print(f'[Topbar] Update model info file [{models_info_path}] failed!')
            print(e)
    
    return


def complement_model_hash():
    global models_info, models_info_file, models_info_path

    flag = False
    execution_start_time = time.perf_counter()
    print(f'[Topbar] Computing hash for {len(models_info.keys())} model files. It\'s some minutes.')
    try:
        for filename in models_info.keys():
            if filename.startswith('checkpoints'):
                file_path = os.path.join(config.path_checkpoints, filename[12:])
            elif filename.startswith('loras'):
                file_path = os.path.join(config.path_loras, filename[6:])
            elif filename.startswith('embeddings'):
                file_path = os.path.join(config.path_embeddings, filename[11:])
            else:
                file_path = os.path.abspath(f'./models/{filename}')
            if models_info[filename]['url'] is None:
                sha256obj = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    while True:
                        b = f.read(4*1024*1024)
                        if not b:
                            break
                        sha256obj.update(b)
                hash = base64.b64encode(sha256obj.digest()).decode('utf-8')
                models_info.update({filename:{'size': models_info[filename]['size'], 'hash': hash, 'url': models_info[filename]['url'], 'muid': models_info[filename]['muid']}})
                flag = True
                print(f'[Topbar] Computing the file hash: {hash} <- {filename}')
        if flag:
            with open(models_info_path, "w", encoding="utf-8") as json_file:
                json.dump(models_info, json_file, indent=4)
            models_info_file[1] = time.localtime(os.path.getmtime(models_info_path))
    except Exception as e:
        print(f'[Topbar] Load and compute file hash failed!')
        print(e)
    execution_time = time.perf_counter() - execution_start_time
    print(f'[Topbar] Total time for models file hash {execution_time:.2f} seconds')

    return gr.update(value=f'Total time for models file hash {execution_time:.2f} seconds')


def sync_model_info_click(*args):
    global models_info, models_info_rsync, models_info_file, models_info_path

    downurls = list(args)
    #print(f'downurls:{downurls} \nargs:{args}, len={len(downurls)}')
    keys = sorted(models_info.keys())
    keylist = []
    for i in range(len(keys)):
        if keys[i].startswith('checkpoints'):
            keylist.append(keys[i])
        if keys[i].startswith('loras'):
            keylist.append(keys[i])
    for i in range(len(keys)):
        if not keys[i].startswith('checkpoints') and not keys[i].startswith('loras'):
            keylist.append(keys[i])

    models_info_rsync = {}
    for i in range(len(keylist)):
        #print(f'downurls: i={i}, k={keylist[i]}, {downurls[i]}')
        durl = downurls[i]
        if durl and models_info[keylist[i]]['url'] != durl: 
            models_info_rsync.update({keylist[i]: {"hash": models_info[keylist[i]]['hash'], "url": durl, "muid": models_info[keylist[i]]['muid']}})
            models_info[keylist[i]]['url'] = durl

    flag = len(models_info_rsync.keys())
    file_mtime = time.localtime(os.path.getmtime(models_info_path))

    for k in models_info.keys():
        if models_info[k]['muid'] is not None and len(models_info[k]['muid'])>0:
            models_info_muid.update({models_info[k]['muid']: k})
        else:
            models_info_rsync.update({k: {"hash": models_info[k]['hash'], "url": models_info[k]['url']}})
    try:
        response = requests.post(f'{models_hub_host}/register_claim/', data = token_did.get_register_claim('SimpleSDXLHub'))
        rsync_muid_msg = { "files": token_did.encrypt_default(json.dumps(models_info_rsync)) }
        headers = { "DID": token_did.DID}
        response = requests.post(f'{models_hub_host}/rsync_muid/', data = json.dumps(rsync_muid_msg), headers = headers)
        results = json.loads(response.text)
        if (results["message"] == "it's ok!" and results["results"]):
            for k in results["results"].keys():
                models_info[k]['muid'] = results["results"][k]['muid']
                models_info[k]['url'] = results["results"][k]['url']
                print(f'[Topbar] Rsync info from model hub: MUID={models_info[k]["muid"]} <- {k}')
            with open(models_info_path, "w", encoding="utf-8") as json_file:
                json.dump(models_info, json_file, indent=4)
            models_info_file[1] = time.localtime(os.path.getmtime(models_info_path))
    except Exception as e:
            print(f'[Topbar] Connect the models hub site failed!')
            print(e)

    file_mtime2 = time.localtime(os.path.getmtime(models_info_path))
    if (flag and file_mtime == file_mtime2):
        with open(models_info_path, "w", encoding="utf-8") as json_file:
            json.dump(models_info, json_file, indent=4)
        models_info_file[1] = time.localtime(os.path.getmtime(models_info_path))

    results = []
    for k in keylist:
        muid = ' ' if models_info[k]['muid'] is None else models_info[k]['muid']
        durl = None if models_info[k]['url'] is None else models_info[k]['url']
        results += [gr.update(info=f'MUID={muid}', value=durl)]

    results += [gr.update(value=f'Rsync info of models in local file: {len(models_info_rsync)}')]
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


init_models_info()

