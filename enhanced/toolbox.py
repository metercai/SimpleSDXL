import os
import json
import copy
import re
import math
import gradio as gr
import modules.config as config
import modules.sdxl_styles as sdxl_styles
import modules.advanced_parameters as ads
import enhanced.topbar as topbar
import enhanced.gallery as gallery
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from enhanced.models_info import models_info, models_info_muid, refresh_models_info_from_path, sync_model_info
from modules.model_loader import load_file_from_url, load_file_from_muid


css = '''
.toolbox {
    height: auto;
    position: absolute;
    top: 210px;
    left: 86%;
    width: 100px !important;
    z-index: 20;
    text-align: center;
}

.infobox {
    height: auto;
    position: absolute;
    top: -15rem;
    left: 50%;
    transform: translateX(-50%);
    width: 28rem !important;
    z-index: 20;
    text-align: left;
    opacity: 0.85;
    border-radius: 8px;
    padding: 6px;
    line-height: 120%;
    border: groove;
}

.infobox_mobi {
    height: auto;
    position: absolute;
    top: -16rem;
    left: 50%;
    transform: translateX(-50%);
    width: 22rem !important;
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

.note_text {
    padding: 2px;
    text-align: center;
}
.preset_input textarea {
    width: 120px;
}
'''


# app context
toolbox_note_preset_title='Save a new preset for the current params and configuration.'
toolbox_note_regenerate_title='Extract parameters to backfill for regeneration. Please note that some parameters will be modified!'
toolbox_note_embed_title='Embed parameters into images for easy identification of image sources and communication and learning.'
toolbox_note_missing_muid='The model in the params and configuration is missing MUID. And the system will spend some time calculating the hash of model files and synchronizing information to obtain the muid for usability and transferability.'

def make_infobox_markdown(info):
    bgcolor = '#ddd'
    if config.theme == "dark":
        bgcolor = '#444'
    html = f'<div style="background: {bgcolor}">'
    if info:
        for key in info:
            if key == 'Filename' or key == 'Advanced_parameters':
                continue
            html += f'<b>{key}:</b> {info[key]}<br/>'
    else:
        html += '<p>info</p>'
    html += '</div>'
    return html


def toggle_toolbox(state, state_params):
    state_params.update({"infobox_state": 0})
    state_params.update({"note_box_state": ['',0,0]})
    return [gr.update(visible=state)] + [gr.update(visible=False)] * 5 + [state_params]


def toggle_prompt_info(state_params):
    infobox_state = state_params["infobox_state"]
    infobox_state = not infobox_state
    state_params.update({"infobox_state": infobox_state})
    #print(f'[ToolBox] Toggle_image_info: {infobox_state}')
    [choice, selected] = state_params["prompt_info"]
    prompt_info = gallery.get_images_prompt(choice, selected, state_params["__max_per_page"])
    return gr.update(value=make_infobox_markdown(prompt_info), visible=infobox_state), state_params


def check_preset_models(checklist, state_params):
    note_box_state = state_params["note_box_state"]
    note_box_state[2] = 0
    for i in range(len(checklist)):
        if checklist[i] and checklist[i] != 'None':
            k1 = "checkpoints/"+checklist[i]
            k2 = "loras/"+checklist[i]
            if (i<2 and (k1 not in models_info.keys() or not models_info[k1]['muid'])) or (i>=2 and (k2 not in models_info.keys() or not models_info[k2]['muid'])):
                note_box_state[2] = 1
                break
    state_params.update({"note_box_state": note_box_state})
    return state_params


def toggle_note_box(item, state_params):
    note_box_state = state_params["note_box_state"]
    if note_box_state[0] is None:
        note_box_state[0] = item
    if item == note_box_state[0]:
        note_box_state[1] = not note_box_state[1]
    elif not note_box_state[1]:
        note_box_state[1] = not note_box_state[1]
        note_box_state[0] = item
    else:
        state_params.update({"note_box_state": note_box_state})
        return [gr.update(visible=True)] + [gr.update()] * (3 if item == 'preset' else 2) + [state_params]
    state_params.update({"note_box_state": note_box_state})
    flag = note_box_state[1]
    title_extra = ""
    if note_box_state[2]:
        title_extra = '\n' + toolbox_note_missing_muid
    if item == 'delete':
        [choice, selected] = state_params["prompt_info"]
        info = gallery.get_images_prompt(choice, selected, state_params["__max_per_page"])
        return gr.update(value=f'DELETE the image from output directory and logs!', visible=True), gr.update(visible=flag), gr.update(visible=flag), state_params
    if item == 'regen':
        return gr.update(value=toolbox_note_regenerate_title, visible=True), gr.update(visible=flag), gr.update(visible=flag), state_params
    if item == 'preset':
        return gr.update(value=toolbox_note_preset_title + title_extra, visible=True), gr.update(visible=flag), gr.update(visible=flag), gr.update(visible=flag), state_params
    if item == 'embed':
        return gr.update(value=toolbox_note_embed_title + title_extra, visible=True), gr.update(visible=flag), gr.update(visible=flag), state_params

def toggle_note_box_delete(state_params):
    return toggle_note_box('delete', state_params)


def toggle_note_box_regen(prompt, negative_prompt, base_model, refiner_model, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5, state_params):
    checklist = [base_model, refiner_model, lora_model1, lora_model2, lora_model3, lora_model4, lora_model5]
    state_params = check_preset_models(checklist, state_params)
    return toggle_note_box('regen', state_params)


def toggle_note_box_preset(prompt, negative_prompt, base_model, refiner_model, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5, state_params):
    checklist = [base_model, refiner_model, lora_model1, lora_model2, lora_model3, lora_model4, lora_model5]
    state_params = check_preset_models(checklist, state_params)
    return toggle_note_box('preset', state_params)


def toggle_note_box_embed(prompt, negative_prompt, base_model, refiner_model, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5, state_params):
    checklist = [base_model, refiner_model, lora_model1, lora_model2, lora_model3, lora_model4, lora_model5]
    state_params = check_preset_models(checklist, state_params)
    return toggle_note_box('embed', state_params)



filename_regex = re.compile(r'\<div id=\"(.*?)_png\"')

def delete_image(state_params):
    [choice, selected] = state_params["prompt_info"]
    info = gallery.get_images_prompt(choice, selected, state_params["__max_per_page"])
    file_name = info["Filename"]
    output_index = choice.split('/')
    dir_path = os.path.join(config.path_outputs, '20' + output_index[0])
    
    log_path = os.path.join(dir_path, 'log.html')
    if os.path.exists(log_path):
        file_text = ''
        d_line_flag = False
        with open(log_path, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            while line:
                match = filename_regex.search(line)
                if match:
                    if match.group(1)==file_name[:-4]:
                        d_line_flag = True
                        line = log_file.readline()
                        continue
                    if d_line_flag:
                        d_line_flag = False
                if d_line_flag:
                    line = log_file.readline()
                    continue
                file_text += line
                line = log_file.readline()
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(file_text)
        print(f'[ToolBox] Delete item from log.html: {file_name}')

    log_name = os.path.join(dir_path, "log_ads.json")
    log_ext = {}
    if os.path.exists(log_name):
        log_ext = {}
        with open(log_name, "r", encoding="utf-8") as log_file:
            log_ext.update(json.load(log_file))
        log_ext.pop(file_name)
        with open(log_name, 'w', encoding='utf-8') as log_file:
            json.dump(log_ext, log_file)

    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    print(f'[ToolBox] Delete image file: {file_path}')

    image_list_nums = len(gallery.refresh_images_catalog(output_index[0], True))
    if image_list_nums<=0:
        os.remove(log_path)
        os.rmdir(dir_path)
        index = state_params["__output_list"].index(choice)
        state_params.update({"__output_list": gallery.refresh_output_list(state_params["__max_per_page"])})
        if index>= len(state_params["__output_list"]):
            index = len(state_params["__output_list"]) -1
            if index<0:
                index = 0
        choice = state_params["__output_list"][index]
    elif image_list_nums < state_params["__max_per_page"]:
        if selected > image_list_nums-1:
            selected = image_list_nums-1
    else:
        if image_list_nums % state_params["__max_per_page"] == 0:
            page = int(output_index[1])
            if page > image_list_nums//state_params["__max_per_page"]:
                page = image_list_nums//state_params["__max_per_page"]
            if page == 1:
                choice = output_index[0]
            else:
                choice = output_index[0] + '/' + str(page)
            state_params.update({"__output_list": gallery.refresh_output_list(state_params["__max_per_page"])})

    state_params.update({"prompt_info":[choice, selected]})
    images_gallery = gallery.get_images_from_gallery_index(choice, state_params["__max_per_page"])
    state_params.update({"note_box_state": ['',0,0]})
    return gr.update(value=images_gallery), gr.update(choices=state_params["__output_list"], value=choice), gr.update(visible=False), gr.update(visible=False), state_params


def reset_image_params(state_params):
    [choice, selected] = state_params["prompt_info"]
    metainfo = gallery.get_images_prompt(choice, selected, state_params["__max_per_page"])
    metadata = copy.deepcopy(metainfo)
    metadata['Refiner Model'] = None if metainfo['Refiner Model']=='' else metainfo['Refiner Model']
    loras = [['None', 1.0], ['None', 1.0], ['None', 1.0], ['None', 1.0], ['None', 1.0]]
    for key in metainfo:
        i=0
        if key.startswith('LoRA ['):
            loras.insert(i, [key[6:-8], float(metainfo[key])])
    loras = loras[:5]
    metadata.update({"loras": loras})
    metadata.update({"task_from": f'regeneration:{metadata["Filename"]}'})
    results = topbar.reset_params(metadata)
    state_params.update({"note_box_state": ['',0,0]})
    print(f'[ToolBox] Reset_params: update {len(metainfo.keys())} params from current image log file.')
    return results + [gr.update(visible=False)] * 2 + [state_params]


def save_preset(name, state_params, prompt, negative_prompt, style_selections, performance_selection, aspect_ratios_selection, sharpness, guidance_scale, base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, adaptive_cfg, overwrite_step, overwrite_switch, inpaint_engine, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, seed_random, image_seed):

    if name is not None and name != '':
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
        if ads.default["adm_scaler_positive"] != adm_scaler_positive:
            preset["default_adm_scaler_positive"] = adm_scaler_positive
        if ads.default["adm_scaler_negative"] != adm_scaler_negative:
            preset["default_adm_scaler_negative"] = adm_scaler_negative
        if ads.default["adm_scaler_end"] != adm_scaler_end:
            preset["default_adm_scaler_end"] = adm_scaler_end
        if ads.default["adaptive_cfg"] != adaptive_cfg:
            preset["default_cfg_tsnr"] = adaptive_cfg
        if ads.default["overwrite_step"] != overwrite_step:
            preset["default_overwrite_step"] = overwrite_step
        if ads.default["overwrite_switch"] != overwrite_switch:
            preset["default_overwrite_switch"] = overwrite_switch
        if ads.default["inpaint_engine"] != inpaint_engine:
            preset["default_inpaint_engine"] = inpaint_engine
        if not seed_random:
            preset["default_image_seed"] = image_seed

        def get_muid_link(k):
            muid = models_info[k]['muid']
            return '' if muid is None else f'MUID:{muid}'

        preset["checkpoint_downloads"] = {base_model: get_muid_link("checkpoints/"+base_model)}
        if refiner_model and refiner_model != 'None':
            preset["checkpoint_downloads"].update({refiner_model: get_muid_link("checkpoints/"+refiner_model)})

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
                preset["embeddings_downloads"].update({k[11:]: get_muid_link(k)})

        preset["lora_downloads"] = {}
        if lora_model1 and lora_model1 != 'None':
            preset["lora_downloads"].update({lora_model1: get_muid_link("loras/"+lora_model1)})
        if lora_model2 and lora_model2 != 'None':
            preset["lora_downloads"].update({lora_model2: get_muid_link("loras/"+lora_model2)})
        if lora_model3 and lora_model3 != 'None':
            preset["lora_downloads"].update({lora_model3: get_muid_link("loras/"+lora_model3)})
        if lora_model4 and lora_model4 != 'None':
            preset["lora_downloads"].update({lora_model4: get_muid_link("loras/"+lora_model4)})
        if lora_model5 and lora_model5 != 'None':
            preset["lora_downloads"].update({lora_model5: get_muid_link("loras/"+lora_model5)})
        
        m_dict = {}
        for key in style_selections:
            if key!='Fooocus V2':
                m_dict.update({key: sdxl_styles.styles[key]})
        if len(m_dict.keys())>0:
            preset["styles_definition"] = m_dict

        #print(f'preset:{preset}')
        save_path = 'presets/' + name + '.json'
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(preset, json_file, indent=4)

        state_params.update({"__preset": name})
        print(f'[ToolBox] Saved the current params and reset to {save_path}.')
    state_params.update({"note_box_state": ['',0,0]})
    results = [gr.update(visible=False)] * 3 + [state_params]
    results += topbar.refresh_nav_bars(state_params)
    return results


def embed_params(state_params):
    refresh_models_info_from_path()
    sync_model_info([])
    [choice, selected] = state_params["prompt_info"]
    info = gallery.get_images_prompt(choice, selected, state_params["__max_per_page"])
    #print(f'info:{info}')
    filename = info['Filename']
    file_path = os.path.join(os.path.join(config.path_outputs, '20' + choice.split('/')[0]), filename)
    img = Image.open(file_path)
    metadata = PngInfo()
    for x in info.keys():
        if x != 'Filename':
            metadata.add_text(x, json.dumps(info[x]), True)

    # the models(checkpoint, lora, embeddings) and styles referenced by the image
    resource_id = lambda x:f'HASH:{models_info[x]["hash"]}' if not models_info[x]['muid'] else f'MUID:{models_info[x]["muid"]}'
    m_dict = {info["Base Model"]: resource_id("checkpoints/" + info["Base Model"])}
    if info['Refiner Model'] and info['Refiner Model'] != 'None':
        m_dict.update({info["Refiner Model"]: resource_id("checkpoints/" + info["Refiner Model"])})
    metadata.add_text('checkpoint_downloads', json.dumps(m_dict), True) 
    
    m_dict = {}
    for key in info:
        if key.startswith('LoRA ['):
            m_dict.update({key[6:-8]: resource_id("loras/" + key[6:-8])})
    if len(m_dict.keys())>0:
        metadata.add_text('lora_downloads', json.dumps(m_dict), True)

    embeddings = topbar.embeddings_model_split(info["Prompt"], info["Negative Prompt"])
    m_dict = {}
    for key in embeddings:
        m_dict.update({key[11:]: resource_id(key)})
    if len(m_dict.keys())>0:
        metadata.add_text('embeddings_downloads', json.dumps(m_dict), True)

    styles_name = [f[1:-1] for f in info['Styles'][1:-1].split(', ')]
    for key in styles_name:
        if key!='Fooocus V2':
            m_dict.update({key: sdxl_styles.styles[key]})
    if len(m_dict.keys())>0:
        metadata.add_text('styles_definition', json.dumps(m_dict), True)

    embed_dirs = os.path.join(config.path_outputs, 'embed')
    if not os.path.exists(embed_dirs):
        os.mkdir(embed_dirs)
    embed_file = os.path.join(embed_dirs, filename)
    img.save(embed_file, pnginfo=metadata)
    print(f'[ToolBox] Embed_params: embed {len(info.keys())} params to image and save to {embed_file}.')
    return [gr.update(visible=False)] * 2 + [state_params]

def extract_reset_image_params(img_path):
    img = Image.open(img_path)
    metadata = {}
    if hasattr(img,'text'):
        for k in img.text:
            metadata.update({k: json.loads(img.text[k])})
    else:
        print(f'[ToolBox] Reset_params_from_image: it\'s not the embedded parameter image.')
        return [gr.update()] * 31
    print(f'[ToolBox] Extraction successful and ready to reset: {metadata}') 
    refresh_models_info_from_path()
    sync_model_info([])
    metadata.update({"task_from": f'embed_image:{img_path}'})
    results = topbar.reset_params(topbar.check_prepare_for_reset(metadata))   
    print(f'[ToolBox] Reset_params_from_image: update {len(metadata.keys())} params from input image.')
    return results

extract_reset_image_params_js = '''
function() {
refresh_style_localization()
}
'''

