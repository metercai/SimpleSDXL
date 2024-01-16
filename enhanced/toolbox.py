import os
import json
import copy
import re
import math
import gradio as gr
import modules.config as config
import modules.advanced_parameters as ads
import enhanced.topbar as topbar
import enhanced.gallery as gallery
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from enhanced.models_info import models_info

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

.preset_input textarea {
    width: 120px;
}
'''


# app context
toolbox_note_preset_title='Save a new preset for the current params and configuration.'
toolbox_note_regenerate_title='Extract parameters to backfill for regeneration. Please note that some parameters will be modified!'
toolbox_note_embed_title='Embed parameters into images for easy identification of image sources and communication and learning.'
toolbox_note_invalid_url='The model in the params and configuration is missing MUID. For usability and transferability, please click "Sync model info" in the right model tab.'

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
    prompt_info = gallery.get_images_prompt(choice, selected)
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
        title_extra = '\n' + toolbox_note_invalid_url
    if item == 'delete':
        [choice, selected] = state_params["prompt_info"]
        info = gallery.get_images_prompt(choice, selected)
        return gr.update(value=f'DELETE the image from output directory and logs!', visible=True), gr.update(visible=flag), gr.update(visible=flag), state_params
    if item == 'regen':
        return gr.update(value=toolbox_note_regenerate_title + title_extra, visible=True), gr.update(visible=flag), gr.update(visible=flag), state_params
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
    info = gallery.get_images_prompt(choice, selected)
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

    log_name = os.path.join(dirname, "log_ads.json")
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

    image_list_nums = len(gallery.refresh_images_list(output_index[0], True))
    if image_list_nums<=0:
        os.remove(log_path)
        os.rmdir(dir_path)
        index = gallery.output_list.index(choice)
        gallery.refresh_output_list()
        if index>= len(gallery.output_list):
            index = len(gallery.output_list) -1
            if index<0:
                index = 0
        choice = gallery.output_list[index]
    elif image_list_nums<gallery.max_per_page:
        if selected>image_list_nums-1:
            selected = image_list_nums-1
    else:
        if image_list_nums % gallery.max_per_page == 0:
            page = int(output_index[1])
            if page>image_list_nums//gallery.max_per_page:
                page = image_list_nums//gallery.max_per_page
            if page == 1:
                choice = output_index[0]
            else:
                choice = output_index[0] + '/' + str(page)
            gallery.refresh_output_list()

    state_params.update({"prompt_info":[choice, selected]})
    images_gallery = gallery.get_images_from_gallery_index(choice)
    state_params.update({"note_box_state": ['',0,0]})
    return gr.update(value=images_gallery), gr.update(choices=gallery.output_list, value=choice), gr.update(visible=False), gr.update(visible=False), state_params


def reset_default_preset(state_params):
    state_params["__preset"]= 'default'
    state_params.update({"note_box_state": ['',0,0]})
    #print(f'reset_default_preset:{state_params}')
    return state_params, state_params


def reset_preset_params(state_params):
    state_params.update({"__nav_id_list": topbar.nav_id_list})
    state_params.update({"__nav_preset_html": topbar.nav_preset_html})
    #print(f'preset_params_out:{state_params}')
    return state_params, state_params


reset_preset_params_js = '''
function(system_params) {
    var preset=system_params["__preset"];
    var theme=system_params["__theme"];
    var nav_id_list=system_params["__nav_id_list"];
    var nav_preset_html = system_params["__nav_preset_html"];
    update_topbar("top_preset",nav_preset_html)
    mark_position_for_topbar(nav_id_list,preset,theme);
    return
}
'''


def reset_params(state_params):
    [choice, selected] = state_params["prompt_info"]
    info = gallery.get_images_prompt(choice, selected)
    results = _reset_params(info)
    print(f'[ToolBox] Reset_params: update {len(info.keys())} params from current image log file.')
    return results + [gr.update(visible=False)] * 2

def _reset_params(info):
    print(f'[ToolBox] Get params to reset:{info}')
    aspect_ratios = info['Resolution'][1:-1].replace(', ', '*')
    adm_scaler_positive, adm_scaler_negative, adm_scaler_end = [float(f) for f in info['ADM Guidance'][1:-1].split(', ')]
    refiner_model = None if info['Refiner Model']=='' else info['Refiner Model']
    
    if info['Advanced_parameters'] and 'adaptive_cfg' in info['Advanced_parameters'].keys():
        adaptive_cfg = info['Advanced_parameters']['adaptive_cfg']
    else:
        adaptive_cfg = ads.default['adaptive_cfg']
    if info['Advanced_parameters'] and 'overwrite_step' in info['Advanced_parameters'].keys():
        overwrite_step = info['Advanced_parameters']['overwrite_step']
    else:
        overwrite_step = ads.default['overwrite_step']
    if info['Advanced_parameters'] and 'overwrite_switch' in info['Advanced_parameters'].keys():
        overwrite_switch = info['Advanced_parameters']['overwrite_switch']
    else:
        overwrite_switch = ads.default['overwrite_switch']

    lora_results = []
    for k in range(0,5):
        lora_results += [gr.update(value='None'), gr.update(value=1.0)]
    for key in info:
        i=0
        if key.startswith('LoRA ['):
            lora_model = key[6:-9]
            lora_weight = float(info[key])
            lora_results.insert(i, gr.update(value=lora_model))
            lora_results.insert(i+1, gr.update(value=lora_weight))
            i += 2
    lora_results = lora_results[:10]
    styles = [f[1:-1] for f in info['Styles'][1:-1].split(', ')]
    results = []
    results += [gr.update(value=info['Prompt']), gr.update(value=info['Negative Prompt']), gr.update(value=copy.deepcopy(styles)), \
            gr.update(value=info['Performance']),  gr.update(value=config.add_ratio(aspect_ratios)), gr.update(value=float(info['Sharpness'])), \
            gr.update(value=float(info['Guidance Scale'])), gr.update(value=info['Base Model']), gr.update(value=refiner_model), \
            gr.update(value=float(info['Refiner Switch'])), gr.update(value=info['Sampler']), gr.update(value=info['Scheduler']), \
            gr.update(value=adaptive_cfg), gr.update(value=overwrite_step), gr.update(value=overwrite_switch)]
    results += lora_results
    results += [gr.update(value=adm_scaler_positive), gr.update(value=adm_scaler_negative), gr.update(value=adm_scaler_end), gr.update(value=int(info['Seed']))]
    return results


def save_preset(name, state_params, prompt, negative_prompt, style_selections, performance_selection, aspect_ratios_selection, sharpness, guidance_scale, base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, adaptive_cfg, overwrite_step, overwrite_switch, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5):

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
        preset["default_cfg_tsnr"]=adaptive_cfg
        preset["default_overwrite_step"]=overwrite_step
        preset["default_overwrite_switch"]=overwrite_switch

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


        #print(f'preset:{preset}')
        save_path = 'presets/' + name + '.json'
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(preset, json_file, indent=4)

        state_params.update({"__preset": name})
        print(f'[ToolBox] Saved the current params and config to {save_path}.')
    state_params.update({"note_box_state": ['',0,0]})
    topbar.make_html()
    return [gr.update(visible=False)] * 4 + [state_params]


def embed_params(state_params):
    [choice, selected] = state_params["prompt_info"]
    info = gallery.get_images_prompt(choice, selected)
    #print(f'info:{info}')
    filename = info['Filename']
    file_path = os.path.join(os.path.join(config.path_outputs, '20' + choice.split('/')[0]), filename)
    img = Image.open(file_path)
    metadata = PngInfo()
    for x in info.keys():
        metadata.add_text(x, json.dumps(info[x]), True)

    embed_dirs = os.path.join(config.path_outputs, 'embed')
    if not os.path.exists(embed_dirs):
        os.mkdir(embed_dirs)
    embed_file = os.path.join(embed_dirs, filename)
    img.save(embed_file, pnginfo=metadata)
    print(f'[ToolBox] Embed_params: embed {len(info.keys())} params to image and save to {embed_file}.')
    return [gr.update(visible=False)] * 2 + [state_params]

def extract_parameters(img_path, state_params):
    img = Image.open(img_path)
    metadata = {}
    if hasattr(img,'text'):
        for k in img.text:
            metadata.update({k: json.loads(img.text[k])})
    state_params.update({'image_params': metadata})
    return state_params


def reset_params_from_image(state_params):
    info = state_params['image_params']
    results = _reset_params(info)   
    print(f'[ToolBox] Reset_params_from_image: update {len(info.keys())} params from input image.')
    return results
