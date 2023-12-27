import os
import json
import copy
import re
import gradio as gr
import modules.config as config
import enhanced.topbar as topbar
from enhanced.models_info import models_info

css = '''
.toolbox {
    height: auto;
    position: absolute;
    top: 230px;
    left: 86%;
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


def make_infobox_markdown(info):
    bgcolor = '#ddd'
    if config.theme == "dark":
        bgcolor = '#444'
    html = f'<div style="background: {bgcolor}">'
    if info:
        for key in info:
            if key == 'Filename':
                continue
            html += f'<b>{key}:</b> {info[key]}<br/>'
    else:
        html += '<p>info</p>'
    html += '</div>'
    return html


infobox_state = False


def toggle_prompt_info(prompt_info):
    global infobox_state
    infobox_state = not infobox_state
    print(f'[ToolBox] Toggle_image_info: {infobox_state}')
    return gr.update(value=make_infobox_markdown(prompt_info), visible=infobox_state)


def reset_params(info):
    aspect_ratios = info['Resolution'][1:-1].replace(', ', '*')
    adm_scaler_positive, adm_scaler_negative, adm_scaler_end = [float(f) for f in info['ADM Guidance'][1:-1].split(', ')]
    refiner_model = None if info['Refiner Model']=='' else info['Refiner Model']
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
            gr.update(value=float(info['Refiner Switch'])), gr.update(value=info['Sampler']), gr.update(value=info['Scheduler'])]
    results += lora_results
    results += [gr.update(value=adm_scaler_positive), gr.update(value=adm_scaler_negative), gr.update(value=adm_scaler_end), gr.update(value=int(info['Seed']))]
    print(f'[ToolBox] Reset_params: update {len(results)} params from current image.')
    return results + [gr.update(visible=False)] * 2



toolbox_note_preset_title='Save a new preset for the current params and configuration.'
toolbox_note_regenerate_title='Extract parameters to backfill for regeneration. Please note that some parameters will be modified!'
toolbox_note_invalid_url='The model in the params and configuration is missing MUID. For usability and transferability, please click "Sync model info" in the right model tab.'

note_box_state = [None,False,False]

def check_preset_models(checklist):
    note_box_state[2] = False
    for i in range(len(checklist)):
        if checklist[i] and checklist[i] != 'None':
            k1 = "checkpoints/"+checklist[i]
            k2 = "loras/"+checklist[i]
            if (i<2 and (k1 not in models_info.keys() or not models_info[k1]['muid'])) or (i>=2 and (k2 not in models_info.keys() or not models_info[k2]['muid'])):
                note_box_state[2] = True
                break
    return


def toggle_note_box_regen(prompt, negative_prompt, base_model, refiner_model, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5):
    checklist = [base_model, refiner_model, lora_model1, lora_model2, lora_model3, lora_model4, lora_model5]
    check_preset_models(checklist)
    return toggle_note_box('regen')

def toggle_note_box_preset(prompt, negative_prompt, base_model, refiner_model, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5):
    checklist = [base_model, refiner_model, lora_model1, lora_model2, lora_model3, lora_model4, lora_model5]
    check_preset_models(checklist)
    return toggle_note_box('preset')

def toggle_note_box(item):
    global note_box_state

    if note_box_state[0] is None:
        note_box_state[0] = item
    if item == note_box_state[0]:
        note_box_state[1] = not note_box_state[1]
    elif not note_box_state[1]:
        note_box_state[1] = not note_box_state[1]
        note_box_state[0] = item
    else:
        return [gr.update()] * 5 if item == 'preset' else 3
    flag = note_box_state[1]
    title_extra = ""
    if note_box_state[2]:
        title_extra = '\n' + toolbox_note_invalid_url
    if item == 'regen':
        return [gr.update(value=toolbox_note_regenerate_title + title_extra), gr.update(visible=flag), gr.update(visible=flag)]
    if item == 'preset':
        return [gr.update(value=toolbox_note_preset_title + title_extra), gr.update(visible=flag), gr.update(visible=flag), gr.update(visible=flag), gr.update(visible=flag)]


def reset_default_preset(preset_params):
    global note_box_state
    preset_params["__preset"] = 'default'
    note_box_state = [None,False,False]
    return preset_params


def save_preset(name, url, prompt, negative_prompt, style_selections, performance_selection, aspect_ratios_selection, sharpness, guidance_scale, base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, lora_model1, lora_weight1, lora_model2, lora_weight2, lora_model3, lora_weight3, lora_model4, lora_weight4, lora_model5, lora_weight5):
    global note_box_state

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

        def get_muid(k):
            muid = models_info[k]['muid']
            return '' if muid is None else f'MUID:{muid}'

        preset["checkpoint_downloads"] = {base_model:get_muid("checkpoints/"+base_model)}
        if refiner_model and refiner_model != 'None':
            preset["checkpoint_downloads"].update({refiner_model:get_muid("checkpoints/"+refiner_model)})

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
                preset["embeddings_downloads"].update({k[11:]:get_muid(k)})

        preset["lora_downloads"] = {}
        if lora_model1 and lora_model1 != 'None':
            preset["lora_downloads"].update({lora_model1:get_muid("loras/"+lora_model1)})
        if lora_model2 and lora_model2 != 'None':
            preset["lora_downloads"].update({lora_model2:get_muid("loras/"+lora_model2)})
        if lora_model3 and lora_model3 != 'None':
            preset["lora_downloads"].update({lora_model3:get_muid("loras/"+lora_model3)})
        if lora_model4 and lora_model4 != 'None':
            preset["lora_downloads"].update({lora_model4:get_muid("loras/"+lora_model4)})
        if lora_model5 and lora_model5 != 'None':
            preset["lora_downloads"].update({lora_model5:get_muid("loras/"+lora_model5)})

        if url:
            preset["reference"] = url

        #print(f'preset:{preset}')
        save_path = 'presets/' + name + '.json'
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(preset, json_file, indent=4)

        topbar.preset_name = name
        topbar.preset_url = url
        print(f'[ToolBox] Saved the current params and config to {save_path}.')
    note_box_state = [None,False,False]
    topbar.make_html()
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def reset_preset_params(preset_params):
    preset_params["__preset"]=topbar.preset_name
    preset_params["__nav_id_list"]=topbar.nav_id_list
    preset_params["__nav_preset_html"]=topbar.nav_preset_html
    return preset_params


reset_preset_params_js = '''
function(preset_params) {
    var preset=preset_params["__preset"];
    var theme=preset_params["__theme"];
    var nav_id_list=preset_params["__nav_id_list"];
    var nav_preset_html = preset_params["__nav_preset_html"];
    update_topbar("top_preset",nav_preset_html)
    mark_position_for_topbar(nav_id_list,preset,theme);
    return 
}
'''

