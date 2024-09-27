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
import modules.meta_parser as meta_parser
import enhanced.all_parameters as ads
import modules.sdxl_styles as sdxl_styles
import modules.style_sorter as style_sorter
import enhanced.gallery as gallery_util
import enhanced.superprompter as superprompter
import enhanced.comfy_task as comfy_task
import shared
from enhanced.simpleai import comfyd
from modules.model_loader import load_file_from_url, load_file_from_muid

css = '''
'''

# app context
nav_name_list = ''
system_message = ''
config_ext = {}
enhanced_config = os.path.abspath(f'./enhanced/config.json')
if os.path.exists(enhanced_config):
    with open(enhanced_config, "r", encoding="utf-8") as json_file:
        config_ext.update(json.load(json_file))
else:
    config_ext.update({'fooocus_line': '# 2.1.852', 'simplesdxl_line': '# 2023-12-20'})

def get_welcome_image(is_mobile=False):
    path_welcome = os.path.abspath(f'./enhanced/attached/')
    file_welcome = os.path.join(path_welcome, 'welcome.png')
    file_suffix = 'welcome_w' if not is_mobile else 'welcome_m'
    welcomes = [p for p in util.get_files_from_folder(path_welcome, ['.jpg', '.jpeg', 'png'], file_suffix, None) if not p.startswith('.')]
    if len(welcomes)>0:
        file_welcome = random.choice(welcomes)
    return file_welcome

def get_preset_name_list():
    path_preset = os.path.abspath(f'./presets/')
    presets = [p for p in util.get_files_from_folder(path_preset, ['.json'], None) if not p.startswith('.')]
    file_times = [(f, os.path.getmtime(os.path.join(path_preset, f))) for f in presets]
    sorted_file_times = sorted(file_times, key=lambda x: x[1], reverse=True)
    sorted_files = [f[0] for f in sorted_file_times]
    sorted_files.pop(sorted_files.index(f'{config.preset}.json'))
    sorted_files.insert(0, f'{config.preset}.json')
    presets = sorted_files[:shared.BUTTON_NUM]
    name_list = ''
    for i in range(len(presets)):
        name_list += f'{presets[i][:-5]},'
    name_list = name_list[:-1]
    return name_list

def is_models_file_absent(preset_name):
    preset_path = os.path.abspath(f'./presets/{preset_name}.json')
    if os.path.exists(preset_path):
        with open(preset_path, "r", encoding="utf-8") as json_file:
            config_preset = json.load(json_file)
        if config_preset["default_model"] and config_preset["default_model"] != 'None':
            if 'Flux' in preset_name and config_preset["default_model"]== 'auto':
                config_preset["default_model"] = comfy_task.get_default_base_Flux_name('+' in preset_name)
            model_key = f'checkpoints/{config_preset["default_model"]}'
            return not shared.modelsinfo.exists_model(catalog="checkpoints", model_path=config_preset["default_model"])
        if config_preset["default_refiner"] and config_preset["default_refiner"] != 'None':
           return not shared.modelsinfo.exists_model(catalog="checkpoints", model_path=config_preset["default_refiner"])
    return False


def get_system_message():
    global config_ext

    fooocus_log = os.path.abspath(f'./update_log.md')
    simplesdxl_log = os.path.abspath(f'./simplesdxl_log.md')
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
        body_f = f'<b id="update_f">[Fooocus更新信息]</b>: {update_msg_f}<a href="{args_manager.args.webroot}/file={f_log_path}">更多>></a>   '
    else:
        body_f = '<b id="update_f"> </b>'
    if len(update_msg_s)>0:
        body_s = f'<b id="update_s">[系统消息 - 已更新内容]</b>: {update_msg_s}<a href="{args_manager.args.webroot}/file={s_log_path}">更多>></a>'
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



def preset_instruction():
    head = "<div style='max-width:100%; max-height:86px; overflow:hidden'>"
    foot = "</div>"
    body = '预置包简介:<span style="position: absolute;right: 0;"><a href="https://gitee.com/metercai/SimpleSDXL/blob/SimpleSDXL/presets/readme.md">\U0001F4DD 什么是预置包</a></span>'
    body += f'<iframe id="instruction" src="{get_preset_inc_url()}" frameborder="0" scrolling="no" width="100%"></iframe>'
    
    return head + body + foot



get_system_params_js = '''
function(system_params) {
    const params = new URLSearchParams(window.location.search);
    const url_params = Object.fromEntries(params);
    if (url_params["__lang"]!=null) {
        lang=url_params["__lang"];
        system_params["__lang"]=lang;
    }
    if (url_params["__theme"]!=null) {
        theme=url_params["__theme"];
        system_params["__theme"]=theme;
    }
    setObserver();

    return system_params;
}
'''


refresh_topbar_status_js = '''
function(system_params) {
    const preset=system_params["__preset"];
    const theme=system_params["__theme"];
    const nav_name_list_str = system_params["__nav_name_list"];
    let nav_name_list = new Array();
    nav_name_list = nav_name_list_str.split(",")
    for (let i=0;i<nav_name_list.length;i++) {
        let item_id = "bar"+i;
        let item_name = nav_name_list[i];
        let nav_item = gradioApp().getElementById(item_id);
        if (nav_item!=null) {
            if (item_name != preset) {
                if (theme == "light") {
                    nav_item.style.color = 'var(--neutral-400)';
                    nav_item.style.background= 'var(--neutral-100)';
                } else {
                    nav_item.style.color = 'var(--neutral-400)';
                    nav_item.style.background= 'var(--neutral-700)';
                }
            } else {
                if (theme == 'light') {
                    nav_item.style.color = 'var(--neutral-800)';
                    nav_item.style.background= 'var(--secondary-200)';
                } else {
                    nav_item.style.color = 'white';
                    nav_item.style.background= 'var(--secondary-400)';
                }
            }
        }
    }
    const message=system_params["__message"];
    if (message!=null && message.length>60) {
        showSysMsg(message, theme);
    }
    let infobox=gradioApp().getElementById("infobox");
    if (infobox!=null) {
        let css = infobox.getAttribute("class")
        if (browser.device.is_mobile && css.indexOf("infobox_mobi")<0)
            infobox.setAttribute("class", css.replace("infobox", "infobox_mobi"));
    }
    webpath = system_params["__webpath"];
    const lang=system_params["__lang"];
    if (lang!=null) {
        set_language(lang);
    }
    let preset_url = system_params["__preset_url"];
    if (preset_url!=null) {
        set_iframe_src(theme,lang,preset_url);
    }
    return {}
}
'''


def init_nav_bars(state_params, request: gr.Request):
    #print(f'request.headers:{request.headers}')
    if "__lang" not in state_params.keys():
        if 'accept-language' in request.headers and 'zh-CN' in request.headers['accept-language']:
            args_manager.args.language = 'cn'
        else:
            print(f'no accept-language in request.headers:{request.headers}')
        state_params.update({"__lang": args_manager.args.language}) 
    if "__theme" not in state_params.keys():
        state_params.update({"__theme": args_manager.args.theme})
    if "__preset" not in state_params.keys():
        state_params.update({"__preset": config.preset})
    if "__session" not in state_params.keys() and "cookie" in request.headers.keys():
        cookies = dict([(s.split('=')[0], s.split('=')[1]) for s in request.headers["cookie"].split('; ')])
        #print(f'cookies in request.headers:{cookies}, {request.headers}')
        if "SESSION" in cookies.keys():
            state_params.update({"__session": cookies["SESSION"]})
    user_agent = request.headers["user-agent"]
    if "__is_mobile" not in state_params.keys():
        state_params.update({"__is_mobile": True if user_agent.find("Mobile")>0 and user_agent.find("AppleWebKit")>0 else False})
    if "__webpath" not in state_params.keys():
        state_params.update({"__webpath": f'{args_manager.args.webroot}/file={os.getcwd()}'})
    if "__max_per_page" not in state_params.keys():
        if state_params["__is_mobile"]:
            state_params.update({"__max_per_page": 9})
        else:
            state_params.update({"__max_per_page": 18})
    if "__max_catalog" not in state_params.keys():
        state_params.update({"__max_catalog": config.default_image_catalog_max_number })
    max_per_page = state_params["__max_per_page"]
    max_catalog = state_params["__max_catalog"]
    output_list, finished_nums, finished_pages = gallery_util.refresh_output_list(max_per_page, max_catalog)
    state_params.update({"__output_list": output_list})
    state_params.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})
    state_params.update({"infobox_state": 0})
    state_params.update({"note_box_state": ['',0,0]})
    state_params.update({"array_wildcards_mode": '['})
    state_params.update({"wildcard_in_wildcards": 'root'})
    state_params.update({"bar_button": config.preset})
    state_params.update({"init_process": 'finished'})
    results = refresh_nav_bars(state_params)
    results += [gr.update(value=f'enhanced/attached/{get_welcome_image(state_params["__is_mobile"])}')]
    results += [gr.update(value=modules.flags.language_radio(state_params["__lang"])), gr.update(value=state_params["__theme"])]
    results += [gr.update(choices=state_params["__output_list"], value=None), gr.update(visible=len(state_params["__output_list"])>0, open=False)]
    results += [gr.update(value=False if state_params["__is_mobile"] else config.default_inpaint_advanced_masking_checkbox)]
    preset = 'default'
    preset_url = get_preset_inc_url(preset)
    state_params.update({"__preset_url":preset_url})
    results += [gr.update(visible=True if 'blank.inc.html' not in preset_url else False)]
    
    return results

def get_preset_inc_url(preset_name='blank'):
    preset_name = f'{preset_name}.inc'
    preset_inc_path = os.path.abspath(f'./presets/html/{preset_name}.html')
    blank_inc_path = os.path.abspath(f'./presets/html/blank.inc.html')
    if os.path.exists(preset_inc_path):
        return f'{args_manager.args.webroot}/file={preset_inc_path}'
    else:
        return f'{args_manager.args.webroot}/file={blank_inc_path}'

def refresh_nav_bars(state_params):
    state_params.update({"__nav_name_list": get_preset_name_list()})
    preset_name_list = state_params["__nav_name_list"].split(',')
    for i in range(shared.BUTTON_NUM-len(preset_name_list)):
        preset_name_list.append('')
    results = []
    if state_params["__is_mobile"]:
        results += [gr.update(visible=False)]
    else:
        results += [gr.update(visible=True)]
    for i in range(len(preset_name_list)):
        name = preset_name_list[i]
        name += '\u2B07' if is_models_file_absent(name) else ''
        visible_flag = i<(7 if state_params["__is_mobile"] else shared.BUTTON_NUM)
        if name:
            results += [gr.update(value=name, visible=visible_flag)]
        else: 
            results += [gr.update(value='', interactive=False, visible=visible_flag)]
    return results


def process_before_generation(state_params, backend_params, backfill_prompt, translation_methods, comfyd_active_checkbox):
    if "__nav_name_list" not in state_params.keys():
        state_params.update({"__nav_name_list": get_preset_name_list()})
    superprompter.remove_superprompt()
    remove_tokenizer()
    backend_params.update({
        'backfill_prompt': backfill_prompt,
        'translation_methods': translation_methods,
        'comfyd_active_checkbox': comfyd_active_checkbox,
        'preset': state_params["__preset"],
        })
    # stop_button, skip_button, generate_button, gallery, state_is_generating, index_radio, image_toolbox, prompt_info_box
    results = [gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True, gr.update(visible=False, open=False), gr.update(visible=False), gr.update(visible=False)]
    # prompt, random_button, translator_button, super_prompter, background_theme, image_tools_checkbox, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button
    preset_nums = len(state_params["__nav_name_list"].split(','))
    results += [gr.update(interactive=False)] * (preset_nums + 6)
    results += [gr.update()] * (shared.BUTTON_NUM-preset_nums)
    results += [backend_params]
    state_params["gallery_state"]='preview'
    return results


def process_after_generation(state_params):
    #if "__max_per_page" not in state_params.keys():
    #    state_params.update({"__max_per_page": 18})
    max_per_page = state_params["__max_per_page"]
    max_catalog = state_params["__max_catalog"]
    output_list, finished_nums, finished_pages = gallery_util.refresh_output_list(max_per_page, max_catalog)
    state_params.update({"__output_list": output_list})
    state_params.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})
    # generate_button, stop_button, skip_button, state_is_generating
    results = [gr.update(visible=True, interactive=True)] + [gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False]
    # gallery_index, index_radio
    results += [gr.update(choices=state_params["__output_list"], value=None), gr.update(visible=len(state_params["__output_list"])>0, open=False)]
    # prompt, random_button, translator_button, super_prompter, background_theme, image_tools_checkbox, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button
    preset_nums = len(state_params["__nav_name_list"].split(','))
    results += [gr.update(interactive=True)] * (preset_nums + 6)
    results += [gr.update()] * (shared.BUTTON_NUM-preset_nums)
    
    if len(state_params["__output_list"]) > 0:
        output_index = state_params["__output_list"][0].split('/')[0]
        gallery_util.refresh_images_catalog(output_index, True)
        gallery_util.parse_html_log(output_index, True)
    
    return results


def sync_message(state_params):
    state_params.update({"__message":system_message})
    return state_params

preset_down_note_info = 'The preset package being loaded has model files that need to be downloaded, and it will take some time to wait...'
def check_absent_model(bar_button, state_params):
    #print(f'check_absent_model,state_params:{state_params}')
    state_params.update({'bar_button': bar_button})
    return state_params

def down_absent_model(state_params):
    state_params.update({'bar_button': state_params["bar_button"].replace('\u2B07', '')})
    return gr.update(visible=False), state_params


def reset_layout_params(prompt, negative_prompt, state_params, is_generating, inpaint_mode, comfyd_active_checkbox):
    global system_message, preset_down_note_info

    state_params.update({"__message": system_message})
    system_message = 'system message was displayed!'
    if '__preset' not in state_params.keys() or 'bar_button' not in state_params.keys() or state_params["__preset"]==state_params['bar_button']:
        return [gr.update()] * (35 + shared.BUTTON_NUM) + [state_params] + [gr.update()] * 55
    if '\u2B07' in state_params["bar_button"]:
        gr.Info(preset_down_note_info)
    preset = state_params["bar_button"] if '\u2B07' not in state_params["bar_button"] else state_params["bar_button"].replace('\u2B07', '')
    print(f'[Topbar] Reset_context: preset={state_params["__preset"]}-->{preset}, theme={state_params["__theme"]}, lang={state_params["__lang"]}')
    state_params.update({"__preset": preset})
    #state_params.update({"__prompt": prompt})
    #state_params.update({"__negative_prompt": negative_prompt})

    config_preset = config.try_get_preset_content(preset)
    preset_prepared = meta_parser.parse_meta_from_preset(config_preset)
    #print(f'preset_prepared:{preset_prepared}')
    
    engine = preset_prepared.get('engine', {}).get('backend_engine', 'Fooocus')
    state_params.update({"engine": engine})

    task_method = preset_prepared.get('engine', {}).get('backend_params', modules.flags.get_engine_default_backend_params(engine))
    state_params.update({"task_method": task_method})

    if comfyd_active_checkbox:
        comfyd.stop()
   
    default_model = preset_prepared.get('base_model')
    previous_default_models = preset_prepared.get('previous_default_models', [])
    checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
    embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
    lora_downloads = preset_prepared.get('lora_downloads', {})
    vae_downloads = preset_prepared.get('vae_downloads', {})

    model_dtype = preset_prepared.get('engine', {}).get('backend_params', {}).get('base_model_dtype', '')
    if engine == 'SD3m' and  model_dtype == 'auto':
        base_model = comfy_task.get_default_base_SD3m_name()
        if shared.modelsinfo.exists_model(catalog="checkpoints", model_path=base_model):
            default_model = base_model
            preset_prepared['base_model'] = base_model
            checkpoint_downloads = {}
    if engine == 'Flux' and default_model=='auto':
        default_model = comfy_task.get_default_base_Flux_name('FluxS' in preset)
        preset_prepared['base_model'] = default_model
        if shared.modelsinfo.exists_model(catalog="checkpoints", model_path=default_model):
            checkpoint_downloads = {}
        else:
            checkpoint_downloads = {default_model: comfy_task.flux_model_urls[default_model]}
        if 'merged' in default_model:
            preset_prepared.update({'default_overwrite_step': 6})

    download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads)

    preset_url = preset_prepared.get('reference', get_preset_inc_url(preset))
    state_params.update({"__preset_url":preset_url})

    results = refresh_nav_bars(state_params)
    results += meta_parser.switch_layout_template(preset_prepared, state_params, preset_url)
    results += meta_parser.load_parameter_button_click(preset_prepared, is_generating, inpaint_mode)

    return results


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):

    if shared.args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not shared.args.always_download_new_model:
        if not os.path.isfile(shared.modelsinfo.get_file_path_by_name('checkpoints', default_model)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(shared.modelsinfo.get_file_path_by_name('checkpoints', alternative_model_name)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(shared.modelsinfo.get_file_path_by_name('checkpoints', file_name))
        load_file_from_url(url=url, model_dir=model_dir, file_name=os.path.basename(file_name))
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(shared.modelsinfo.get_file_path_by_name('loras', file_name))
        load_file_from_url(url=url, model_dir=model_dir, file_name=os.path.basename(file_name))
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads


from transformers import CLIPTokenizer
import shutil

cur_clip_path = os.path.join(config.path_clip_vision, "clip-vit-large-patch14")
if not os.path.exists(cur_clip_path):
    org_clip_path = os.path.join(shared.root, 'models/clip_vision/clip-vit-large-patch14')
    shutil.copytree(org_clip_path, cur_clip_path)
tokenizer = CLIPTokenizer.from_pretrained(cur_clip_path)
 
def remove_tokenizer():
    global tokenizer

    if 'tokenizer' in globals():
        del tokenizer
    return

def prompt_token_prediction(text, style_selections):
    global tokenizer, cur_clip_path
    if 'tokenizer' not in globals():
        globals()['tokenizer'] = None
    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(cur_clip_path)
    return len(tokenizer.tokenize(text))

    from extras.expansion import safe_str
    from modules.util import remove_empty_str
    import enhanced.translator as translator
    import enhanced.enhanced_parameters as enhanced_parameters
    import enhanced.wildcards as wildcards
    from modules.sdxl_styles import apply_style, fooocus_expansion

    prompt = translator.convert(text, enhanced_parameters.translation_methods)
    return len(tokenizer.tokenize(prompt))
    
    if fooocus_expansion in style_selections:
        use_expansion = True
        style_selections.remove(fooocus_expansion)
    else:
        use_expansion = False

    use_style = len(style_selections) > 0
    prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')

    prompt = prompts[0]

    if prompt == '':
        # disable expansion when empty since it is not meaningful and influences image prompt
        use_expansion = False

    extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
    task_rng = random.Random(random.randint(constants.MIN_SEED, constants.MAX_SEED))
    prompt, wildcards_arrays, arrays_mult, seed_fixed = wildcards.compile_arrays(prompt, task_rng)
    task_prompt = wildcards.apply_arrays(prompt, 0, wildcards_arrays, arrays_mult)
    task_prompt = wildcards.replace_wildcard(task_prompt, task_rng)
    task_extra_positive_prompts = [wildcards.apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
    positive_basic_workloads = []
    use_style = False
    if use_style:
        for s in style_selections:
            p, n = apply_style(s, positive=task_prompt)
            positive_basic_workloads = positive_basic_workloads + p
    else:
        positive_basic_workloads.append(task_prompt)
    positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
    positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
    #print(f'positive_basic_workloads:{positive_basic_workloads}')
    return len(tokenizer.tokenize(positive_basic_workloads[0]))


nav_name_list = get_preset_name_list()
system_message = get_system_message()
