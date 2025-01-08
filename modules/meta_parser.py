import os
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
import random

import gradio as gr
from PIL import Image

import fooocus_version
import modules.config
import modules.sdxl_styles
import shared

from modules.flags import MetadataScheme, Performance, Steps, task_class_mapping, get_taskclass_by_fullname, default_class_params, scheduler_list, sampler_list
from modules.flags import SAMPLERS, CIVITAI_NO_KARRAS
from modules.util import quote, unquote, extract_styles_from_prompt, is_json, sha256, get_files_from_folder
import enhanced.all_parameters as ads
from modules.hash_cache import sha256_from_cache

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")

get_layout_visible = lambda x,y:gr.update(visible=x not in y)
get_layout_visible_inter = lambda x,y,z:gr.update(visible=x not in y, interactive=x not in z)
get_layout_invert_visible_inter = lambda x,y,z:gr.update(value=x not in z, visible=x not in y, interactive=x not in z)
get_layout_toggle_visible_inter = lambda x,y,z: gr.update(visible=x not in y, interactive=x not in z) if x not in z else gr.update(value=x not in z, visible=x not in y, interactive=x not in z)
get_layout_choices_visible_inter = lambda l,x,y,z:gr.update(choices=l, visible=x not in y, interactive=x not in z)
get_layout_setting_choices_visible_inter = lambda l,v,x,y,z:gr.update(choices=l, value=v, visible=x not in y, interactive=x not in z)
get_layout_empty_visible_inter = lambda x,y,z: gr.update(visible=x not in y, interactive=x not in z) if x not in z else gr.update(value=None, visible=x not in y, interactive=x not in z)
get_layout_update_label_visible_inter = lambda t,v,x,y,z:gr.update(label=t, value=v, visible=x not in y, interactive=x not in z) if t else gr.update(value=v, visible=x not in y, interactive=x not in z)
get_layout_update_label_and_choice_visible_inter = lambda t,l,v,x,y,z:gr.update(label=t, choices=l, value=v, visible=x not in y, interactive=x not in z) if t else gr.update(choices=l, value=v, visible=x not in y, interactive=x not in z)
get_layout_update_and_visible_inter = lambda v,x,y,z:gr.update(value=v, visible=x not in y, interactive=x not in z)

def get_layout_visible_inter_loras(y,z,max_number):
    x = 'loras'
    y1 = max_number if x in y else -1 
    for key in y:
        if '-' in key and x==key.split('-')[0]:
            y1 = int(key.split('-')[1])
            break
    z1 = max_number if x in z else -1
    for key in z:
        if '-' in key and x==key.split('-')[0]:
            z1 = int(key.split('-')[1])
            break
    results = []
    for i in range(max_number):
        results += [gr.update(visible= i+y1<max_number or y1<0, interactive= i+z1<max_number or z1<0)] * 3
    return results

def switch_scene_theme(state, image_number, theme=None):
    scenes = state.get("scene_frontend",{})
    visible = scenes.get('disvisible', [])
    inter = scenes.get('disinteractive', [])
    results = [None]
    themes = scenes.get('theme', [])
    index = themes.index(theme) if theme and themes and theme in themes else 0
    results.append(get_layout_setting_choices_visible_inter(themes, themes[index], 'scene_theme', visible, inter))
    title = scenes.get('additional_prompt_title', '')
    additional_prompt = scenes.get('additional_prompt', [])
    if isinstance(additional_prompt, dict):
        if index==0:
            additional_prompt = additional_prompt[next(iter(additional_prompt))] if additional_prompt else ''
        else:
            additional_prompt = additional_prompt[theme]
    results.append(get_layout_update_label_visible_inter(title, additional_prompt, 'scene_additional_prompt', visible, inter))
    aspect_ratio = scenes.get('aspect_ratio', [])
    if isinstance(aspect_ratio, dict):
        if index==0:
            aspect_ratio = aspect_ratio[next(iter(aspect_ratio))] if aspect_ratio else []
        else:
            aspect_ratio = aspect_ratio[theme]
    aspect_ratio = modules.flags.scene_aspect_ratios_mapping_list(aspect_ratio)
    aspect_ratio_default = '' if len(aspect_ratio)==0 else aspect_ratio[0]
    results.append(get_layout_setting_choices_visible_inter(aspect_ratio, aspect_ratio_default, 'scene_aspect_ratio', visible, inter))
    results.append(get_layout_update_and_visible_inter(image_number, 'scene_image_number', visible, inter))
    results.append(gr.update(interactive=False))   #generate_button
    return results


def switch_layout_template(presetdata: dict | str, state_params, preset_url=''):
    presetdata_dict = presetdata
    if isinstance(presetdata, str):
        presetdata_dict = json.loads(presetdata)
    assert isinstance(presetdata_dict, dict)
    enginedata_dict = presetdata_dict.get('engine', {})
    is_scene_frontend = 'scene_frontend' in enginedata_dict
    template_engine = get_taskclass_by_fullname(presetdata_dict.get('Backend Engine', presetdata_dict.get('backend_engine', 
        task_class_mapping[enginedata_dict.get('backend_engine', 'Fooocus')])))
    default_params = default_class_params[template_engine]
    visible = enginedata_dict.get('disvisible', default_params.get('disvisible', default_class_params['Fooocus']['disvisible']))
    inter = enginedata_dict.get('disinteractive', default_params.get('disinteractive', default_class_params['Fooocus']['disinteractive']))
    sampler_list = enginedata_dict.get('available_sampler_name', default_params.get('available_sampler_name', default_class_params['Fooocus']['available_sampler_name']))
    scheduler_list = enginedata_dict.get('available_scheduler_name', default_params.get('available_scheduler_name', default_class_params['Fooocus']['available_scheduler_name']))
    uov_method_list = enginedata_dict.get('available_uov_method', default_params.get('available_uov_method', default_class_params['Fooocus']['available_uov_method']))

    params_backend  = enginedata_dict.get('backend_params', modules.flags.get_engine_default_backend_params(template_engine))
    params_backend.update(dict(
            nickname=state_params["user_name"],
            user_did=state_params["user_did"],
            translation_methods=modules.config.default_translation_methods,
            backfill_prompt=modules.config.default_backfill_prompt,
            comfyd_active_checkbox=modules.config.default_comfyd_active_checkbox,
            backend_engine=template_engine 
            ))

    task_method = params_backend.get('task_method', None)
    base_model_list = modules.config.get_base_model_list(template_engine, task_method)
    engine_class_display = template_engine if template_engine in ['Flux', 'Kolors'] else 'SD15' if template_engine=='Comfy' else 'SDXL'

    results = [params_backend]
    results.append(get_layout_invert_visible_inter('advanced_checkbox', visible, inter))
    results.append(get_layout_visible_inter('performance_selection', visible, inter))
    results.append(get_layout_choices_visible_inter(scheduler_list, 'scheduler_name', visible, inter))
    results.append(get_layout_choices_visible_inter(sampler_list, 'sampler_name', visible, inter))
    if task_method and '_aio' in task_method:
        results.append(get_layout_visible_inter('input_image_checkbox', visible, inter))
    else:
        results.append(get_layout_toggle_visible_inter('input_image_checkbox', visible, inter))
    results.append(get_layout_toggle_visible_inter('prompt_panel_checkbox', visible, inter))
    results.append(get_layout_toggle_visible_inter('enhance_checkbox', visible, inter))
    results.append(get_layout_choices_visible_inter(base_model_list, 'base_model', visible, inter))
    results.append(get_layout_visible_inter('refiner_model', visible, inter))
    results.append(get_layout_visible_inter('overwrite_step', visible, inter))
    results.append(get_layout_visible_inter('guidance_scale', visible, inter))
    results.append(get_layout_empty_visible_inter('negative_prompt', visible, inter))
    results.append(gr.update(visible=True if 'blank.inc.html' not in preset_url else False))
    results.append(gr.update(visible=False)) # identity_dialog
    state_params['identity_dialog'] = False

    # [engine_class_display, uov_method, layer_method, layer_input_image, enhance_checkbox, enhance_input_image]
    results.append(engine_class_display)
    results.append(get_layout_setting_choices_visible_inter(uov_method_list, modules.flags.disabled, 'uov_method', visible, inter))
    results.append(get_layout_empty_visible_inter('layer_method', visible, inter) if not shared.token.is_guest(state_params["user_did"]) else gr.update(interactive=False))
    results.append(get_layout_empty_visible_inter('layer_input_image', visible, inter) if not shared.token.is_guest(state_params["user_did"]) else gr.update(interactive=False))
    results.append(get_layout_toggle_visible_inter('enhance_checkbox', visible, inter))
    results.append(get_layout_empty_visible_inter('enhance_input_image', visible, inter))
    results += get_layout_visible_inter_loras(visible, inter, modules.config.default_max_lora_number)

    #[output_format, inpaint_advanced_masking_checkbox, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, backfill_prompt, translation_methods, input_image_checkbox]
    # if default_X in config_prese then update the value to gr.X else update with default value in ads.default[X]
    update_value_if_existed = lambda x: gr.update() if x not in presetdata_dict else presetdata_dict[x]
    results.append(update_value_if_existed("output_format"))
    results.append(update_value_if_existed("inpaint_advanced_masking_checkbox"))
    results.append(update_value_if_existed("mixing_image_prompt_and_vary_upscale"))
    results.append(update_value_if_existed("mixing_image_prompt_and_inpaint"))
    results.append(update_value_if_existed("backfill_prompt"))
    results.append(update_value_if_existed("translation_methods"))
    results.append(False if template_engine not in ['Fooocus', 'Comfy'] and task_method and '_aio' not in task_method else update_value_if_existed("input_image_checkbox"))

    # [prompt_internal_panel, disable_intermediate_results, image_tools_checkbox, scene_panel, scene_theme], [generate_button, load_parameter_button]
    if is_scene_frontend:
        scenes = enginedata_dict.get("scene_frontend",{})
        scenes_visible = scenes.get('disvisible', [])
        visible.extend(scenes_visible)
        scenes_inter = scenes.get('disinteractive', [])
        inter.extend(scenes_inter)
        results.append(gr.update(visible=False))
        results.append(gr.update(value=True))
        results.append(gr.update(value=False))
        results.append(gr.update(visible=True))
        themes = scenes.get('theme', [])
        theme_default = themes[0] if themes else None
        themes_title = scenes.get('theme_title', '')
        results.append(get_layout_update_label_and_choice_visible_inter(themes_title, themes, None, 'scene_theme', visible, inter))
        results.append(gr.update(visible=True, interactive=False)) #generate_button
        results.append(gr.update(visible=False))                   #load_parameter_button
    else:
        results.append(gr.update(visible=True))
        results.append(gr.update(value=False))
        results.append(gr.update(value=True))
        results.append(gr.update(visible=False))
        results.append(gr.update(visible=True, interactive=True))
        results.append(gr.update(visible=True, interactive=True))  #generate_button
        results.append(gr.update(visible=False))                   #load_parameter_button

    if 'image_catalog_max_number' in presetdata_dict:
        state_params.update({'__max_catalog': presetdata_dict['image_catalog_max_number']})

    return results

def get_welcome_image(preset=None, is_mobile=False):
    path_welcome = os.path.abspath(f'./enhanced/attached/')
    if preset:
        suffix = 'w' if not is_mobile else 'm'
        file_welcome = os.path.join(path_welcome, f'welcome_{preset}_{suffix}.jpg')
        if os.path.exists(file_welcome):
            return file_welcome
    file_welcome = os.path.join(path_welcome, 'welcome.png')
    file_suffix = 'welcome_w' if not is_mobile else 'welcome_m'
    welcomes = [p for p in get_files_from_folder(path_welcome, ['.jpg', '.jpeg', 'png'], file_suffix, None) if not p.startswith('.')]
    if len(welcomes)>0:
        file_welcome = os.path.join(path_welcome, random.choice(welcomes))
    return file_welcome


def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool, inpaint_mode: str):
    loaded_parameter_dict = raw_metadata
    if isinstance(raw_metadata, str):
        loaded_parameter_dict = json.loads(raw_metadata)
    assert isinstance(loaded_parameter_dict, dict)
   
    preset = loaded_parameter_dict.get("preset", None)
    is_mobile = loaded_parameter_dict.get("is_mobile", False)
    results = [gr.update(value=f'{get_welcome_image(preset, is_mobile)}', visible=True), gr.update(visible=False), gr.update(visible=False), None] 

    get_image_number('image_number', 'Image Number', loaded_parameter_dict, results)
    get_str('prompt', 'Prompt', loaded_parameter_dict, results)
    get_str('negative_prompt', 'Negative Prompt', loaded_parameter_dict, results)
    get_list('styles', 'Styles', loaded_parameter_dict, results)
    performance = get_str('performance', 'Performance', loaded_parameter_dict, results)
    get_steps('steps', 'Steps', loaded_parameter_dict, results)
    get_number('overwrite_switch', 'Overwrite Switch', loaded_parameter_dict, results)
    get_resolution('resolution', 'Resolution', loaded_parameter_dict, results)
    get_number('guidance_scale', 'Guidance Scale', loaded_parameter_dict, results)
    get_number('sharpness', 'Sharpness', loaded_parameter_dict, results)
    get_adm_guidance('adm_guidance', 'ADM Guidance', loaded_parameter_dict, results)
    get_str('refiner_swap_method', 'Refiner Swap Method', loaded_parameter_dict, results)
    get_number('adaptive_cfg', 'CFG Mimicking from TSNR', loaded_parameter_dict, results)
    get_number('clip_skip', 'CLIP Skip', loaded_parameter_dict, results, cast_type=int)
    get_str('base_model', 'Base Model', loaded_parameter_dict, results)
    get_str('refiner_model', 'Refiner Model', loaded_parameter_dict, results)
    get_number('refiner_switch', 'Refiner Switch', loaded_parameter_dict, results)
    get_str('sampler', 'Sampler', loaded_parameter_dict, results)
    get_str('scheduler', 'Scheduler', loaded_parameter_dict, results)
    get_str('vae', 'VAE', loaded_parameter_dict, results)
    get_seed('seed', 'Seed', loaded_parameter_dict, results)
    get_inpaint_engine_version('inpaint_engine_version', 'Inpaint Engine Version', loaded_parameter_dict, results, inpaint_mode)
    get_inpaint_method('inpaint_method', 'Inpaint Mode', loaded_parameter_dict, results)

    #if is_generating:
    #    results.append(gr.update())
    #else:
    #    results.append(gr.update(visible=True))
    #results.append(gr.update(visible=False))

    get_freeu('freeu', 'FreeU', loaded_parameter_dict, results)

    # prevent performance LoRAs to be added twice, by performance and by lora
    performance_filename = None
    if performance is not None and performance in Performance.values():
        performance = Performance(performance)
        performance_filename = performance.lora_filename()

    for i in range(modules.config.default_max_lora_number):
        get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', loaded_parameter_dict, results, performance_filename)

    return results


def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str)
        results.append(h)
        return h
    except:
        results.append(gr.update())
        return None

def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = eval(h)
        assert isinstance(h, list)
        results.append(h)
    except:
        results.append(gr.update())
    if key in ['styles', 'Styles']:
        if h:
            for k in h:
                if k and 'styles_definition' in source_dict and k not in modules.sdxl_styles.styles and k in source_dict.get('styles_definition', default):
                    modules.sdxl_styles.styles.update({k: source_dict["styles_definition"][k]})


def get_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None, cast_type=float):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = cast_type(h)
        results.append(h)
    except:
        results.append(gr.update())


def get_image_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        h = min(h, modules.config.default_max_image_number)
        m = int(source_dict.get('max_image_number', ads.default["max_image_number"]))
        results.append(gr.update(value=h, maximum=m))
    except:
        results.append(1)


def get_steps(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        # if not in steps or in steps and performance is not the same
        performance_name = source_dict.get('performance', '').replace(' ', '_').replace('-', '_').casefold()
        performance_candidates = [key for key in Steps.keys() if key.casefold() == performance_name and Steps[key] == h]
        if len(performance_candidates) == 0:
            results.append(h)
            return
        results.append(-1)
    except:
        results.append(-1)


def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        width, height = eval(h)
        formatted = modules.flags.add_ratio(f'{width}*{height}')
        engine = get_taskclass_by_fullname(source_dict.get('Backend Engine', source_dict.get('backend_engine', task_class_mapping['Fooocus']))) 
        if 'engine' in source_dict:
            engine = source_dict['engine'].get('backend_engine', engine)
            template = source_dict['engine'].get('available_aspect_ratios_selection', default_class_params[engine].get('available_aspect_ratios_selection', default_class_params['Fooocus']['available_aspect_ratios_selection']))
        else:
            template = default_class_params[engine].get('available_aspect_ratios_selection', default_class_params['Fooocus']['available_aspect_ratios_selection'])
        if formatted in modules.flags.available_aspect_ratios_list[template]:
            h = f'{formatted},{template}'
            results.append(h)
            results.append(-1)
            results.append(-1)
        else:
            results.append(gr.update())
            results.append(int(width))
            results.append(int(height))
    except e:
        print(f'in except:{e}')
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        results.append(gr.update())
        results.append(gr.update())


def get_inpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, inpaint_mode: str, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        if inpaint_mode != modules.flags.inpaint_option_detail:
            results.append(h)
        else:
            results.append(gr.update())
        results.append(h)
        return h
    except:
        results.append(gr.update())
        results.append('empty')
        return None


def get_inpaint_method(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str) and h in modules.flags.inpaint_options
        results.append(h)
        for i in range(modules.config.default_enhance_tabs):
            results.append(h)
        return h
    except:
        results.append(gr.update())
        for i in range(modules.config.default_enhance_tabs):
            results.append(gr.update())


def get_adm_guidance(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        p, n, e = eval(h)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_freeu(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        b1, b2, s1, s2 = eval(h)
        results.append(True)
        results.append(float(b1))
        results.append(float(b2))
        results.append(float(s1))
        results.append(float(s2))
    except:
        results.append(False)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_lora(key: str, fallback: str | None, source_dict: dict, results: list, performance_filename: str | None):
    try:
        split_data = source_dict.get(key, source_dict.get(fallback)).split(' : ')
        enabled = True
        name = split_data[0]
        weight = split_data[1]

        if len(split_data) == 3:
            enabled = split_data[0] == 'True'
            name = split_data[1]
            weight = split_data[2]

        if name == performance_filename:
            raise Exception
        w_min = float(source_dict.get('loras_min_weight', ads.default['loras_min_weight']))
        w_max = float(source_dict.get('loras_max_weight', ads.default['loras_max_weight']))
        weight = float(weight)
        results.append(enabled)
        results.append(name)
        results.append(gr.update(value=weight, minimum=w_min, maximum=w_max))
    except:
        results.append(True)
        results.append('None')
        results.append(1)


def get_sha256(filepath):
    global hash_cache
    if not os.path.isfile(filepath):
        return ''
    if filepath not in hash_cache:
        filehash = shared.modelsinfo.get_file_muid(filepath)
        if not filehash:
            filehash = sha256(filepath)
        hash_cache[filepath] = filehash
    return hash_cache[filepath]


def parse_meta_from_preset(preset_content):
    assert isinstance(preset_content, dict)
    preset_prepared = {}
    items = preset_content

    for settings_key, meta_key in modules.config.possible_preset_keys.items():
        if settings_key == "default_loras":
            loras = getattr(modules.config, settings_key)
            if settings_key in items:
                loras = items[settings_key]
            for index, lora in enumerate(loras[:modules.config.default_max_lora_number]):
                if len(lora) == 2:
                    lora = (lora[0].replace('\\', os.sep).replace('/', os.sep), *lora[1:])
                elif  len(lora) == 3:
                    lora = (lora[0], lora[1].replace('\\', os.sep).replace('/', os.sep), *lora[2:])
                preset_prepared[f'lora_combined_{index + 1}'] = ' : '.join(map(str, lora))
        elif settings_key == "default_aspect_ratio":
            if settings_key in items and items[settings_key] is not None:
                default_aspect_ratio = items[settings_key]
                width, height = default_aspect_ratio.split('*')
            else:
                default_aspect_ratio = getattr(modules.config, settings_key)
                width, height = default_aspect_ratio.split('×')
                height = height[:height.index(" ")]
            preset_prepared[meta_key] = (width, height)
        elif settings_key not in items and settings_key in modules.config.allow_missing_preset_key:
            continue
        else:
            preset_prepared[meta_key] = items[settings_key] if settings_key in items and items[settings_key] is not None else getattr(modules.config, settings_key)

        if settings_key == "default_styles" or settings_key == "default_aspect_ratio":
            preset_prepared[meta_key] = str(preset_prepared[meta_key])
        if settings_key in ["default_model", "default_refiner"]:
            preset_prepared[meta_key] = preset_prepared[meta_key].replace('\\', os.sep).replace('/', os.sep)

    return preset_prepared


class MetadataParser(ABC):
    def __init__(self):
        self.raw_prompt: str = ''
        self.full_prompt: str = ''
        self.raw_negative_prompt: str = ''
        self.full_negative_prompt: str = ''
        self.steps: int = Steps.SPEED.value
        self.base_model_name: str = ''
        self.base_model_hash: str = ''
        self.refiner_model_name: str = ''
        self.refiner_model_hash: str = ''
        self.loras: list = []
        self.vae_name: str = ''
        self.styles_definition = {}

    @abstractmethod
    def get_scheme(self) -> MetadataScheme:
        raise NotImplementedError

    @abstractmethod
    def to_json(self, metadata: dict | str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def to_string(self, metadata: dict) -> str:
        raise NotImplementedError

    def set_data(self, raw_prompt, full_prompt, raw_negative_prompt, full_negative_prompt, steps, base_model_name,
                 refiner_model_name, loras, vae_name, styles_definition):
        self.raw_prompt = raw_prompt
        self.full_prompt = full_prompt
        self.raw_negative_prompt = raw_negative_prompt
        self.full_negative_prompt = full_negative_prompt
        self.steps = steps
        self.base_model_name = Path(base_model_name).stem

        if base_model_name not in ['', 'None']:
            base_model_path = shared.modelsinfo.get_model_filepath('checkpoints', base_model_name)
            self.base_model_hash = shared.modelsinfo.get_file_muid(base_model_path)

        if refiner_model_name not in ['', 'None']:
            self.refiner_model_name = Path(refiner_model_name).stem
            refiner_model_path = shared.modelsinfo.get_model_filepath('checkpoints', refiner_model_name)
            self.refiner_model_hash = shared.modelsinfo.get_file_muid(refiner_model_path)

        self.loras = []
        for (lora_name, lora_weight) in loras:
            if lora_name != 'None':
                lora_path = shared.modelsinfo.get_model_filepath('loras', lora_name)
                lora_hash = shared.modelsinfo.get_file_muid(lora_path)
                self.loras.append((Path(lora_name).stem, lora_weight, lora_hash))
        self.vae_name = Path(vae_name).stem
        if styles_definition != 'None':
            self.styles_definition = styles_definition


class A1111MetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.A1111

    fooocus_to_a1111 = {
        'raw_prompt': 'Raw prompt',
        'raw_negative_prompt': 'Raw negative prompt',
        'negative_prompt': 'Negative prompt',
        'styles': 'Styles',
        'performance': 'Performance',
        'steps': 'Steps',
        'sampler': 'Sampler',
        'scheduler': 'Scheduler',
        'vae': 'VAE',
        'guidance_scale': 'CFG scale',
        'seed': 'Seed',
        'resolution': 'Size',
        'sharpness': 'Sharpness',
        'adm_guidance': 'ADM Guidance',
        'refiner_swap_method': 'Refiner Swap Method',
        'adaptive_cfg': 'Adaptive CFG',
        'clip_skip': 'Clip skip',
        'overwrite_switch': 'Overwrite Switch',
        'freeu': 'FreeU',
        'base_model': 'Model',
        'base_model_hash': 'Model hash',
        'refiner_model': 'Refiner',
        'refiner_model_hash': 'Refiner hash',
        'lora_hashes': 'Lora hashes',
        'lora_weights': 'Lora weights',
        'created_by': 'User',
        'version': 'Version',
        'backend_engine': 'Backend Engine'
    }

    def to_json(self, metadata: str) -> dict:
        metadata_prompt = ''
        metadata_negative_prompt = ''

        done_with_prompt = False

        *lines, lastline = metadata.strip().split("\n")
        if len(re_param.findall(lastline)) < 3:
            lines.append(lastline)
            lastline = ''

        for line in lines:
            line = line.strip()
            if line.startswith(f"{self.fooocus_to_a1111['negative_prompt']}:"):
                done_with_prompt = True
                line = line[len(f"{self.fooocus_to_a1111['negative_prompt']}:"):].strip()
            if done_with_prompt:
                metadata_negative_prompt += ('' if metadata_negative_prompt == '' else "\n") + line
            else:
                metadata_prompt += ('' if metadata_prompt == '' else "\n") + line

        found_styles, prompt, negative_prompt = extract_styles_from_prompt(metadata_prompt, metadata_negative_prompt)

        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }

        for k, v in re_param.findall(lastline):
            try:
                if v != '' and v[0] == '"' and v[-1] == '"':
                    v = unquote(v)

                m = re_imagesize.match(v)
                if m is not None:
                    data['resolution'] = str((m.group(1), m.group(2)))
                else:
                    data[list(self.fooocus_to_a1111.keys())[list(self.fooocus_to_a1111.values()).index(k)]] = v
            except Exception:
                print(f"Error parsing \"{k}: {v}\"")

        # workaround for multiline prompts
        if 'raw_prompt' in data:
            data['prompt'] = data['raw_prompt']
            raw_prompt = data['raw_prompt'].replace("\n", ', ')
            if metadata_prompt != raw_prompt and modules.sdxl_styles.fooocus_expansion not in found_styles:
                found_styles.append(modules.sdxl_styles.fooocus_expansion)

        if 'raw_negative_prompt' in data:
            data['negative_prompt'] = data['raw_negative_prompt']

        data['styles'] = str(found_styles)

        # try to load performance based on steps, fallback for direct A1111 imports
        if 'steps' in data and 'performance' in data is None:
            try:
                data['performance'] = Performance.by_steps(data['steps']).value
            except ValueError | KeyError:
                pass

        if 'sampler' in data:
            data['sampler'] = data['sampler'].replace(' Karras', '')
            # get key
            for k, v in SAMPLERS.items():
                if v == data['sampler']:
                    data['sampler'] = k
                    break

        for key in ['base_model', 'refiner_model', 'vae']:
            if key in data:
                if key == 'vae':
                    self.add_extension_to_filename(data, modules.config.vae_filenames, 'vae')
                else:
                    self.add_extension_to_filename(data, modules.config.model_filenames, key)

        lora_data = ''
        if 'lora_weights' in data and data['lora_weights'] != '':
            lora_data = data['lora_weights']
        elif 'lora_hashes' in data and data['lora_hashes'] != '' and data['lora_hashes'].split(', ')[0].count(':') == 2:
            lora_data = data['lora_hashes']

        if lora_data != '':
            for li, lora in enumerate(lora_data.split(', ')):
                lora_split = lora.split(': ')
                lora_name = lora_split[0]
                lora_weight = lora_split[2] if len(lora_split) == 3 else lora_split[1]
                for filename in modules.config.lora_filenames:
                    path = Path(filename)
                    if lora_name == path.stem:
                        data[f'lora_combined_{li + 1}'] = f'{filename} : {lora_weight}'
                        break

        return data

    def to_string(self, metadata: dict) -> str:
        data = {k: v for _, k, v in metadata}

        width, height = eval(data['resolution'])

        sampler = data['sampler']
        scheduler = data['scheduler']

        if sampler in SAMPLERS and SAMPLERS[sampler] != '':
            sampler = SAMPLERS[sampler]
            if sampler not in CIVITAI_NO_KARRAS and scheduler == 'karras':
                sampler += f' Karras'

        generation_params = {
            self.fooocus_to_a1111['steps']: self.steps,
            self.fooocus_to_a1111['sampler']: sampler,
            self.fooocus_to_a1111['seed']: data['seed'],
            self.fooocus_to_a1111['resolution']: f'{width}x{height}',
            self.fooocus_to_a1111['guidance_scale']: data['guidance_scale'],
            self.fooocus_to_a1111['sharpness']: data['sharpness'],
            self.fooocus_to_a1111['adm_guidance']: data['adm_guidance'],
            self.fooocus_to_a1111['base_model']: Path(data['base_model']).stem,
            self.fooocus_to_a1111['base_model_hash']: self.base_model_hash,

            self.fooocus_to_a1111['performance']: data['performance'],
            self.fooocus_to_a1111['scheduler']: scheduler,
            self.fooocus_to_a1111['vae']: Path(data['vae']).stem,
            # workaround for multiline prompts
            self.fooocus_to_a1111['raw_prompt']: self.raw_prompt,
            self.fooocus_to_a1111['raw_negative_prompt']: self.raw_negative_prompt,
        }

        if self.refiner_model_name not in ['', 'None']:
            generation_params |= {
                self.fooocus_to_a1111['refiner_model']: self.refiner_model_name,
                self.fooocus_to_a1111['refiner_model_hash']: self.refiner_model_hash
            }

        for key in ['adaptive_cfg', 'clip_skip', 'overwrite_switch', 'refiner_swap_method', 'freeu']:
            if key in data:
                generation_params[self.fooocus_to_a1111[key]] = data[key]

        if len(self.loras) > 0:
            lora_hashes = []
            lora_weights = []
            for index, (lora_name, lora_weight, lora_hash) in enumerate(self.loras):
                # workaround for Fooocus not knowing LoRA name in LoRA metadata
                lora_hashes.append(f'{lora_name}: {lora_hash}')
                lora_weights.append(f'{lora_name}: {lora_weight}')
            lora_hashes_string = ', '.join(lora_hashes)
            lora_weights_string = ', '.join(lora_weights)
            generation_params[self.fooocus_to_a1111['lora_hashes']] = lora_hashes_string
            generation_params[self.fooocus_to_a1111['lora_weights']] = lora_weights_string

        generation_params[self.fooocus_to_a1111['version']] = data['version']

        if modules.config.metadata_created_by != '':
            generation_params[self.fooocus_to_a1111['created_by']] = modules.config.metadata_created_by

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if
             v is not None])
        positive_prompt_resolved = ', '.join(self.full_prompt)
        negative_prompt_resolved = ', '.join(self.full_negative_prompt)
        negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}" if negative_prompt_resolved else ""
        return f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()

    @staticmethod
    def add_extension_to_filename(data, filenames, key):
        for filename in filenames:
            path = Path(filename)
            if data[key] == path.stem:
                data[key] = filename
                break


class FooocusMetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.FOOOCUS

    def to_json(self, metadata: dict) -> dict:

        for key, value in metadata.items():
            if value in ['', 'None']:
                continue
            if key in ['base_model', 'refiner_model']:
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.model_filenames)
            elif key.startswith('lora_combined_'):
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.lora_filenames)
            elif key == 'vae':
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.vae_filenames)
            else:
                continue

        return metadata

    def to_string(self, metadata: list) -> str:
        for li, (label, key, value) in enumerate(metadata):
            # remove model folder paths from metadata
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                name = Path(name).stem
                value = f'{name} : {weight}'
                metadata[li] = (label, key, value)

        res = {k: v for _, k, v in metadata}

        res['full_prompt'] = self.full_prompt
        res['full_negative_prompt'] = self.full_negative_prompt
        res['steps'] = self.steps
        res['base_model'] = self.base_model_name
        res['base_model_hash'] = self.base_model_hash

        if self.refiner_model_name not in ['', 'None']:
            res['refiner_model'] = self.refiner_model_name
            res['refiner_model_hash'] = self.refiner_model_hash

        res['vae'] = self.vae_name
        res['loras'] = self.loras

        if modules.config.metadata_created_by != '':
            res['created_by'] = modules.config.metadata_created_by

        return json.dumps(dict(sorted(res.items())))

    @staticmethod
    def replace_value_with_filename(key, value, filenames):
        for filename in filenames:
            path = Path(filename)
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                if name == path.stem:
                    return f'{filename} : {weight}'
            elif value == path.stem:
                return filename

        return None

class SIMPLEMetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.SIMPLE


    def to_json(self, metadata: dict) -> dict:
        engine = get_taskclass_by_fullname(metadata.get('Backend Engine', metadata.get('backend_engine', task_class_mapping['Fooocus'])))
        model_filenames = modules.config.get_base_model_list(engine)
        for key, value in metadata.items():
            if value in ['', 'None']:
                if key in ['base_model', 'refiner_model', 'Base Model', 'Refiner Model']:
                    metadata[key] = 'None'
                continue
            if key in ['base_model', 'refiner_model', 'Base Model', 'Refiner Model']:
                metadata[key] = self.replace_value_with_filename(key, value, model_filenames)
                if metadata[key]=='None':
                    print(f'[MetaParser] ⚠️  WARNING! The model is not available in the local: {value}.')
            elif key.startswith('LoRA '):
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.lora_filenames)
            elif key in ['vae', 'VAE']:
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.vae_filenames)
            else:
                continue

        return metadata

    def to_string(self, metadata: list) -> str:
        for li, (label, key, value) in enumerate(metadata):
            # remove model folder paths from metadata
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                name = Path(name).stem
                value = f'{name} : {weight}'
                metadata[li] = (label, key, value)

        res = {k: v for k, _, v in metadata}

        res['Full Prompt'] = self.full_prompt
        res['Full Negative Prompt'] = self.full_negative_prompt
        res['Steps'] = self.steps
        res['Base Model'] = self.base_model_name
        res['Base Model Hash'] = self.base_model_hash

        if self.refiner_model_name not in ['', 'None']:
            res['Refiner Model'] = self.refiner_model_name
            res['Refiner Model Hash'] = self.refiner_model_hash

        res['VAE'] = self.vae_name
        res['LoRAs'] = self.loras
        res['styles_definition'] = self.styles_definition

        if modules.config.metadata_created_by != '':
            res['User'] = modules.config.metadata_created_by

        return json.dumps(dict(sorted(res.items())))

    @staticmethod
    def replace_value_with_filename(key, value, filenames):
        if key in ['vae', 'VAE'] and value=='Default (model)':
            return value
        for filename in filenames:
            path = Path(filename)
            if key.startswith('LoRA '):
                name, weight = value.split(' : ')
                if Path(name).stem == path.stem or name == path.stem:
                    return f'{filename} : {weight}'
            elif Path(value).stem == path.stem or value == path.stem:
                return filename
        return 'None'


def get_metadata_parser(metadata_scheme: MetadataScheme) -> MetadataParser:
    match metadata_scheme:
        case MetadataScheme.FOOOCUS:
            return FooocusMetadataParser()
        case MetadataScheme.A1111:
            return A1111MetadataParser()
        case MetadataScheme.SIMPLE:
            return SIMPLEMetadataParser()
        case _:
            raise NotImplementedError


def read_info_from_image(file) -> tuple[str | None, MetadataScheme | None]:
    items = (file.info or {}).copy()

    parameters = items.pop('parameters', None)
    metadata_scheme = items.pop('fooocus_scheme', None)
    exif = items.pop('exif', None)
    if not parameters and 'Comment' in items:
        metadata_scheme = 'simple'
        parameters = items.pop('Comment', None)

    if parameters is not None and is_json(parameters):
        parameters = json.loads(parameters)
        parameters = params_lora_fixed(parameters)
    elif exif is not None:
        exif = file.getexif()
        # 0x9286 = UserComment
        parameters = exif.get(0x9286, None)
        # 0x927C = MakerNote
        metadata_scheme = exif.get(0x927C, None)
        
        if parameters and is_json(parameters):
            parameters = json.loads(parameters)
            parameters = params_lora_fixed(parameters)

    try:
        if metadata_scheme == 'fooocus':
            metadata_scheme = 'simple'
            parameters.update({'metadata_scheme': 'simple'})
        metadata_scheme = MetadataScheme(metadata_scheme)
    except ValueError:
        metadata_scheme = None

        # broad fallback
        #if isinstance(parameters, dict):
        #    metadata_scheme = MetadataScheme.FOOOCUS

        if isinstance(parameters, str):
            metadata_scheme = MetadataScheme.A1111
    return parameters, metadata_scheme

def params_lora_fixed(parameters):
    loras_p = {k: v for k, v in parameters.items() if k.startswith("LoRA [")}
    if loras_p:
        for k, _ in loras_p.items():
            del parameters[k]
        loras_p = {f'LoRA {i}': f'{k[6:-8]} : {v}' for i, (k, v) in enumerate(loras_p.items(), 1)}
        parameters.update(loras_p)
    return parameters

def get_exif(metadata: str | None, metadata_scheme: str):
    exif = Image.Exif()
    # tags see see https://github.com/python-pillow/Pillow/blob/9.2.x/src/PIL/ExifTags.py
    # 0x9286 = UserComment
    exif[0x9286] = metadata
    # 0x0131 = Software
    import enhanced.version as version
    exif[0x0131] = f'{version.branch}_{version.get_simplesdxl_ver()}'
    # 0x927C = MakerNote
    exif[0x927C] = metadata_scheme
    return exif
