import re
import shared

bool_map = {
    "true": True,
    "false": False
}
float_pattern = r"^[+-]?\d*\.?\d+$|^[+-]?\d+\.?\d*$"

cache_vars = {}

def convert_value(value):
    global bool_map, float_pattern
    
    if value.lower() in bool_map:
        value = bool_map[value.lower()]
    elif value.isdigit() or ((value.startswith('-') or value.startswith('+')) and value[1:].isdigit()):
        value = int(value)
    elif re.match(float_pattern, value):
        value = float(value)
    elif value == 'None' or value == 'Unknown':
        value = None
    return value

def get_admin_default(admin_key):
    global cache_vars, default

    cache_key = f'admin_{admin_key}'
    if cache_key in cache_vars:
        return cache_vars[cache_key]
    admin_value = shared.token.get_local_admin_vars(admin_key).strip()
    if admin_value is None or admin_value=="None" or admin_value=="Unknown":
        if admin_key in default:
            admin_value = str(default[admin_key])
        else:
            admin_value = 'None'
    admin_value = convert_value(admin_value)
    cache_vars[cache_key] = admin_value
    return admin_value

def get_user_default(user_key, state, config_default=None):
    global cache_vars, default

    cache_key = f'{state["__session"]}_{user_key}'
    if cache_key in cache_vars:
        return cache_vars[cache_key]
    user_value = shared.token.get_local_vars(user_key, 'None', state["__session"], state["ua_hash"]).strip()
    if user_value is None or user_value=="None" or user_value=="Unknown":
        if config_default is not None:
            user_value = str(config_default)
        else:
            if user_key in default:
                user_value = str(default[user_key])
            else:
                user_value = 'None'
    user_value = convert_value(user_value)
    cache_vars[cache_key] = user_value
    return user_value

def set_admin_default_value(key, value, state):
    global cache_vars

    cache_key = f'admin_{key}'
    cache_vars[cache_key] = value
    shared.token.set_local_admin_vars(key, str(value), state["__session"], state["ua_hash"])

def set_user_default_value(key, value, state):
    global cache_vars

    cache_key = f'{state["__session"]}_{key}'
    cache_vars[cache_key] = value
    shared.token.set_local_vars(key, str(value), state["__session"], state["ua_hash"])

default = {
    'disable_preview': False,
    'adm_scaler_positive': 1.5,
    'adm_scaler_negative': 0.8,
    'adm_scaler_end': 0.3,
    'adaptive_cfg': 7.0,
    'sampler_name': 'dpmpp_2m_sde_gpu',
    'scheduler_name': 'karras',
    'generate_image_grid': False,
    'overwrite_step': -1,
    'overwrite_switch': -1,
    'overwrite_width': -1,
    'overwrite_height': -1,
    'overwrite_vary_strength': -1,
    'overwrite_upscale_strength': -1,
    'mixing_image_prompt_and_vary_upscale': False,
    'mixing_image_prompt_and_inpaint': False,
    'debugging_cn_preprocessor': False,
    'skipping_cn_preprocessor': False,
    'controlnet_softness': 0.25,
    'canny_low_threshold': 64,
    'canny_high_threshold': 128,
    'refiner_swap_method': 'joint',
    'freeu': [1.01, 1.02, 0.99, 0.95],
    'debugging_inpaint_preprocessor': False,
    'inpaint_disable_initial_latent': False,
    'inpaint_engine': 'v2.6',
    'inpaint_strength': 1,
    'inpaint_respective_field': 0.618,
    'inpaint_advanced_masking_checkbox': True,
    'invert_mask_checkbox': False,
    'inpaint_erode_or_dilate': 0,
    'loras_min_weight': -2,
    'loras_max_weight': 2,
    'max_lora_number': 5,
    'max_image_number': 32,
    'image_number': 2,
    'output_format': 'jpeg',
    'save_metadata_to_images': False,
    'metadata_scheme': 'simple',
    'input_image_checkbox': False,
    'advanced_checkbox': True,
    'backfill_prompt': False,
    'translation_methods': 'Third APIs',
    'backend': 'SDXL',
    'comfyd_active_checkbox': False,
    'image_catalog_max_number': 65,
    'clip_skip': 2,
    'vae': 'Default (model)',
    'developer_debug_mode_checkbox': True,
    'fast_comfyd_checkbox': False,
    'reserved_vram': 0,
    'minicpm_checkbox': False,
    'advanced_logs': False,
    'wavespeed_strength': 0.12,
    'p2p_active_checkbox': False,
    'p2p_remote_process': 'Disable',
    'p2p_in_did_list': '',
    'p2p_out_did_list': '',
    'style_preview_checkbox': True,
    }


all_args = [
        'generate_image_grid',
        'prompt',
        'negative_prompt',
        'style_selections',
        'performance_selection',
        'aspect_ratios_selection',
        'image_number',
        'output_format',
        'image_seed',
        'read_wildcards_in_order',
        'sharpness',
        'guidance_scale',
        'base_model',
        'refiner_model',
        'refiner_switch',
        'loras',
        'input_image_checkbox',
        'current_tab',
        'uov_method',
        'uov_input_image',
        'outpaint_selections',
        'inpaint_input_image',
        'inpaint_additional_prompt',
        'inpaint_mask_image',
        'layer_methon',
        'layer_input_image',
        'iclight_enable',
        'iclight_source_radio',
        'disable_preview',
        'disable_intermediate_results',
        'disable_seed_increment',
        'black_out_nsfw',
        'adm_scaler_positive',
        'adm_scaler_negative',
        'adm_scaler_end',
        'adaptive_cfg',
        'clip_skip',
        'sampler_name',
        'scheduler_name',
        'vae_name',
        'overwrite_step',
        'overwrite_switch',
        'overwrite_width',
        'overwrite_height',
        'overwrite_vary_strength',
        'overwrite_upscale_strength',
        'mixing_image_prompt_and_vary_upscale',
        'mixing_image_prompt_and_inpaint',
        'debugging_cn_preprocessor',
        'skipping_cn_preprocessor',
        'canny_low_threshold',
        'canny_high_threshold',
        'refiner_swap_method',
        'controlnet_softness',
        'freeu_enabled',
        'freeu_b1',
        'freeu_b2', 
        'freeu_s1', 
        'freeu_s2',
        'debugging_inpaint_preprocessor',
        'inpaint_disable_initial_latent', 
        'inpaint_engine', 
        'inpaint_strength', 
        'inpaint_respective_field', 
        'inpaint_advanced_masking_checkbox', 
        'invert_mask_checkbox', 
        'inpaint_erode_or_dilate',
        'params_backend',
        'save_final_enhanced_image_only',
        'save_metadata_to_images',
        'metadata_scheme',
        'ip_ctrls',
        'debugging_dino', 
        'dino_erode_or_dilate', 
        'debugging_enhance_masks_checkbox',
        'enhance_input_image',
        'enhance_checkbox', 
        'enhance_uov_method', 
        'enhance_uov_processing_order',
        'enhance_uov_prompt_type',
        'enhance_ctrls'
    ]

backend_args = [
        'backend_engine',
        'preset',
        'task_method',
        'nickname',
        'user_did',
        'scene_frontend',
        'scene_theme',
        'scene_canvas_image',
        'scene_canvas_mask',
        'scene_input_image1',
        'scene_input_image2',
        'scene_additional_prompt',
        'scene_steps',
        'scene_aspect_ratio',
        'scene_var_number',
        'scene_image_number',
        'base_model_dtype',
        'clip_model',
        'llms_model',
        'display_step',
        'hires_fix_blurred',
        'hires_fix_weight',
        'hires_fix_stop',
        'tiling',
        'tiled_offset_x'
        'tiled_offset_y'
    ]

def normalization(args, default_max_lora_number, default_controlnet_image_count, default_enhance_tabs):
    args_norm = []
    args_norm += args[:15]
    index = 15

    lora_list = [[bool(args[index + i * 3]), str(args[index + i * 3 + 1]), float(args[index + i * 3 + 2])] 
                 for i in range(default_max_lora_number)]
    args_norm.append(lora_list)
    index += default_max_lora_number * 3

    args_norm += args[index:index + 51]
    index += 51

    args_norm.append(normalization_backend(args[index]))
    index += 1

    args_norm += args[index:index + 3]
    index += 3

    controlnet_list = [[args[index + i * 4 + j] for j in range(4)] 
                       for i in range(default_controlnet_image_count)]
    args_norm.append(controlnet_list)
    index += default_controlnet_image_count * 4

    args_norm += args[index:index + 8]
    index += 8

    enhance_tabs_list = [[args[index + i * 16 + j] for j in range(16)] 
                         for i in range(default_enhance_tabs)]
    args_norm.append(enhance_tabs_list)
    
    return args_norm

def normalization_backend(args):
    global backend_args

    args_norm = [args[k] if k in args else None for k in backend_args]
    if args_norm[7] is not None:
        args_norm[7] = args['scene_canvas_image']['image']
        args_norm[8] = args['scene_canvas_image']['mask']

    #print(f'normalization_backend:{args_norm}')
    return args_norm

def convert_dict(args):
    global backend_args

    if len(backend_args) != len(args):
        raise ValueError("args length mismatch: takes {len(backend_args)} arguments but {len(args)} were given.")
    args_dict = {backend_args[i]: v for i, v in enumerate(args) if v is not None}
    if 'scene_canvas_image' in args_dict and 'scene_canvas_mask' in args_dict:
        args_dict['scene_canvas_image'] = { 'image': args_dict['scene_canvas_image'], 'mask': args_dict.pop('scene_canvas_mask') }
    #print(f'convert_dict:{args_dict}')
    return args_dict

