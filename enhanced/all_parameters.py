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
    'enhance_mask_model': 'sam',
    'enhance_mask_cloth_category': 'full',
    'enhance_mask_sam_model': 'vit_b',
    'enhance_mask_text_threshold': 0.25,
    'enhance_mask_box_threshold': 0.3,
    'enhance_mask_sam_max_detections': 0,
    'enhance_inpaint_disable_initial_latent': False,
    'enhance_inpaint_engine': 'None',
    'enhance_inpaint_strength': 0.5,
    'enhance_inpaint_respective_field': 0,
    'enhance_inpaint_erode_or_dilate': 0,
    'enhance_mask_invert': False,
    }

