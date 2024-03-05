all_args = {}
max_lora_number = 5
flag_disable_metadata = True

def get_diff_from_default(mode, *ctrls):
    global all_args, default, max_lora_number

    ctrls = list(ctrls)
    diff_dict = {}
    exclude_args = ['uov_input_image', 'inpaint_input_image', 'inpaint_mask_image', 'metadata_scheme']
    for key in all_args.keys():
        if key not in exclude_args and key != 'loras' and key != 'ip_ctrls':
            if key not in default.keys() or default[key] != ctrls[all_args[key]]:
                diff_dict.update({key: ctrls[all_args[key]] })
    loras = apply_enabled_loras([[bool(ctrls[all_args['loras']+i*3]), str(ctrls[all_args['loras']+1+i*3]), float(ctrls[all_args['loras']+2+i*3]) ] for i in range(max_lora_number)])
    if len(loras)>0:
        diff_dict.update({'loras':loras})
    #ip_ctrls =  

    if mode == 'log':
        log_ext = {}
        ads_params_for_log_list = ['adaptive_cfg', 'overwrite_step', 'overwrite_switch', 'inpaint_engine']
        for k in diff_dict.keys():
            if k in ads_params_for_log_list:
                log_ext.update({k: diff_ads[k]})
        diff_dict = log_ext

    return diff_dict

def apply_enabled_loras(loras):
        enabled_loras = []
        for lora_enabled, lora_model, lora_weight in loras:
            if lora_enabled:
                enabled_loras.append([lora_model, lora_weight])

        return enabled_loras

def get_dict_args(*ctrls):
    global all_args, max_lora_number

    ctrls = list(ctrls)[0]
    args_dict = {}
    exclude_args = ['uov_input_image', 'inpaint_input_image', 'inpaint_mask_image', 'metadata_scheme']
    for key in all_args.keys():
        if key != 'loras' and key != 'ip_ctrls':
            args_dict.update({key:ctrls[all_args[key]]})
    loras = apply_enabled_loras([[bool(ctrls[all_args['loras']+i*3]), str(ctrls[all_args['loras']+1+i*3]), float(ctrls[all_args['loras']+2+i*3]) ] for i in range(max_lora_number)])
    if len(loras)>0:
        args_dict.update({'loras': loras})
    return args_dict


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
    'inpaint_mask_upload_checkbox': True,
    'invert_mask_checkbox': False,
    'inpaint_erode_or_dilate': 0,
    'loras_min_weight': -2,
    'loras_max_weight': 2,
    'max_lora_number': 5,
    'max_image_number': 32,
    'image_number': 4,
    'output_format': 'png',
    'save_metadata_to_images': False,
    'metadata_scheme': 'fooocus',
    'input_image_checkbox': False,
    'advanced_checkbox': True
    }


def init_all_params_index(lora_number, disable_metadata):
    global all_args, max_lora_number, flag_disable_metadata
    
    max_lora_number = lora_number
    flag_disable_metadata = disable_metadata
    c = 0
    a = c + lora_number * 3
    b = a + (0 if disable_metadata else 2) 

    all_args = {
        'prompt': 0+c,
        'negative_prompt': 1+c,
        'style_selections': 2+c,
        'performance_selection': 3+c,
        'aspect_ratios_selection': 4+c,
        'image_number': 5+c,
        'output_format': 6+c,
        'image_seed': 7+c,
        'sharpness': 8+c,
        'guidance_scale': 9+c,
        'base_model': 10+c,
        'refiner_model': 11+c,
        'refiner_switch': 12+c,
        'loras': 13+c,
        'input_image_checkbox': 13+a,
        'current_tab': 14+a,
        'uov_method': 15+a,
        'uov_input_image': 16+a,
        'outpaint_selections': 17+a,
        'inpaint_input_image': 18+a,
        'inpaint_additional_prompt': 19+a,
        'inpaint_mask_image': 20+a,
        'disable_preview': 21+a,
        'disable_intermediate_results': 22+a,
        'disable_seed_increment': 23+a,
        'adm_scaler_positive': 24+a,
        'adm_scaler_negative': 25+a,
        'adm_scaler_end': 26+a,
        'adaptive_cfg': 27+a,
        'sampler_name': 28+a,
        'scheduler_name': 29+a,
        'overwrite_step': 30+a,
        'overwrite_switch': 31+a,
        'overwrite_width': 32+a,
        'overwrite_height': 33+a,
        'overwrite_vary_strength': 34+a,
        'overwrite_upscale_strength': 35+a,
        'mixing_image_prompt_and_vary_upscale': 36+a,
        'mixing_image_prompt_and_inpaint': 37+a,
        'debugging_cn_preprocessor': 38+a,
        'skipping_cn_preprocessor': 39+a,
        'canny_low_threshold': 40+a,
        'canny_high_threshold': 41+a,
        'refiner_swap_method': 42+a,
        'controlnet_softness': 43+a,
        'freeu_enabled': 44+a,
        'freeu_b1': 45+a,
        'freeu_b2': 46+a, 
        'freeu_s1': 47+a, 
        'freeu_s2': 48+a,
        'debugging_inpaint_preprocessor': 49+a,
        'inpaint_disable_initial_latent': 50+a, 
        'inpaint_engine': 51+a, 
        'inpaint_strength': 52+a, 
        'inpaint_respective_field': 53+a, 
        'inpaint_mask_upload_checkbox': 54+a, 
        'invert_mask_checkbox': 55+a, 
        'inpaint_erode_or_dilate': 56+a,
        'save_metadata_to_images': 57+a,
        'metadata_scheme': 58+a,
        'ip_ctrls': 57+b,
    }
