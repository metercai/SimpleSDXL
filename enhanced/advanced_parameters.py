disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name,  \
    scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height, \
    overwrite_vary_strength, overwrite_upscale_strength, \
    mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
    debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, \
    refiner_swap_method, \
    freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
    debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, \
    inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate = [None] * 35


def set_all_advanced_parameters(*args):
    global disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name, \
        scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height, \
        overwrite_vary_strength, overwrite_upscale_strength, \
        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, \
        refiner_swap_method, \
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
        debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, \
        inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate

    disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name, \
        scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height, \
        overwrite_vary_strength, overwrite_upscale_strength, \
        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, \
        refiner_swap_method, \
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
        debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, \
        inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate = args

    return


def get_diff_for_log_ext():
    log_ext = {}
    diff_ads =  get_diff_from_default()
    ads_params_list = ['adaptive_cfg', 'overwrite_step', 'overwrite_switch', 'inpaint_engine']
    for k in diff_ads.keys():
        if k in ads_params_list:
            log_ext.update({f'{k}': diff_ads[k]})
    return log_ext

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
    'freeu_enabled': False,
    'freeu_b1': 1.01,
    'freeu_b2': 1.02,
    'freeu_s1': 0.99,
    'freeu_s2': 0.95,
    'debugging_inpaint_preprocessor': False,
    'inpaint_disable_initial_latent': False,
    'inpaint_engine': 'v2.6',
    'inpaint_strength': 1,
    'inpaint_respective_field': 0.618,
    'inpaint_mask_upload_checkbox': True,
    'invert_mask_checkbox': False,
    'inpaint_erode_or_dilate': 0
    }

def get_diff_from_default():
    global disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name, \
        scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height, \
        overwrite_vary_strength, overwrite_upscale_strength, \
        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, \
        refiner_swap_method, \
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
        debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, \
        inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate

    diff_dict = {}
    if default["disable_preview"] != disable_preview:
        diff_dict.update({"disable_preview": disable_preview })
    elif default["adm_scaler_positive"] != adm_scaler_positive:
        diff_dict.update({"adm_scaler_positive": adm_scaler_positive })
    elif default["adm_scaler_negative"] != adm_scaler_negative:
        diff_dict.update({"adm_scaler_negative": adm_scaler_negative })
    elif default["adm_scaler_end"] != adm_scaler_end:
        diff_dict.update({"adm_scaler_end": adm_scaler_end })
    elif default["adaptive_cfg"] != adaptive_cfg:
        diff_dict.update({"adaptive_cfg": adaptive_cfg })
    elif default["sampler_name"] != sampler_name:
        diff_dict.update({"sampler_name": sampler_name })
    elif default["scheduler_name"] != scheduler_name:
        diff_dict.update({"scheduler_name": scheduler_name })
    elif default["generate_image_grid"] != generate_image_grid:
        diff_dict.update({"generate_image_grid": generate_image_grid })
    elif default["overwrite_step"] != overwrite_step:
        diff_dict.update({"overwrite_step": overwrite_step })
    elif default["overwrite_switch"] != overwrite_switch:
        diff_dict.update({"overwrite_switch": overwrite_switch })
    elif default["overwrite_width"] != overwrite_width:
        diff_dict.update({"overwrite_width": overwrite_width })
    elif default["overwrite_height"] != overwrite_height:
        diff_dict.update({"overwrite_height": overwrite_height })
    elif default["overwrite_vary_strength"] != overwrite_vary_strength:
        diff_dict.update({"overwrite_vary_strength": overwrite_vary_strength })
    elif default["overwrite_upscale_strength"] != overwrite_upscale_strength:
        diff_dict.update({"overwrite_upscale_strength": overwrite_upscale_strength })
    elif default["mixing_image_prompt_and_vary_upscale"] != mixing_image_prompt_and_vary_upscale:
        diff_dict.update({"mixing_image_prompt_and_vary_upscale": mixing_image_prompt_and_vary_upscale })
    elif default["mixing_image_prompt_and_inpaint"] != mixing_image_prompt_and_inpaint:
        diff_dict.update({"mixing_image_prompt_and_inpaint": mixing_image_prompt_and_inpaint })
    elif default["debugging_cn_preprocessor"] != debugging_cn_preprocessor:
        diff_dict.update({"debugging_cn_preprocessor": debugging_cn_preprocessor })
    elif default["skipping_cn_preprocessor"] != skipping_cn_preprocessor:
        diff_dict.update({"skipping_cn_preprocessor": skipping_cn_preprocessor })
    elif default["controlnet_softness"] != controlnet_softness:
        diff_dict.update({"controlnet_softness": controlnet_softness })
    elif default["canny_low_threshold"] != canny_low_threshold:
        diff_dict.update({"canny_low_threshold": canny_low_threshold })
    elif default["canny_high_threshold"] != canny_high_threshold:
        diff_dict.update({"canny_high_threshold": canny_high_threshold })
    elif default["refiner_swap_method"] != refiner_swap_method:
        diff_dict.update({"refiner_swap_method": refiner_swap_method })
    elif default["freeu_enabled"] != freeu_enabled:
        diff_dict.update({"freeu_enabled": freeu_enabled })
    elif default["freeu_b1"] != freeu_b1:
        diff_dict.update({"freeu_b1": freeu_b1 })
    elif default["freeu_b2"] != freeu_b2:
        diff_dict.update({"freeu_b2": freeu_b2 })
    elif default["freeu_s1"] != freeu_s1:
        diff_dict.update({"freeu_s1": freeu_s1 })
    elif default["freeu_s2"] != freeu_s2:
        diff_dict.update({"freeu_s2": freeu_s2 })
    elif default["debugging_inpaint_preprocessor"] != debugging_inpaint_preprocessor:
        diff_dict.update({"debugging_inpaint_preprocessor": debugging_inpaint_preprocessor })
    elif default["inpaint_disable_initial_latent"] != inpaint_disable_initial_latent:
        diff_dict.update({"inpaint_disable_initial_latent": inpaint_disable_initial_latent })
    elif default["inpaint_engine"] != inpaint_engine:
        diff_dict.update({"inpaint_engine": inpaint_engine })
    elif default["inpaint_strength"] != inpaint_strength:
        diff_dict.update({"inpaint_strength": inpaint_strength })
    elif default["inpaint_respective_field"] != inpaint_respective_field:
        diff_dict.update({"inpaint_respective_field": inpaint_respective_field })
    elif default["inpaint_mask_upload_checkbox"] != inpaint_mask_upload_checkbox:
        diff_dict.update({"inpaint_mask_upload_checkbox": inpaint_mask_upload_checkbox })
    elif default["invert_mask_checkbox"] != invert_mask_checkbox:
        diff_dict.update({"invert_mask_checkbox": invert_mask_checkbox })
    elif default["inpaint_erode_or_dilate"] != inpaint_erode_or_dilate:
        diff_dict.update({"inpaint_erode_or_dilate": inpaint_erode_or_dilate })

    return diff_dict
