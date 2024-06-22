backfill_prompt, translation_methods, backend_selection, sd3_aspect_ratios_selection, hydit_aspect_ratios_selection, comfyd_active_checkbox = [None] * 6

def set_all_enhanced_parameters(*args):
    global backfill_prompt, translation_methods, backend_selection, sd3_aspect_ratios_selection, hydit_aspect_ratios_selection, comfyd_active_checkbox

    backfill_prompt, translation_methods, backend_selection, sd3_aspect_ratios_selection, hydit_aspect_ratios_selection, comfyd_active_checkbox = args

    return
