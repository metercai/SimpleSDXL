backfill_prompt, translation_methods, comfyd_active_checkbox = [None] * 3

def set_all_enhanced_parameters(*args):
    global backfill_prompt, translation_methods, comfyd_active_checkbox

    backfill_prompt, translation_methods, comfyd_active_checkbox = args

    return
