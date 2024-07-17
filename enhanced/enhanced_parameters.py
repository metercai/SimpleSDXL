backfill_prompt, translation_methods, backend_selection, comfyd_active_checkbox = [None] * 4

def set_all_enhanced_parameters(*args):
    global backfill_prompt, translation_methods, backend_selection, comfyd_active_checkbox

    backfill_prompt, translation_methods, backend_selection, comfyd_active_checkbox = args

    return
