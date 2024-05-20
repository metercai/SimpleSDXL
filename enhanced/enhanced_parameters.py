backfill_prompt, translation_methods, backend_selection, sd3_aspect_ratios_selection, hydit_aspect_ratios_selection = [None] * 5

def set_all_enhanced_parameters(*args):
    global backfill_prompt, translation_methods, backend_selection, sd3_aspect_ratios_selection, hydit_aspect_ratios_selection

    backfill_prompt, translation_methods, backend_selection, sd3_aspect_ratios_selection, hydit_aspect_ratios_selection = args

    return
