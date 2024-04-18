backfill_prompt, translation_timing, translation_methods, super_prompter, super_prompter_prompt, backend_selection, sd3_aspect_ratios_selection = [None] * 7

def set_all_enhanced_parameters(*args):
    global backfill_prompt, translation_timing, translation_methods, super_prompter, super_prompter_prompt, backend_selection, sd3_aspect_ratios_selection

    backfill_prompt, translation_timing, translation_methods, super_prompter, super_prompter_prompt, backend_selection, sd3_aspect_ratios_selection = args

    return
