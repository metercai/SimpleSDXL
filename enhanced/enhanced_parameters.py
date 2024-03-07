backfill_prompt, translation_timing, translation_methods = [None] * 3

def set_all_enhanced_parameters(*args):
    global backfill_prompt, translation_timing, translation_methods

    backfill_prompt, translation_timing, translation_methods = args

    return
