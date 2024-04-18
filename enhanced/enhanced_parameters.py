backfill_prompt, translation_timing, translation_methods, super_prompter, super_prompter_prompt = [None] * 5

def set_all_enhanced_parameters(*args):
    global backfill_prompt, translation_timing, translation_methods, super_prompter, super_prompter_prompt

    backfill_prompt, translation_timing, translation_methods, super_prompter, super_prompter_prompt = args

    return
