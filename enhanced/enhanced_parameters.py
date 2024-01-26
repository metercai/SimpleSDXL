backfill_prompt, embed_metadata_checkbox, translation_timing, translation_methods = [None] * 4

def set_all_enhanced_parameters(*args):
    global backfill_prompt, embed_metadata_checkbox, translation_timing, translation_methods

    backfill_prompt, embed_metadata_checkbox, translation_timing, translation_methods = args

    return
