import json
from os.path import exists
from custom.OneButtonPrompt.shared import path_manager
from custom.OneButtonPrompt.utils import path_fixed

DEFAULT_SETTINGS = {
    "advanced_mode": False,
    "image_number": 1,
    "seed_random": True,
    "seed": 0,
    "style": "Style: sai-cinematic",
    "prompt": "",
    "negative_prompt": "",
    "performance": "Speed",
    "resolution": "1152x896 (4:3)",
    "base_model": path_manager.default_model_names["default_base_model_name"],
    "lora_1_model": path_manager.default_model_names["default_lora_name"],
    "lora_1_weight": path_manager.default_model_names["default_lora_weight"],
    "lora_2_model": "None",
    "lora_2_weight": path_manager.default_model_names["default_lora_weight"],
    "lora_3_model": "None",
    "lora_3_weight": path_manager.default_model_names["default_lora_weight"],
    "lora_4_model": "None",
    "lora_4_weight": path_manager.default_model_names["default_lora_weight"],
    "lora_5_model": "None",
    "lora_5_weight": path_manager.default_model_names["default_lora_weight"],
    "theme": "None",
    "auto_negative_prompt": False,
    "OBP_preset": "Standard",
    "hint_chance": 25,
}


def load_settings():
    if exists(path_fixed("settings/settings.json")):
        with open(path_fixed("settings/settings.json")) as f:
            settings = json.load(f)
    else:
        settings = {}

    # Add any missing default settings
    changed = False
    for key, value in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings[key] = value
            changed = True

    if changed:
        with open(path_fixed("settings/settings.json"), "w") as f:
            json.dump(settings, f, indent=2)

    return settings


default_settings = load_settings()
