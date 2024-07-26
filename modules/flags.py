import math
from enum import IntEnum, Enum


disabled = 'Disabled'
enabled = 'Enabled'
subtle_variation = 'Vary (Subtle)'
strong_variation = 'Vary (Strong)'
upscale_15 = 'Upscale (1.5x)'
upscale_2 = 'Upscale (2x)'
upscale_fast = 'Upscale (Fast 2x)'

uov_list = [disabled, subtle_variation, strong_variation, upscale_15, upscale_2, upscale_fast]

enhancement_uov_before = "Before First Enhancement"
enhancement_uov_after = "After Last Enhancement"
enhancement_uov_processing_order = [enhancement_uov_before, enhancement_uov_after]

enhancement_uov_prompt_type_original = 'Original Prompts'
enhancement_uov_prompt_type_last_filled = 'Last Filled Enhancement Prompts'
enhancement_uov_prompt_types = [enhancement_uov_prompt_type_original, enhancement_uov_prompt_type_last_filled]

CIVITAI_NO_KARRAS = ["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]

# fooocus: a1111 (Civitai)
KSAMPLER = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "",
    "dpmpp_3m_sde_gpu": "",
    "ddpm": "",
    "lcm": "LCM",
    "tcd": "TCD",
    "restart": "Restart"
}

SAMPLER_EXTRA = {
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": ""
}

SAMPLERS = KSAMPLER | SAMPLER_EXTRA

KSAMPLER_NAMES = list(KSAMPLER.keys())

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "lcm", "turbo", "align_your_steps", "tcd", "edm_playground_v2.5"]
SAMPLER_NAMES = KSAMPLER_NAMES + list(SAMPLER_EXTRA.keys())

sampler_list = SAMPLER_NAMES
scheduler_list = SCHEDULER_NAMES

clip_skip_max = 12

default_vae = 'Default (model)'

refiner_swap_method = 'joint'

cn_ip = "ImagePrompt"
cn_ip_face = "FaceSwap"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"

ip_list = [cn_ip, cn_canny, cn_cpds, cn_ip_face]
default_ip = cn_ip

default_parameters = {
    cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0)
}  # stop, weight

output_formats = ['png', 'jpeg', 'webp']

inpaint_mask_models = ['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime', 'sam']
inpaint_mask_cloth_category = ['full', 'upper', 'lower']
inpaint_mask_sam_model = ['vit_b', 'vit_l', 'vit_h']

inpaint_engine_versions = ['None', 'v2.5', 'v2.6']
inpaint_option_default = 'Inpaint or Outpaint (default)'
inpaint_option_detail = 'Improve Detail (face, hand, eyes, etc.)'
inpaint_option_modify = 'Modify Content (add objects, change background, etc.)'
inpaint_options = [inpaint_option_default, inpaint_option_detail, inpaint_option_modify]

desc_type_photo = 'Photograph'
desc_type_anime = 'Art/Anime'

translation_methods = ['Slim Model', 'Big Model', 'Third APIs']

COMFY_KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis"]   
comfy_scheduler_list = COMFY_SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
comfy_sampler_list = COMFY_SAMPLER_NAMES = COMFY_KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

aspect_ratios_templates = ['SDXL', 'HyDiT', 'Common']
default_aspect_ratio = ['1152*896', '1024*1024', '1280*1024']
available_aspect_ratios = [
    ['704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
    '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
    '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
    '1664*576', '1728*576'],

    ['768*1280', '960*1280', '1024*1024',
    '1280*768', '1280*960', '1280*1280',],
    
    ['576*1344', '768*1152', '896*1152', '768*1280', '960*1280',
    '1024*1024', '1024*1280', '1280*1280', '1280*1024',
    '1280*960', '1280*768', '1152*896', '1152*768', '1344*576']
]

def add_ratio(x):
    a, b = x.replace('*', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    c, d = a // g, b // g
    if (a, b) == (576, 1344):
        c, d = 9, 21
    elif (a, b) == (1344, 576):
        c, d = 21, 9
    elif (a, b) == (768, 1280):
        c, d = 9, 15
    elif (a, b) == (1280, 768):
        c, d = 15, 9
    return f'{a}×{b} <span style="color: grey;"> \U00002223 {c}:{d}</span>'

default_aspect_ratios = {
        aspect_ratios_templates[0]: add_ratio(default_aspect_ratio[0]),
        aspect_ratios_templates[1]: add_ratio(default_aspect_ratio[1]),
        aspect_ratios_templates[2]: add_ratio(default_aspect_ratio[2]),
        }
available_aspect_ratios_list = {
        aspect_ratios_templates[0]: [add_ratio(x) for x in available_aspect_ratios[0]],
        aspect_ratios_templates[1]: [add_ratio(x) for x in available_aspect_ratios[1]],
        aspect_ratios_templates[2]: [add_ratio(x) for x in available_aspect_ratios[2]],
        }

backend_engines = ['Fooocus', 'Comfy', 'Kolors', 'Kolors+', 'SD3m', 'HyDiT', 'HyDiT+']

language_radio = lambda x: '中文' if x=='cn' else 'En'

task_class_mapping = {
            'Fooocus': 'SDXL-Fooocus',
            'Comfy'  : 'SDXL-Comfy',
            'Kolors' : 'Kwai-Kolors',
            'Kolors+' : 'Kwai-Kolors+',
            'SD3m'   : 'SD3-medium',
            'HyDiT'  : 'Hunyuan-DiT',
            'HyDiT+'  : 'Hunyuan-DiT+',
            }
def get_taskclass_by_fullname(fullname):
    for taskclass, fname in task_class_mapping.items():
        if fname == fullname:
            return taskclass
    return None

comfy_classes = ['Comfy', 'Kolors', 'Kolors+', 'SD3m', 'HyDiT+']

default_class_params = {
    'Fooocus': {
        'disvisible': [],
        'disinteractive': [],
        'available_aspect_ratios_selection': 'SDXL',
        'available_sampler_name': sampler_list,
        'available_scheduler_name': scheduler_list,
        'backend_params': {},
        },
    'Comfy': {
        },
    'Kolors': {
        'disvisible': ["backend_selection", "performance_selection"],
        'disinteractive': ["input_image_checkbox", "enhance_checkbox", "performance_selection", "base_model", "overwrite_step", "refiner_model"],
        'available_aspect_ratios_selection': 'Common',
        'available_sampler_name': comfy_sampler_list,
        'available_scheduler_name': comfy_scheduler_list,
        'backend_params': {
            "task_method": "kolors_text2image1",
            "llms_model": "quant8",
            },
        },
    'Kolors+': {
        'disvisible': ["backend_selection", "performance_selection"],
        'disinteractive': ["input_image_checkbox", "enhance_checkbox", "performance_selection", "base_model", "overwrite_step", "refiner_model"],
        'available_aspect_ratios_selection': 'Common',
        'available_sampler_name': comfy_sampler_list,
        'available_scheduler_name': comfy_scheduler_list,
        'backend_params': {
            "task_method": "kolors_text2image2",
            "llms_model": "quant8",
            },
        },
    'SD3m': {
        'disvisible': ["backend_selection", "performance_selection"],
        'disinteractive': ["input_image_checkbox", "enhance_checkbox", "performance_selection", "loras", "refiner_model"],
        'available_aspect_ratios_selection': 'Common',
        'available_sampler_name': comfy_sampler_list,
        'available_scheduler_name': comfy_scheduler_list,
        'backend_params': {
            "task_method": "sd3_base",
            },
        },
    'HyDiT': {
        'disvisible': ["backend_selection", "performance_selection"],
        'disinteractive': ["input_image_checkbox", "enhance_checkbox", "performance_selection", "base_model", "loras", "refiner_model", "scheduler_name"],
        'available_aspect_ratios_selection': 'HyDiT',
        'available_sampler_name': ["ddpm", "ddim", "dpmms"],
        'backend_params': {
            "task_method": "hydit_base",
            },
        },
    'HyDiT+': {
        'disvisible': ["backend_selection", "performance_selection"],
        'disinteractive': ["input_image_checkbox", "enhance_checkbox", "performance_selection", "base_model", "loras", "refiner_model", "scheduler_name"],
        'available_aspect_ratios_selection': 'HyDiT',
        'available_sampler_name': comfy_sampler_list,
        'available_scheduler_name': comfy_scheduler_list,
        'backend_params': {
            "task_method": "hydit_base",
            },
        }
    }


class MetadataScheme(Enum):
    FOOOCUS = 'fooocus'
    A1111 = 'a1111'
    SIMPLE = 'simple'


metadata_scheme = [
    (f'{MetadataScheme.SIMPLE.value}', MetadataScheme.SIMPLE.value),
    (f'{MetadataScheme.FOOOCUS.value}', MetadataScheme.FOOOCUS.value),
    (f'{MetadataScheme.A1111.value}', MetadataScheme.A1111.value),
]

controlnet_image_count = 4


class OutputFormat(Enum):
    PNG = 'png'
    JPEG = 'jpeg'
    WEBP = 'webp'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: c.value, cls))


class PerformanceLoRA(Enum):
    QUALITY = None
    SPEED = None
    EXTREME_SPEED = 'sdxl_lcm_lora.safetensors'
    LIGHTNING = 'sdxl_lightning_4step_lora.safetensors'
    HYPER_SD = 'sdxl_hyper_sd_4step_lora.safetensors'


class Steps(IntEnum):
    QUALITY = 60
    SPEED = 30
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4

    @classmethod
    def keys(cls) -> list:
        return list(map(lambda c: c, Steps.__members__))


class StepsUOV(IntEnum):
    QUALITY = 36
    SPEED = 18
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4


class Performance(Enum):
    QUALITY = 'Quality'
    SPEED = 'Speed'
    EXTREME_SPEED = 'Extreme Speed'
    LIGHTNING = 'Lightning'
    HYPER_SD = 'Hyper-SD'

    @classmethod
    def list(cls) -> list:
        item = list(map(lambda c: c.value, cls))
        item.remove('Extreme Speed')
        return item

    @classmethod
    def values(cls) -> list:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def by_steps(cls, steps: int | str):
        return cls[Steps(int(steps)).name]

    @classmethod
    def has_restricted_features(cls, x) -> bool:
        if isinstance(x, Performance):
            x = x.value
        return x in [cls.EXTREME_SPEED.value, cls.LIGHTNING.value, cls.HYPER_SD.value]
        #return x in [cls.LIGHTNING.value, cls.HYPER_SD.value]

    def steps(self) -> int | None:
        return Steps[self.name].value if self.name in Steps.__members__ else None

    def steps_uov(self) -> int | None:
        return StepsUOV[self.name].value if self.name in StepsUOV.__members__ else None

    def lora_filename(self) -> str | None:
        return PerformanceLoRA[self.name].value if self.name in PerformanceLoRA.__members__ else None
