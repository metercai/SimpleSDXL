import math
from enum import IntEnum, Enum


disabled = 'Disabled'
enabled = 'Enabled'
subtle_variation = 'Vary (Subtle)'
strong_variation = 'Vary (Strong)'
upscale_15 = 'Upscale (1.5x)'
upscale_2 = 'Upscale (2x)'
upscale_fast = 'Upscale (Fast 2x)'
hires_fix = 'Vary (Hires.fix)'

uov_list = [disabled, subtle_variation, strong_variation, upscale_15, upscale_2, upscale_fast]
uov_list_flux = [disabled, subtle_variation, strong_variation, hires_fix, upscale_15, upscale_2, upscale_fast]

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

default_input_image_tab = 'ip_tab'
input_image_tab_ids = [ 'ip_tab', 'uov_tab', 'inpaint_tab', 'layer_tab', 'enhance_tab']

cn_ip = "ImagePrompt"
cn_ip_face = "FaceSwap"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"
cn_pose = "OpenPose"

ip_list = [cn_ip, cn_canny, cn_cpds, cn_ip_face, cn_pose]
default_ip = cn_ip

default_parameters = {
    cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0), cn_pose: (0.5, 1.0)
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

describe_type_photo = 'Photograph'
describe_type_anime = 'Art/Anime'
describe_types = [describe_type_photo, describe_type_anime]

scene_themes = ["New Year's graffiti"] #, "Fireworks", "Year of the Snake Blessings"]
scene_prompts = {
    "New Year's graffiti": "Text titled \"Happy New Year\",\nDoodle, graffiti style Keith Haring, cute, marker pen illustration, MBE illustration, stars, moon, bold lines, grunge aesthetic style, mixed pattern, text and emoji installation, ",
    "新年涂鸦(中)": "标题文本为\"Happy New Year\",\nKeith Haring的涂鸦风格，可爱，马克笔插图，MBE插图，星星，月亮，粗线条，垃圾美学风格，混合的图案，包含文字和表情符,",
    } #"Fireworks": "",
    #"Year of the Snake Blessings": "" }

scene_aspect_ratios = ["Vertical|9:16", "Portrait|4:5", "Photo|4:3", "Landscape|3:2", "Widescreen|16:9", "Cinematic|21:9"]
scene_aspect_ratios_size = {
    "Vertical|9:16": '576×1024',
    "Portrait|4:5": '864×1080', 
    "Photo|4:3": '1024×768', 
    "Landscape|3:2": '1080×720', 
    "Widescreen|16:9": '1024×576', 
    "Cinematic|21:9": '1260×540' }

translation_methods = ['Slim Model', 'Big Model', 'Third APIs']

COMFY_KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis"]   
comfy_scheduler_list = COMFY_SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]
comfy_sampler_list = COMFY_SAMPLER_NAMES = COMFY_KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

aspect_ratios_templates = ['SDXL', 'HyDiT', 'Common', 'Flux']
default_aspect_ratio = ['1152*896', '1024*1024', '1280*768', '1280*720']
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
    '1280*960', '1280*768', '1152*896', '1152*768', '1344*576'],

    ['576*1344', '768*1152', '896*1152', '720*1280', '768*1280', '960*1280',
    '1024*1024', '1024*1280', '1280*1280', '1280*1024',
    '1280*960', '1280*768', '1280*720', '1152*896', '1152*768', '1344*576']
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
    template: add_ratio(ratio)
    for template, ratio in zip(aspect_ratios_templates, default_aspect_ratio)
}

available_aspect_ratios_list = {
    template: [add_ratio(x) for x in ratios]
    for template, ratios in zip(aspect_ratios_templates, available_aspect_ratios)
}


backend_engines = ['Fooocus', 'Comfy', 'Kolors', 'SD3x', 'HyDiT', 'Flux']

model_file_filter = {
        'SD3x'   : ['sd3'],
        'Flux'   : [['flux'], ['f.1']],
        'HyDiT'  : ['hunyuan'],
        'Kolors' : ['kolors'],
        }
model_file_filter['Fooocus'] = model_file_filter['SD3x'] + model_file_filter['Flux'] + model_file_filter['HyDiT']

language_radio = lambda x: '中文' if x=='cn' else 'En'

task_class_mapping = {
            'Fooocus': 'SDXL-Fooocus',
            'Comfy'  : 'SDXL-Comfy',
            'Kolors' : 'Kwai-Kolors',
            'SD3x'   : 'SD3m-SD3.5x',
            'HyDiT'  : 'Hunyuan-DiT',
            'Flux'   : 'Flux.1',
            }
def get_taskclass_by_fullname(fullname):
    for taskclass, fname in task_class_mapping.items():
        if fname == fullname:
            return taskclass
    return None

comfy_classes = ['Comfy', 'Kolors', 'SD3x', 'HyDiT', 'Flux']

default_class_params = {
    'Fooocus': {
        'disvisible': [],
        'disinteractive': [],
        'available_aspect_ratios_selection': 'SDXL',
        'available_sampler_name': sampler_list,
        'available_scheduler_name': scheduler_list,
        'available_uov_method': uov_list,
        'backend_params': {},
        },
    'Comfy': {
        'disvisible': [],
        'disinteractive': [],
        'available_aspect_ratios_selection': 'SDXL',
        'available_sampler_name': comfy_sampler_list,
        'available_scheduler_name': comfy_scheduler_list,
        'backend_params': {},
        },
    'Kolors': {
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
    'SD3x': {
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
        'available_sampler_name': comfy_sampler_list,
        'available_scheduler_name': comfy_scheduler_list,
        'backend_params': {
            "task_method": "hydit_base",
            },
        },
    'Flux': {
        'disvisible': ["backend_selection", "performance_selection"],
        'disinteractive': ["input_image_checkbox", "enhance_checkbox", "performance_selection", "loras-4", "refiner_model"],
        'available_aspect_ratios_selection': 'Flux',
        'available_sampler_name': comfy_sampler_list,
        'available_scheduler_name': comfy_scheduler_list,
        'available_uov_method': uov_list_flux,
        'backend_params': {
            "task_method": "flux_base",
            "clip_model": "auto",
            "base_model_dtype": "auto",
            },
        },
    }

get_engine_default_params = lambda x: default_class_params['Fooocus'] if x not in default_class_params else default_class_params[x]
get_engine_default_backend_params = lambda x: get_engine_default_params(x).get('backend_params', default_class_params['Fooocus']['backend_params'])

class MetadataScheme(Enum):
    FOOOCUS = 'fooocus'
    A1111 = 'a1111'
    SIMPLE = 'simple'


metadata_scheme = [
    (f'{MetadataScheme.SIMPLE.value}', MetadataScheme.SIMPLE.value),
    #(f'{MetadataScheme.FOOOCUS.value}', MetadataScheme.FOOOCUS.value),
    (f'{MetadataScheme.A1111.value}', MetadataScheme.A1111.value),
]


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
    HYPER_SD = 'Hyper-SDXL-8steps-lora.safetensors' #'sdxl_hyper_sd_4step_lora.safetensors'


class Steps(IntEnum):
    QUALITY = 60
    SPEED = 30
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 8

    @classmethod
    def keys(cls) -> list:
        return list(map(lambda c: c, Steps.__members__))


class StepsUOV(IntEnum):
    QUALITY = 36
    SPEED = 18
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 8


class Performance(Enum):
    QUALITY = 'Quality'
    SPEED = 'Speed'
    EXTREME_SPEED = 'Extreme Speed'
    HYPER_SD = 'Hyper-SD'
    LIGHTNING = 'Lightning'

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

areacode = [
    "86-CN-中国",
    "1-US-United States",
    "7-KZ-Kazakhstan",
    "7-RU-Russia",
    "20-EG-Egypt",
    "27-ZA-South Africa",
    "30-GR-Greece",
    "31-NL-Netherlands",
    "32-BE-Belgium",
    "33-FR-France",
    "34-ES-Spain",
    "36-HU-Hungary",
    "39-IT-Italy",
    "40-RO-Romania",
    "41-CH-Switzerland",
    "43-AT-Austria",
    "44-GB-United Kingdom",
    "45-DK-Denmark",
    "46-SE-Sweden",
    "47-NO-Norway",
    "48-PL-Poland",
    "49-DE-Germany",
    "51-PE-Peru",
    "52-MX-Mexico",
    "53-CU-Cuba",
    "54-AR-Argentina Republic",
    "55-BR-Brazil",
    "56-CL-Chile",
    "57-CO-Colombia",
    "58-VE-Venezuela",
    "60-MY-Malaysia",
    "61-AU-Australia",
    "62-ID-Indonesia",
    "63-PH-Philippines",
    "64-NZ-New Zealand",
    "65-SG-Singapore",
    "66-TH-Thailand",
    "81-JP-Japan",
    "82-KR-Korea S, Republic of",
    "84-VN-Vietnam",
    "90-TR-Turkey",
    "91-IN-India",
    "92-PK-Pakistan",
    "93-AF-Afghanistan",
    "94-LK-Sri Lanka",
    "95-MM-Myanmar (Burma)",
    "98-IR-Iran",
    "211-SS-South Sudan (Republic of)",
    "212-MA-Morocco",
    "213-DZ-Algeria",
    "216-TN-Tunisia",
    "218-LY-Libya",
    "220-GM-Gambia",
    "221-SN-Senegal",
    "222-MR-Mauritania",
    "223-ML-Mali",
    "224-GN-Guinea",
    "225-CI-Ivory Coast",
    "226-BF-Burkina Faso",
    "227-NE-Niger",
    "228-TG-Togo",
    "229-BJ-Benin",
    "230-MU-Mauritius",
    "231-LR-Liberia",
    "232-SL-Sierra Leone",
    "233-GH-Ghana",
    "234-NG-Nigeria",
    "235-TD-Chad",
    "236-CF-Central African Rep.",
    "237-CM-Cameroon",
    "238-CV-Cape Verde",
    "239-ST-Sao Tome & Principe",
    "240-GQ-Equatorial Guinea",
    "241-GA-Gabon",
    "242-CG-Congo, Republic",
    "243-CD-Congo, Dem. Rep.",
    "244-AO-Angola",
    "245-GW-Guinea-Bissau",
    "247-AC-Ascension Island",
    "248-SC-Seychelles",
    "249-SD-Sudan",
    "250-RW-Rwanda",
    "251-ET-Ethiopia",
    "252-SO-Somalia",
    "253-DJ-Djibouti",
    "254-KE-Kenya",
    "255-TZ-Tanzania",
    "256-UG-Uganda",
    "257-BI-Burundi",
    "258-MZ-Mozambique",
    "260-ZM-Zambia",
    "261-MG-Madagascar",
    "262-RE-Reunion",
    "263-ZW-Zimbabwe",
    "264-NA-Namibia",
    "265-MW-Malawi",
    "266-LS-Lesotho",
    "267-BW-Botswana",
    "268-SZ-Swaziland",
    "269-KM-Comoros",
    "297-AW-Aruba",
    "298-FO-Faroe Islands",
    "299-GL-Greenland",
    "350-GI-Gibraltar",
    "351-PT-Portugal",
    "352-LU-Luxembourg",
    "353-IE-Ireland",
    "354-IS-Iceland",
    "355-AL-Albania",
    "356-MT-Malta",
    "357-CY-Cyprus",
    "358-FI-Finland",
    "359-BG-Bulgaria",
    "370-LT-Lithuania",
    "371-LV-Latvia",
    "372-EE-Estonia",
    "373-MD-Moldova",
    "374-AM-Armenia",
    "375-BY-Belarus",
    "376-AD-Andorra",
    "377-MC-Monaco",
    "378-SM-San Marino",
    "380-UA-Ukraine",
    "381-RS-Serbia",
    "382-ME-Montenegro",
    "385-HR-Croatia",
    "386-SI-Slovenia",
    "387-BA-Bosnia & Herzegov.",
    "389-MK-Macedonia",
    "420-CZ-Czech Rep.",
    "421-SK-Slovakia",
    "423-LI-Liechtenstein",
    "501-BZ-Belize",
    "502-GT-Guatemala",
    "503-SV-El Salvador",
    "504-HN-Honduras",
    "505-NI-Nicaragua",
    "506-CR-Costa Rica",
    "507-PA-Panama",
    "508-PM-Saint Pierre and Miquelon",
    "509-HT-Haiti",
    "590-GP-Guadeloupe",
    "591-BO-Bolivia",
    "592-GY-Guyana",
    "593-EC-Ecuador",
    "594-FG-French Guiana",
    "595-PY-Paraguay",
    "596-MQ-Martinique (French Department of)",
    "597-SR-Suriname",
    "598-UY-Uruguay",
    "599-AN-Netherlands Antilles",
    "670-TP-Timor-Leste",
    "672-NF-Norfolk Island",
    "673-BN-Brunei Darussalam",
    "675-PG-Papua New Guinea",
    "676-TO-Tonga",
    "677-SB-Solomon Islands",
    "678-VU-Vanuatu",
    "679-FJ-Fiji",
    "680-PW-Palau (Republic of)",
    "682-CK-Cook Islands",
    "685-WS-Samoa",
    "686-KI-Kiribati",
    "687-NC-New Caledonia",
    "689-PF-French Polynesia",
    "691-FM-Micronesia",
    "850-KP-Korea N. Dem.Peoples Rep.",
    "852-HK-Hong Kong, China",
    "853-MO-Macao, China",
    "855-KH-Cambodia",
    "856-LA-Laos P.D.R.",
    "880-BD-Bangladesh",
    "886-TW-Taiwan, Province of China",
    "960-MV-Maldives",
    "961-LB-Lebanon",
    "962-JO-Jordan",
    "963-SY-Syrian Arab Republic",
    "964-IQ-Iraq",
    "965-KW-Kuwait",
    "966-SA-Saudi Arabia",
    "967-YE-Yemen",
    "968-OM-Oman",
    "970-PS-Palestinian Territory",
    "971-AE-United Arab Emirates",
    "972-IL-Israel",
    "973-BH-Bahrain",
    "974-QA-Qatar",
    "975-BT-Bhutan",
    "976-MN-Mongolia",
    "977-NP-Nepal",
    "992-TK-Tajikistan",
    "993-TM-Turkmenistan",
    "994-AZ-Azerbaijan",
    "995-GE-Georgia",
    "996-KG-Kyrgyzstan",
    "998-UZ-Uzbekistan",
    "1242-BS-Bahamas",
    "1246-BB-Barbados",
    "1264-AI-Anguilla",
    "1268-AG-Antigua and Barbuda",
    "1284-VG-British Virgin Islands",
    "1340-VI-US Virgin Islands",
    "1345-KY-Cayman Islands",
    "1441-BM-Bermuda",
    "1473-GD-Grenada",
    "1649-TC-Turks and Caicos Islands",
    "1664-MS-Montserrat",
    "1670-MP-Northern Mariana Islands",
    "1671-GU-Guam",
    "1684-AS-American Samoa",
    "1758-LC-Saint Lucia",
    "1767-DM-Dominica",
    "1784-VC-St. Vincent & Gren.",
    "1849-DO-Dominican Republic",
    "1868-TT-Trinidad and Tobago",
    "1869-KN-Saint Kitts and Nevis",
    "1876-JM-Jamaica",
    "1939-PR-Puerto Rico",
    "-CN-(旧身份)"
    ]
