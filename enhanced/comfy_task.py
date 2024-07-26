import os
import zipfile
import shutil
import modules.config
from enhanced.simpleai import ComfyTaskParams, models_info, modelsinfo, sysinfo, refresh_models_info
from modules.model_loader import load_file_from_url

default_method_names = ['Blending given FG and IC-light', 'Generate foreground with Conv Injection']
default_method_list = {
    default_method_names[0]: 'iclight_fc',
    default_method_names[1]: 'layerdiffuse_fg',
}

iclight_source_names = ['Top -  Left', 'Top - Light', 'Top - Right', 'Left  Light', 'CenterLight', 'Right Light', 'Bottom Left', 'BottomLight', 'BottomRight']
iclight_source_text = {
    iclight_source_names[0]: "Top Left Light",
    iclight_source_names[1]: "Top Light",
    iclight_source_names[2]: "Top Right Light",
    iclight_source_names[3]: "Left Light",
    iclight_source_names[5]: "Right Light",
    iclight_source_names[6]: "Bottom Left Light",
    iclight_source_names[7]: "Bottom Light",
    iclight_source_names[8]: "Bottom Right Light",
    }

default_base_SD15_name = 'realisticVisionV60B1_v51VAE.safetensors'

default_base_SD3m_name_list = ['sd3_medium_incl_clips.safetensors', 'sd3_medium_incl_clips_t5xxlfp8.safetensors', 'sd3_medium_incl_clips_t5xxlfp16.safetensors']

def get_default_base_SD3m_name():
    for sd3name in default_base_SD3m_name_list:
        if f'checkpoints/{sd3name}' in models_info:
            return sd3name
    return default_base_SD3m_name_list[0]


quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm'
]
quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]


class ComfyTask:

    def __init__(self, name, params, images=None):
        self.name = name
        self.params = params
        self.images = images


def get_comfy_task(task_name, task_method, default_params, input_images, options={}):
    global defaul_method_names, default_method_list

    if task_name == 'default':
        if task_method == default_method_names[1]:
            comfy_params = ComfyTaskParams(default_params)
            comfy_params.update_params({"layer_diffuse_injection": "SDXL, Conv Injection"})
            return ComfyTask(default_method_list[task_method], comfy_params)
        else:
            comfy_params = ComfyTaskParams(default_params)
            if input_images is None:
                raise ValueError("input_images cannot be None for this method")
            images = {"input_image": input_images[0]}
            if 'iclight_enable' in options and options["iclight_enable"]:
                #if f'checkpoints/{default_base_SD15_name}' not in models_info:
                if modelsinfo.exists_model(catalog="checkpoints", model_path=default_base_SD15_name):
                    modules.config.downloading_base_sd15_model()
                comfy_params.update_params({"base_model": default_base_SD15_name})
                if options["iclight_source_radio"] == 'CenterLight':
                    comfy_params.update_params({"light_source_text_switch": False})
                else:
                    comfy_params.update_params({
                        "light_source_text_switch": True,
                        "light_source_text": iclight_source_text[options["iclight_source_radio"]]
                        })
                return ComfyTask(default_method_list[task_method], comfy_params, images)
            else:
                width, height = fixed_width_height(default_params["width"], default_params["height"], 64)
                comfy_params.update_params({
                    "layer_diffuse_cond": "SDXL, Foreground",
                    "width": width,
                    "height": height,
                    })
                comfy_params.delete_params(['denoise'])
                return ComfyTask('layerdiffuse_cond', comfy_params, images)

    elif task_name == 'SD3m':
        comfy_params = ComfyTaskParams(default_params)
        if modelsinfo.exists_model(catalog="checkpoints", model_path=default_params["base_model"]):
        #if f'checkpoints/{default_params["base_model"]}' not in models_info:
            modules.config.downloading_sd3_medium_model()
        return ComfyTask(task_method, comfy_params)
    elif task_name == 'Kolors':
        comfy_params = ComfyTaskParams(default_params)
        if 'llms_model' not in default_params or default_params['llms_model'] == 'auto':
            comfy_params.update_params({
                "llms_model": 'quant4' if sysinfo["gpu_memory"]<8180 else 'quant8' #'fp16'
                })
        check_download_kolors_model(modules.config.path_models_root)
        comfy_params.delete_params(['sampler'])
        return ComfyTask(task_method, comfy_params)
    elif task_name == 'Kolors+':
        comfy_params = ComfyTaskParams(default_params)
        if 'llms_model' not in default_params or default_params['llms_model'] == 'auto':
            comfy_params.update_params({
                "llms_model": 'quant4' if sysinfo["gpu_memory"]<8180 else 'quant8' #'fp16'
                })
        check_download_kolors_model(modules.config.path_models_root)
        return ComfyTask(task_method, comfy_params)
    elif task_name == 'HyDiT+':
        comfy_params = ComfyTaskParams(default_params)
        if modelsinfo.exists_model(catalog="checkpoints", model_path=default_params["base_model"]):
            modules.config.downloading_hydit_model()
        return ComfyTask(task_method, comfy_params)



def fixed_width_height(width, height, factor): 
    fixed_width = int(((height // factor + 1) * factor * width)/height)
    fixed_width = fixed_width if fixed_width % factor == 0 else int((fixed_width // factor + 1) * factor )
    width = width if height % factor == 0 else fixed_width
    height = height if height % factor == 0 else int((height // factor + 1) * factor)
    return width, height

default_kolors_base_model_name = 'kolors_unet_fp16.safetensors'

kolors_scheduler_list = [ "EulerDiscreteScheduler",
                          "EulerAncestralDiscreteScheduler",
                          "DPMSolverMultistepScheduler",
                          "DPMSolverMultistepScheduler_SDE_karras",
                          "UniPCMultistepScheduler",
                          "DEISMultistepScheduler" ]
default_kolors_scheduler = kolors_scheduler_list[0]

def check_task_model():
    pass

def check_download_kolors_model(path_root):
    check_modle_file = [
            "diffusers/Kolors/text_encoder/pytorch_model-00007-of-00007.bin",
            "unet/kolors_unet_fp16.safetensors",
            "vae/sdxl_fp16.vae.safetensors",
            ]
    path_temp = os.path.join(path_root, 'temp')
    if not os.path.exists(path_temp):
        os.makedirs(path_temp)
    exists_kolors_model_path = False
    for f in models_info:
        if '00007-of-00007.bin' in f and f.startswith('diffusers/Kolors'):
            exists_kolors_model_path = True
    if not exists_kolors_model_path:
        load_file_from_url(
            url='https://huggingface.co/metercai/SimpleSDXL2/resolve/main/models_kolors_simpleai_diffusers_fp16.zip',
            model_dir=path_temp,
            file_name='models_kolors_simpleai_diffusers_fp16.zip'
        )
        downfile = os.path.join(path_temp, 'models_kolors_simpleai_diffusers_fp16.zip')
        with zipfile.ZipFile(downfile, 'r') as zipf:
            print(f'extractall: {downfile}')
            zipf.extractall(path_temp)
        shutil.move(os.path.join(path_temp, 'models/diffusers/Kolors'), modules.config.paths_diffusers[0])
        shutil.rmtree(os.path.join(path_temp, 'models'))
        os.remove(downfile)
    
    if check_modle_file[1] not in models_info:
        path_org = os.path.join(modules.config.paths_diffusers[0], 'Kolors/unet/diffusion_pytorch_model.fp16.safetensors')
        path_dst = os.path.join(modules.config.path_unet, 'kolors_unet_fp16.safetensors')
        print(f'model file copy: {path_org} to {path_dst}')
        shutil.copy(path_org, path_dst)

    if check_modle_file[2] not in models_info:
        path_org = os.path.join(modules.config.paths_diffusers[0], 'Kolors/vae/diffusion_pytorch_model.fp16.safetensors')
        path_dst = os.path.join(modules.config.path_vae, 'sdxl_fp16.vae.safetensors')
        print(f'model file copy: {path_org} to {path_dst}')
        shutil.copy(path_org, path_dst)
   
    refresh_models_info()    
    return

