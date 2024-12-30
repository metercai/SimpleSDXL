import os
import pathlib
import zipfile
import shutil
import modules.config as config
from shared import sysinfo, modelsinfo
from enhanced.simpleai import ComfyTaskParams
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

RAM32G = 32500
RAM32G1 = 32768
RAM16G = 16300
VRAM8G = 8180
VRAM8G1 = 8192  # include 8G
VRAM16G = 16300

def is_lowlevel_device():
    return sysinfo["gpu_memory"]<VRAM8G

def is_highlevel_device():
    return sysinfo["gpu_memory"]>VRAM16G

default_base_SD15_name = 'realisticVisionV60B1_v51VAE.safetensors'
default_base_SD3m_name_list = ['sd3_medium_incl_clips.safetensors', 'sd3_medium_incl_clips_t5xxlfp8.safetensors', 'sd3_medium_incl_clips_t5xxlfp16.safetensors']

def get_default_base_SD3m_name():
    dtype = 0 if sysinfo["gpu_memory"]<VRAM8G and sysinfo["ram_total"]<RAM16G else 1 if sysinfo["gpu_memory"]<VRAM16G and sysinfo["ram_total"]<RAM32G else 2
    for i in range(dtype, -1 ,-1):
        sd3name = default_base_SD3m_name_list[i]
        if modelsinfo.exists_model_key(f'checkpoints/{sd3name}'):
            return sd3name
    return default_base_SD3m_name_list[0]

default_base_Flux_name_list = ['flux1-dev.safetensors', 'flux1-dev-bnb-nf4.safetensors', 'flux1-dev-bnb-nf4-v2.safetensors', 'flux-hyp8-Q5_K_M.gguf', 'flux1-schnell.safetensors', 'flux1-schnell-bnb-nf4.safetensors']
flux_model_urls = {
    "flux1-dev.safetensors": "https://huggingface.co/metercai/SimpleSDXL2/resolve/main/flux1/flux1-dev.safetensors",
    "flux1-dev-bnb-nf4-v2.safetensors": "https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4-v2.safetensors",
    "flux1-schnell.safetensors": "https://huggingface.co/metercai/SimpleSDXL2/resolve/main/flux1/flux1-schnell.safetensor",
    "flux1-schnell-bnb-nf4.safetensors": "https://huggingface.co/silveroxides/flux1-nf4-weights/resolve/main/flux1-schnell-bnb-nf4.safetensors",
    "flux-hyp8-Q5_K_M.gguf": "https://huggingface.co/mhnakif/flux-hyp8-gguf-k/resolve/main/flux-hyp8-Q5_K_M.gguf"
    }

def get_default_base_Flux_name(plus=False):
    if plus:
        if is_lowlevel_device():
            checklist = [default_base_Flux_name_list[5], default_base_Flux_name_list[3]]
        else:
            checklist = [default_base_Flux_name_list[4], default_base_Flux_name_list[5], default_base_Flux_name_list[3]]
    else:
        if is_highlevel_device():
            checklist = [default_base_Flux_name_list[0], default_base_Flux_name_list[2], default_base_Flux_name_list[1], default_base_Flux_name_list[3]]
        else:
            checklist = [default_base_Flux_name_list[2], default_base_Flux_name_list[1], default_base_Flux_name_list[3]]
    for i in range(0, len(checklist)):
        fluxname = checklist[i]
        if modelsinfo.exists_model(catalog="checkpoints", model_path=fluxname):
            return fluxname
    return checklist[0]
        

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

    def __init__(self, name, params, images=None, steps=None):
        self.name = name
        self.params = params
        self.images = images
        self.steps = steps


def get_comfy_task(user_did, task_name, task_method, default_params, input_images, options={}):
    global defaul_method_names, default_method_list

    #print(f'task_name:{task_name}, task_method:{task_method}')
    comfy_params = ComfyTaskParams(default_params, user_did)
    if task_name == 'default':
        if task_method == default_method_names[1]:
            comfy_params.update_params({"layer_diffuse_injection": "SDXL, Conv Injection"})
            return ComfyTask(default_method_list[task_method], comfy_params)
        else:
            if input_images is None or not  input_images.exists('input_image'):
                raise ValueError("input_images cannot be None for this method")
            if 'iclight_enable' in options and options["iclight_enable"]:
                if modelsinfo.exists_model(catalog="checkpoints", model_path=default_base_SD15_name):
                    config.downloading_base_sd15_model()
                comfy_params.update_params({"base_model": default_base_SD15_name})
                if options["iclight_source_radio"] == 'CenterLight':
                    comfy_params.update_params({"light_source_text_switch": False})
                else:
                    comfy_params.update_params({
                        "light_source_text_switch": True,
                        "light_source_text": iclight_source_text[options["iclight_source_radio"]]
                        })
                return ComfyTask(default_method_list[task_method], comfy_params, input_images)
            else:
                width, height = fixed_width_height(default_params["width"], default_params["height"], 64)
                comfy_params.update_params({
                    "layer_diffuse_cond": "SDXL, Foreground",
                    "width": width,
                    "height": height,
                    })
                comfy_params.delete_params(['denoise'])
                return ComfyTask('layerdiffuse_cond', comfy_params, input_images, 60)

    elif task_name == 'SD3x':
        if not modelsinfo.exists_model(catalog="checkpoints", model_path=default_params["base_model"]):
            config.downloading_sd3_medium_model()
        if 'base_model_dtype' in default_params:
            comfy_params.delete_params(['base_model_dtype'])
        return ComfyTask(task_method, comfy_params)
    
    elif task_name in ['Kolors']:
        if 'llms_model' not in default_params or default_params['llms_model'] == 'auto':
            comfy_params.update_params({
                "llms_model": 'quant4' if sysinfo["gpu_memory"]<VRAM8G else 'quant8' if sysinfo["gpu_memory"]<VRAM16G else 'fp16'
                })
        total_steps = default_params["steps"] if 'i2i_uov_tiled_steps' in default_params or ('i2i_inpaint_fn' in default_params and default_params['i2i_inpaint_fn'] == 1) else None
        check_download_kolors_model(config.path_models_root)
        return ComfyTask(task_method, comfy_params, input_images, total_steps)
    
    elif task_name in ['HyDiT']:
        if not modelsinfo.exists_model(catalog="checkpoints", model_path=default_params["base_model"]):
            config.downloading_hydit_model()
        return ComfyTask(task_method, comfy_params)
    
    elif task_name == 'Flux':
        base_model = default_params['base_model']
        clip_model = '' if 'clip_model' not in default_params else default_params['clip_model']
        if 'scene_' in task_method:
            return ComfyTask(task_method, comfy_params, input_images)

        if '_aio' in task_method:
            total_steps = default_params["steps"] if 'i2i_uov_tiled_steps' in default_params or ('i2i_inpaint_fn' in default_params and default_params['i2i_inpaint_fn'] == 1) else None
            check_download_flux_model(default_params["base_model"], default_params.get("clip_model", None))
            if 'base_model_gguf' in default_params:
                comfy_params.delete_params(['base_model'])
            if 'clip_model' not in default_params or default_params['clip_model'] == 'auto':
                clip_model = 't5xxl_fp16.safetensors' if sysinfo["gpu_memory"]>VRAM8G1 and sysinfo["ram_total"]>RAM32G1 and modelsinfo.exists_model("clip", 't5xxl_fp16.safetensors') else 't5xxl_fp8_e4m3fn.safetensors'
                comfy_params.update_params({"clip_model": clip_model})
            return ComfyTask(task_method, comfy_params, input_images, total_steps)
        else:
            if base_model == 'auto':
                model_dev = 'flux1-dev.safetensors'
                model_nf4 = 'flux1-dev-bnb-nf4-v2.safetensors'
                model_hyp8 = 'flux-hyp8-Q5_K_M.gguf'
                base_model = model_nf4 if sysinfo["gpu_memory"]<=VRAM8G1 else model_dev
                if not modelsinfo.exists_model(catalog="checkpoints", model_path=base_model) and modelsinfo.exists_model(catalog="checkpoints", model_path=model_hyp8):
                    base_model = model_hyp8
                    default_params['steps'] = 12
                default_params['base_model'] = base_model  
            base_model_key = f'checkpoints/{base_model}'
            if 'nf4' in base_model.lower() and 'bnb' in base_model.lower():
                if sysinfo["gpu_memory"]<VRAM8G:
                    task_method = 'flux_base_nf4_2'
                else:
                    task_method = 'flux_base_nf4'
                comfy_params.delete_params(['clip_model', 'base_model_dtype', 'lora_1', 'lora_1_strength'])
            elif 'fp8' in base_model.lower() and modelsinfo.exists_model_key(base_model_key)  and modelsinfo.get_model_key_info(base_model_key)["size"]/(1024*1024*1024)>15:
                task_method = 'flux_base_fp8'
                if 'lora_1' in default_params:
                    task_method = 'flux_base2_fp8'
                comfy_params.delete_params(['clip_model', 'base_model_dtype'])
            else:
                if 'clip_model' not in default_params or default_params['clip_model'] == 'auto':
                    clip_model = 't5xxl_fp16.safetensors' if sysinfo["gpu_memory"]>VRAM8G1 and sysinfo["ram_total"]>RAM32G1 else 't5xxl_fp8_e4m3fn.safetensors'
                    if not modelsinfo.exists_model("clip", clip_model):
                        if clip_model == 't5xxl_fp16.safetensors' and modelsinfo.exists_model("clip", 't5xxl_fp8_e4m3fn.safetensors'):
                            clip_model = 't5xxl_fp8_e4m3fn.safetensors'
                    comfy_params.update_params({"clip_model": clip_model})
                if 'base_model_dtype' not in default_params or default_params['base_model_dtype'] == 'auto':
                    comfy_params.update_params({
                        "base_model_dtype": 'fp8_e4m3fn' if sysinfo["gpu_memory"]<VRAM16G or 'fp8' in base_model.lower() or 'lora_1' in default_params else 'default' #'fp16'
                    })
                else:
                    base_model_dtype = default_params['base_model_dtype']
                    if base_model_dtype == 'fp16':
                        base_model_dtype = 'default'
                    elif base_model_dtype != 'default':
                        base_model_dtype = 'fp8_e4m3fn'

                    if base_model_dtype == 'default' and 'lora_1' in default_params:
                        base_model_dtype = 'fp8_e4m3fn'
                    comfy_params.update_params({"base_model_dtype": base_model_dtype})
                if 'lora_1' in default_params and '.gguf' not in base_model:
                    task_method = 'flux_base2'
                if '.gguf' in base_model:
                    task_method = 'flux_base_gguf'
                    if 'lora_1' in default_params:
                        task_method = 'flux_base2_gguf'
                    comfy_params.delete_params(['base_model_dtype'])
            check_download_flux_model(base_model, clip_model if clip_model!='auto' else None)
        return ComfyTask(task_method, comfy_params, input_images)
    elif task_name == 'SD15AIO' and '_aio' in task_method:
        total_steps = default_params["steps"] if 'display_steps' not in default_params else default_params["display_steps"]
        if 'display_steps' in default_params:
            comfy_params.delete_params(["display_steps"])
        return ComfyTask(task_method, comfy_params, input_images, total_steps)
    else:  # SeamlessTiled
        #check_download_base_model(default_params["base_model"])
        return ComfyTask(task_method, comfy_params, input_images)



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
    #check_model_files_from_download_of_preset_file
    pass

def check_download_kolors_model(path_root):
    check_model_file = [
            "diffusers/Kolors/text_encoder/pytorch_model-00007-of-00007.bin",
            ]
    path_temp = os.path.join(path_root, 'temp')
    if not os.path.exists(path_temp):
        os.makedirs(path_temp)
    if not modelsinfo.exists_model_key(check_model_file[0]):
        load_file_from_url(
            url='https://huggingface.co/metercai/SimpleSDXL2/resolve/main/models_kolors_fp16_simpleai_1210.zip',
            model_dir=path_temp,
            file_name='models_kolors_fp16_simpleai_1210.zip'
        )
        downfile = os.path.join(path_temp, 'models_kolors_fp16_simpleai_1210.zip')
        with zipfile.ZipFile(downfile, 'r') as zipf:
            print(f'extractall: {downfile}')
            zipf.extractall(path_root)
        shutil.move(os.path.join(path_temp, 'SimpleModels/diffusers/Kolors'), config.paths_diffusers[0])
        os.remove(downfile)
        shutil.rmtree(path_temp)
        path_dst = os.path.join(config.paths_diffusers[0], 'Kolors/unet/diffusion_pytorch_model.fp16.safetensors')
        pathlib.Path(path_dst).touch()
        path_dst = os.path.join(config.paths_diffusers[0], 'Kolors/vae/diffusion_pytorch_model.fp16.safetensors')
        pathlib.Path(path_dst).touch()
   
    modelsinfo.refresh_from_path()  
    return

def check_download_base_model(base_model):
    if not modelsinfo.exists_model(catalog="checkpoints", model_path=base_model):
        load_file_from_url(
            url='https://huggingface.co/silveroxides/flux1-nf4-weights/resolve/main/{base_model}',
            model_dir=config.paths_checkpoints[0],
            file_name=base_model
        )
    return

def check_download_flux_model(base_model, clip_model=None):
    if not modelsinfo.exists_model(catalog="checkpoints", model_path=base_model):
        if 'nf4' in base_model:
            if 'schnell' in base_model:
                load_file_from_url(
                    url=f'https://huggingface.co/silveroxides/flux1-nf4-weights/resolve/main/{base_model}',
                    model_dir=config.paths_checkpoints[0],
                    file_name=base_model
                )
            else:
                load_file_from_url(
                    url=f'https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/{base_model}',
                    model_dir=config.paths_checkpoints[0],
                    file_name=base_model
                )
        elif 'fp8' in base_model:
            if 'schnell' in base_model:
                load_file_from_url(
                    url=f'https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/{base_model}',
                    model_dir=config.paths_checkpoints[0],
                    file_name=base_model
                )
            else:
                load_file_from_url(
                    url=f'https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/{base_model}',
                    model_dir=config.paths_checkpoints[0],
                    file_name=base_model
                )
        elif 'hyp8' in base_model:
            if '_K' in base_model:
                load_file_from_url(
                    url=f'https://huggingface.co/mhnakif/flux-hyp8-gguf-k/tree/main/{base_model}',
                    model_dir=config.paths_checkpoints[0],
                    file_name=base_model
                )
            else:
                load_file_from_url(
                    url=f'https://huggingface.co/mhnakif/flux-hyp8/tree/main/{base_model}',
                    model_dir=config.paths_checkpoints[0],
                    file_name=base_model
                )
        else:
            load_file_from_url(
                url=f'https://huggingface.co/metercai/SimpleSDXL2/resolve/main/SimpleModels/checkpoints/{base_model}',
                model_dir=config.paths_checkpoints[0],
                file_name=base_model
            )
    if clip_model:
        if not modelsinfo.exists_model(catalog="clip", model_path=clip_model):
            load_file_from_url(
                url=f'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/{clip_model}',
                model_dir=config.path_clip,
                file_name=f'{clip_model}'
            )
        if not modelsinfo.exists_model(catalog="clip", model_path='clip_l.safetensors'):
            load_file_from_url(
                url=f'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors',
                model_dir=config.path_clip,
                file_name=f'clip_l.safetensors'
            )
        if not modelsinfo.exists_model(catalog="vae", model_path='ae.safetensors'):
            load_file_from_url(
                url='https://huggingface.co/metercai/SimpleSDXL2/resolve/main/SimpleModels/vae/ae.safetensors',
                model_dir=config.path_vae,
                file_name='ae.safetensors'
            )



