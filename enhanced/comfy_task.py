from enhanced.simpleai import ComfyTaskParams

method_names = ['Blending given FG and IC-light', 'Generate foreground with Conv Injection']
#method_names = ['Blending given FG', 'Blending given BG', 'Blending given FG & BG', 'Generate foreground with Conv Injection']
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


task_name = {
    method_names[0]: 'iclight_fc',
    method_names[1]: 'layerdiffuse_fg',
}

class ComfyTask:

    def __init__(self, name, params, images=None):
        self.name = name
        self.params = params
        self.images = images


def get_comfy_task(method, default_params, input_images, options={}):
    global method_name, task_name

    if method == method_names[1]:
        comfy_params = ComfyTaskParams(default_params)
        comfy_params.update_params({"layer_diffuse_injection": "SDXL, Conv Injection"})
        return ComfyTask(task_name[method], comfy_params)
    else:
        comfy_params = ComfyTaskParams(default_params)
        if input_images is None:
            raise ValueError("input_images cannot be None for this method")
        images = {"input_image": input_images[0]}
        if 'iclight_enable' in options and options["iclight_enable"]:
            if options["iclight_source_radio"] == 'CenterLight':
                comfy_params.update_params({"light_source_text_switch": False})
            else:
                comfy_params.update_params({
                    "light_source_text_switch": True,
                    "light_source_text": iclight_source_text[options["iclight_source_radio"]]
                    })
            return ComfyTask(task_name[method], comfy_params, images)
        else:
            width, height = fixed_width_height(default_params["width"], default_params["height"], 64)
            comfy_params.update_params({
                "layer_diffuse_cond": "SDXL, Foreground",
                "width": width,
                "height": height,
                })
            return ComfyTask('layerdiffuse_cond', comfy_params, images)

def fixed_width_height(width, height, factor): 
    fixed_width = int(((height // factor + 1) * factor * width)/height)
    fixed_width = fixed_width if fixed_width % factor == 0 else int((fixed_width // factor + 1) * factor )
    width = width if height % factor == 0 else fixed_width
    height = height if height % factor == 0 else int((height // factor + 1) * factor)
    return width, height

