from enhanced.simpleai import ComfyTaskParams

method_names = ['Blending given FG', 'Blending given BG', 'Generate foreground with Conv Injection']

task_name = {
    method_names[0]: 'layerdiffuse_cond',
    method_names[1]: 'layerdiffuse_cond',
    method_names[2]: 'layerdiffuse_fg',
}

class ComfyTask:

    def __init__(self, name, params, images=None):
        self.name = name
        self.params = params
        self.images = images


def get_comfy_task(method, default_params, input_images):
    global method_name, task_name

    if method == method_names[2]:
        comfy_params = ComfyTaskParams(default_params)
        comfy_params.update_params({"layer_diffuse_injection": "SDXL, Conv Injection"})
        return ComfyTask(task_name[method], comfy_params)
    elif method == method_names[0]:
        comfy_params = ComfyTaskParams(default_params)
        width, height = fixed_width_height(default_params["width"], default_params["height"], 64)
        comfy_params.update_params({
            "layer_diffuse_cond": "SDXL, Foreground",
            "width": width,
            "height": height,
            })
        images = {"input_image": input_images[0]}
        return ComfyTask(task_name[method], comfy_params, images)
    else:
        comfy_params = ComfyTaskParams(default_params)
        width, height = fixed_width_height(default_params["width"], default_params["height"], 64)
        comfy_params.update_params({
            "layer_diffuse_cond": "SDXL, Background",
            "width": width,
            "height": height,
            })
        images = {"input_image": input_image[0]}
        return ComfyTask(task_name[method], comfy_params, images)

def fixed_width_height(width, height, factor): 
    fixed_width = int(((height // factor + 1) * factor * width)/height)
    fixed_width = fixed_width if fixed_width % factor == 0 else int((fixed_width // factor + 1) * factor )
    width = width if height % factor == 0 else fixed_width
    height = height if height % factor == 0 else int((height // factor + 1) * factor)
    return width, height

