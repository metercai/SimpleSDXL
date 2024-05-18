# modified version of https://github.com/AUTOMATIC1111/stable-diffusion-webui-nsfw-censor/blob/master/scripts/censor.py
import numpy as np
import os

from extras.safety_checker.models.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPConfig
from PIL import Image
import modules.config

safety_checker_repo_root = os.path.join(os.path.dirname(__file__), 'safety_checker')
config_path = os.path.join(safety_checker_repo_root, "configs", "config.json")
preprocessor_config_path = os.path.join(safety_checker_repo_root, "configs", "preprocessor_config.json")

safety_feature_extractor = None
safety_checker = None


def numpy_to_pil(image):
    image = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(image)

    return pil_image


# check and replace nsfw content
def check_safety(x_image):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None or safety_checker is None:
        safety_checker_model = modules.config.downloading_safety_checker_model()
        safety_feature_extractor = CLIPFeatureExtractor.from_json_file(preprocessor_config_path)
        clip_config = CLIPConfig.from_json_file(config_path)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_checker_model, config=clip_config)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)

    return x_checked_image, has_nsfw_concept


def censor_single(x):
    x_checked_image, has_nsfw_concept = check_safety(x)

    # replace image with black pixels, keep dimensions
    # workaround due to different numpy / pytorch image matrix format
    if has_nsfw_concept[0]:
        imageshape = x_checked_image.shape
        x_checked_image = np.zeros((imageshape[0], imageshape[1], 3), dtype = np.uint8)

    return x_checked_image


def censor_batch(images):
    images = [censor_single(image) for image in images]

    return images