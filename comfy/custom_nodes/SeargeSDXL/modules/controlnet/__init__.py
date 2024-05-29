
import comfy.model_management as model_management

from .canny import CannyDetector
from .hed import HEDdetector
from .leres import LeresDetector
from .utils import annotator_wrapper


def canny(image, low_threshold, high_threshold):
    annotator = CannyDetector()

    low_threshold = int(low_threshold * 255)
    high_threshold = int(high_threshold * 255)

    def annotate(np_image):
        return annotator(np_image, low_threshold=low_threshold, high_threshold=high_threshold)

    return annotator_wrapper(image, annotate)


def leres(image, rm_nearest, rm_background, annotator_model):
    annotator = LeresDetector.from_pretrained(annotator_model).to(model_management.get_torch_device())

    rm_nearest = rm_nearest * 100
    rm_background = (1.0 - rm_background) * 100

    def annotate(np_image):
        return annotator(np_image, thr_a=rm_background, thr_b=rm_nearest)

    out = annotator_wrapper(image, annotate)

    del annotator

    return out


def hed(image, annotator_model):
    annotator = HEDdetector.from_pretrained(annotator_model).to(model_management.get_torch_device())

    def annotate(np_image):
        return annotator(np_image, safe=True)

    out = annotator_wrapper(image, annotate)

    del annotator

    return out
