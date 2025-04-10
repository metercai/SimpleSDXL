from .nodes import MarigoldDepthEstimation, MarigoldDepthEstimationVideo, ColorizeDepthmap, SaveImageOpenEXR, RemapDepth
from .nodes_v2 import MarigoldModelLoader, MarigoldDepthEstimation_v2, MarigoldDepthEstimation_v2_video

NODE_CLASS_MAPPINGS = {
    "MarigoldModelLoader": MarigoldModelLoader,
    "MarigoldDepthEstimation_v2": MarigoldDepthEstimation_v2,
    "MarigoldDepthEstimation_v2_video": MarigoldDepthEstimation_v2_video,
    "MarigoldDepthEstimation": MarigoldDepthEstimation,
    "MarigoldDepthEstimationVideo": MarigoldDepthEstimationVideo,
    "ColorizeDepthmap": ColorizeDepthmap,
    "SaveImageOpenEXR": SaveImageOpenEXR,
    "RemapDepth": RemapDepth
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldModelLoader": "MarigoldModelLoader",
    "MarigoldDepthEstimation_v2": "MarigoldDepthEstimation_v2",
    "MarigoldDepthEstimation_v2_video": "MarigoldDepthEstimation_v2_video",
    "MarigoldDepthEstimation": "MarigoldDepthEstimation",
    "MarigoldDepthEstimationVideo": "MarigoldDepthEstimationVideo",
    "ColorizeDepthmap": "Colorize Depthmap",
    "SaveImageOpenEXR": "SaveImageOpenEXR",
    "RemapDepth": "Remap Depth"
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]