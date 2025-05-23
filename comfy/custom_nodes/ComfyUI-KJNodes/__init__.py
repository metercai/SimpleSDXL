from .nodes.nodes import *
from .nodes.curve_nodes import *
from .nodes.batchcrop_nodes import *
from .nodes.audioscheduler_nodes import *
from .nodes.image_nodes import *
from .nodes.intrinsic_lora_nodes import *
from .nodes.mask_nodes import *
NODE_CONFIG = {
    #constants
    "INTConstant": {"class": INTConstant, "name": "INT Constant"},
    "FloatConstant": {"class": FloatConstant, "name": "Float Constant"},
    "StringConstant": {"class": StringConstant, "name": "String Constant"},
    "StringConstantMultiline": {"class": StringConstantMultiline, "name": "String Constant Multiline"},
    #conditioning
    "ConditioningMultiCombine": {"class": ConditioningMultiCombine, "name": "Conditioning Multi Combine"},
    "ConditioningSetMaskAndCombine": {"class": ConditioningSetMaskAndCombine, "name": "ConditioningSetMaskAndCombine"},
    "ConditioningSetMaskAndCombine3": {"class": ConditioningSetMaskAndCombine3, "name": "ConditioningSetMaskAndCombine3"},
    "ConditioningSetMaskAndCombine4": {"class": ConditioningSetMaskAndCombine4, "name": "ConditioningSetMaskAndCombine4"},
    "ConditioningSetMaskAndCombine5": {"class": ConditioningSetMaskAndCombine5, "name": "ConditioningSetMaskAndCombine5"},
    "CondPassThrough": {"class": CondPassThrough},
    #masking
    "DownloadAndLoadCLIPSeg": {"class": DownloadAndLoadCLIPSeg, "name": "(Down)load CLIPSeg"},
    "BatchCLIPSeg": {"class": BatchCLIPSeg, "name": "Batch CLIPSeg"},
    "ColorToMask": {"class": ColorToMask, "name": "Color To Mask"},
    "CreateGradientMask": {"class": CreateGradientMask, "name": "Create Gradient Mask"},
    "CreateTextMask": {"class": CreateTextMask, "name": "Create Text Mask"},
    "CreateAudioMask": {"class": CreateAudioMask, "name": "Create Audio Mask"},
    "CreateFadeMask": {"class": CreateFadeMask, "name": "Create Fade Mask"},
    "CreateFadeMaskAdvanced": {"class": CreateFadeMaskAdvanced, "name": "Create Fade Mask Advanced"},
    "CreateFluidMask": {"class": CreateFluidMask, "name": "Create Fluid Mask"},
    "CreateShapeMask": {"class": CreateShapeMask, "name": "Create Shape Mask"},
    "CreateVoronoiMask": {"class": CreateVoronoiMask, "name": "Create Voronoi Mask"},
    "CreateMagicMask": {"class": CreateMagicMask, "name": "Create Magic Mask"},
    "GetMaskSizeAndCount": {"class": GetMaskSizeAndCount, "name": "Get Mask Size & Count"},
    "GrowMaskWithBlur": {"class": GrowMaskWithBlur, "name": "Grow Mask With Blur"},
    "MaskBatchMulti": {"class": MaskBatchMulti, "name": "Mask Batch Multi"},
    "OffsetMask": {"class": OffsetMask, "name": "Offset Mask"},
    "RemapMaskRange": {"class": RemapMaskRange, "name": "Remap Mask Range"},
    "ResizeMask": {"class": ResizeMask, "name": "Resize Mask"},
    "RoundMask": {"class": RoundMask, "name": "Round Mask"},
    #images
    "AddLabel": {"class": AddLabel, "name": "Add Label"},
    "ColorMatch": {"class": ColorMatch, "name": "Color Match"},
    "CrossFadeImages": {"class": CrossFadeImages, "name": "Cross Fade Images"},
    "GetImagesFromBatchIndexed": {"class": GetImagesFromBatchIndexed, "name": "Get Images From Batch Indexed"},
    "GetImageRangeFromBatch": {"class": GetImageRangeFromBatch, "name": "Get Image or Mask Range From Batch"},
    "GetImageSizeAndCount": {"class": GetImageSizeAndCount, "name": "Get Image Size & Count"},
    "ImageAndMaskPreview": {"class": ImageAndMaskPreview},
    "ImageAddMulti": {"class": ImageAddMulti, "name": "Image Add Multi"},
    "ImageBatchMulti": {"class": ImageBatchMulti, "name": "Image Batch Multi"},
    "ImageBatchRepeatInterleaving": {"class": ImageBatchRepeatInterleaving},
    "ImageBatchTestPattern": {"class": ImageBatchTestPattern, "name": "Image Batch Test Pattern"},
    "ImageConcanate": {"class": ImageConcanate, "name": "Image Concatenate"},
    "ImageConcatFromBatch": {"class": ImageConcatFromBatch, "name": "Image Concatenate From Batch"},
    "ImageConcatMulti": {"class": ImageConcatMulti, "name": "Image Concatenate Multi"},
    "ImageGrabPIL": {"class": ImageGrabPIL, "name": "Image Grab PIL"},
    "ImageGridComposite2x2": {"class": ImageGridComposite2x2, "name": "Image Grid Composite 2x2"},
    "ImageGridComposite3x3": {"class": ImageGridComposite3x3, "name": "Image Grid Composite 3x3"},
    "ImageGridtoBatch": {"class": ImageGridtoBatch, "name": "Image Grid To Batch"},
    "ImageNormalize_Neg1_To_1": {"class": ImageNormalize_Neg1_To_1, "name": "Image Normalize -1 to 1"},
    "ImagePass": {"class": ImagePass},
    "ImagePadForOutpaintMasked": {"class": ImagePadForOutpaintMasked, "name": "Image Pad For Outpaint Masked"},
    "ImagePadForOutpaintTargetSize": {"class": ImagePadForOutpaintTargetSize, "name": "Image Pad For Outpaint Target Size"},
    "ImageResizeKJ": {"class": ImageResizeKJ, "name": "Resize Image"},
    "ImageUpscaleWithModelBatched": {"class": ImageUpscaleWithModelBatched, "name": "Image Upscale With Model Batched"},
    "InsertImagesToBatchIndexed": {"class": InsertImagesToBatchIndexed, "name": "Insert Images To Batch Indexed"},
    "LoadAndResizeImage": {"class": LoadAndResizeImage, "name": "Load & Resize Image"},
    "LoadImagesFromFolderKJ": {"class": LoadImagesFromFolderKJ, "name": "Load Images From Folder (KJ)"},
    "MergeImageChannels": {"class": MergeImageChannels, "name": "Merge Image Channels"},
    "PreviewAnimation": {"class": PreviewAnimation, "name": "Preview Animation"},
    "RemapImageRange": {"class": RemapImageRange, "name": "Remap Image Range"},
    "ReverseImageBatch": {"class": ReverseImageBatch, "name": "Reverse Image Batch"},
    "ReplaceImagesInBatch": {"class": ReplaceImagesInBatch, "name": "Replace Images In Batch"},
    "SaveImageWithAlpha": {"class": SaveImageWithAlpha, "name": "Save Image With Alpha"},
    "SplitImageChannels": {"class": SplitImageChannels, "name": "Split Image Channels"},    
    #batch cropping
    "BatchCropFromMask": {"class": BatchCropFromMask, "name": "Batch Crop From Mask"},
    "BatchCropFromMaskAdvanced": {"class": BatchCropFromMaskAdvanced, "name": "Batch Crop From Mask Advanced"},
    "FilterZeroMasksAndCorrespondingImages": {"class": FilterZeroMasksAndCorrespondingImages},
    "InsertImageBatchByIndexes": {"class": InsertImageBatchByIndexes, "name": "Insert Image Batch By Indexes"},
    "BatchUncrop": {"class": BatchUncrop, "name": "Batch Uncrop"},
    "BatchUncropAdvanced": {"class": BatchUncropAdvanced, "name": "Batch Uncrop Advanced"},
    "SplitBboxes": {"class": SplitBboxes, "name": "Split Bboxes"},
    "BboxToInt": {"class": BboxToInt, "name": "Bbox To Int"},
    "BboxVisualize": {"class": BboxVisualize, "name": "Bbox Visualize"},
    #noise
    "GenerateNoise": {"class": GenerateNoise, "name": "Generate Noise"},
    "FlipSigmasAdjusted": {"class": FlipSigmasAdjusted, "name": "Flip Sigmas Adjusted"},
    "InjectNoiseToLatent": {"class": InjectNoiseToLatent, "name": "Inject Noise To Latent"},
    "CustomSigmas": {"class": CustomSigmas, "name": "Custom Sigmas"},
    #utility
    "WidgetToString": {"class": WidgetToString, "name": "Widget To String"},
    "DummyOut": {"class": DummyOut, "name": "Dummy Out"},
    "GetLatentsFromBatchIndexed": {"class": GetLatentsFromBatchIndexed, "name": "Get Latents From Batch Indexed"},
    "ScaleBatchPromptSchedule": {"class": ScaleBatchPromptSchedule, "name": "Scale Batch Prompt Schedule"},
    "CameraPoseVisualizer": {"class": CameraPoseVisualizer, "name": "Camera Pose Visualizer"},
    "JoinStrings": {"class": JoinStrings, "name": "Join Strings"},
    "JoinStringMulti": {"class": JoinStringMulti, "name": "Join String Multi"},
    "SomethingToString": {"class": SomethingToString, "name": "Something To String"},
    "Sleep": {"class": Sleep, "name": "Sleep"},
    "VRAM_Debug": {"class": VRAM_Debug, "name": "VRAM Debug"},
    "SomethingToString": {"class": SomethingToString, "name": "Something To String"},
    "EmptyLatentImagePresets": {"class": EmptyLatentImagePresets, "name": "Empty Latent Image Presets"},
    "ModelPassThrough": {"class": ModelPassThrough, "name": "ModelPass"},
    #audioscheduler stuff
    "NormalizedAmplitudeToMask": {"class": NormalizedAmplitudeToMask},
    "NormalizedAmplitudeToFloatList": {"class": NormalizedAmplitudeToFloatList},
    "OffsetMaskByNormalizedAmplitude": {"class": OffsetMaskByNormalizedAmplitude},
    "ImageTransformByNormalizedAmplitude": {"class": ImageTransformByNormalizedAmplitude},
    #curve nodes
    "SplineEditor": {"class": SplineEditor, "name": "Spline Editor"},
    "CreateShapeImageOnPath": {"class": CreateShapeImageOnPath, "name": "Create Shape Image On Path"},
    "CreateShapeMaskOnPath": {"class": CreateShapeMaskOnPath, "name": "Create Shape Mask On Path"},
    "CreateTextOnPath": {"class": CreateTextOnPath, "name": "Create Text On Path"},
    "CreateGradientFromCoords": {"class": CreateGradientFromCoords, "name": "Create Gradient From Coords"},
    "GradientToFloat": {"class": GradientToFloat, "name": "Gradient To Float"},
    "WeightScheduleExtend": {"class": WeightScheduleExtend, "name": "Weight Schedule Extend"},
    "MaskOrImageToWeight": {"class": MaskOrImageToWeight, "name": "Mask Or Image To Weight"},
    "WeightScheduleConvert": {"class": WeightScheduleConvert, "name": "Weight Schedule Convert"},
    "FloatToMask": {"class": FloatToMask, "name": "Float To Mask"},
    "FloatToSigmas": {"class": FloatToSigmas, "name": "Float To Sigmas"},
    "PlotCoordinates": {"class": PlotCoordinates, "name": "Plot Coordinates"},
    "InterpolateCoords": {"class": InterpolateCoords, "name": "Interpolate Coords"},
    "PointsEditor": {"class": PointsEditor, "name": "Points Editor"},
    #experimental
    "StabilityAPI_SD3": {"class": StabilityAPI_SD3, "name": "Stability API SD3"},
    "SoundReactive": {"class": SoundReactive, "name": "Sound Reactive"},
    "StableZero123_BatchSchedule": {"class": StableZero123_BatchSchedule, "name": "Stable Zero123 Batch Schedule"},
    "SV3D_BatchSchedule": {"class": SV3D_BatchSchedule, "name": "SV3D Batch Schedule"},
    "LoadResAdapterNormalization": {"class": LoadResAdapterNormalization},
    "Superprompt": {"class": Superprompt, "name": "Superprompt"},
    "GLIGENTextBoxApplyBatchCoords": {"class": GLIGENTextBoxApplyBatchCoords},
    "Intrinsic_lora_sampling": {"class": Intrinsic_lora_sampling, "name": "Intrinsic Lora Sampling"},
    "CheckpointPerturbWeights": {"class": CheckpointPerturbWeights, "name": "CheckpointPerturbWeights"},
    "Screencap_mss": {"class": Screencap_mss, "name": "Screencap mss"},
    "WebcamCaptureCV2": {"class": WebcamCaptureCV2, "name": "Webcam Capture CV2"},
    "DifferentialDiffusionAdvanced": {"class": DifferentialDiffusionAdvanced, "name": "Differential Diffusion Advanced"},
    "FluxBlockLoraLoader": {"class": FluxBlockLoraLoader, "name": "Flux Block Lora Loader"},
    "FluxBlockLoraSelect": {"class": FluxBlockLoraSelect, "name": "Flux Block Lora Select"},

    #instance diffusion
    "CreateInstanceDiffusionTracking": {"class": CreateInstanceDiffusionTracking},
    "AppendInstanceDiffusionTracking": {"class": AppendInstanceDiffusionTracking},
    "DrawInstanceDiffusionTracking": {"class": DrawInstanceDiffusionTracking},
}

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"

from aiohttp import web
from server import PromptServer
from pathlib import Path

if hasattr(PromptServer, "instance"):

    # NOTE: we add an extra static path to avoid comfy mechanism
    # that loads every script in web.
    PromptServer.instance.app.add_routes(
        [web.static("/kjweb_async", (Path(__file__).parent.absolute() / "kjweb_async").as_posix())]
    )