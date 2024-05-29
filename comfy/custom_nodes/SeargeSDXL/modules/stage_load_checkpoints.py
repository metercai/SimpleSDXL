"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 Searge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import folder_paths

from .data_utils import retrieve_parameter
from .mb_pipeline import PipelineAccess
from .names import Names
from .node_wrapper import NodeWrapper
from .ui import UI


# --------------------------------------------------------------------------------
# Stage: Load Checkpoints
# --------------------------------------------------------------------------------

class SeargeStageLoadCheckpoints:
    def __init__(self):
        pass

    def get_input(self, data, stage_data):
        # if we still don't have stage data,
        if stage_data is None and data is not None:
            stage_data = {
                PipelineAccess.NAME: retrieve_parameter(PipelineAccess.NAME, data),
            }

        return stage_data

    def process(self, data, stage_input):
        access = PipelineAccess(stage_input)

        # TODO: this stage will always execute, even if the pipeline is disabled and in that case unload all models
        if not access.is_pipeline_enabled():
            pass

        freeu_mode = access.get_active_setting(UI.S_FREEU, UI.F_FREEU_MODE, UI.NONE)
        freeu_changed = access.setting_changed(UI.S_FREEU, UI.F_FREEU_MODE)

        freeu_version = access.get_active_setting(UI.S_FREEU, UI.F_FREEU_VERSION, UI.FREEU_V1)
        freeu_changed = freeu_changed or access.setting_changed(UI.S_FREEU, UI.F_FREEU_VERSION)

        if freeu_mode == UI.CUSTOM:
            freeu_changed = freeu_changed or access.setting_changed(UI.S_FREEU, UI.F_FREEU_B1)
            freeu_changed = freeu_changed or access.setting_changed(UI.S_FREEU, UI.F_FREEU_B2)
            freeu_changed = freeu_changed or access.setting_changed(UI.S_FREEU, UI.F_FREEU_S1)
            freeu_changed = freeu_changed or access.setting_changed(UI.S_FREEU, UI.F_FREEU_S2)

        if freeu_mode == UI.FREEU_DEFAULT:
            b1 = 1.0
            b2 = 1.2
            s1 = 0.9
            s2 = 0.8
        elif freeu_mode == UI.FREEU_SUBTLE:
            b1 = 1.1
            b2 = 1.2
            s1 = 1.0
            s2 = 0.9
        elif freeu_mode == UI.FREEU_SD_XL:
            b1 = 1.1
            b2 = 1.2
            s1 = 0.6
            s2 = 0.4
        elif freeu_mode == UI.FREEU_SD_XL_V2:
            b1 = 1.3
            b2 = 1.4
            s1 = 0.9
            s2 = 0.2
        elif freeu_mode == UI.FREEU_DETAILED:
            b1 = 1.2
            b2 = 1.4
            s1 = 0.9
            s2 = 0.7
        elif freeu_mode == UI.FREEU_SD_2_1:
            b1 = 1.1
            b2 = 1.2
            s1 = 0.9
            s2 = 0.2
        elif freeu_mode == UI.FREEU_SD_1_5:
            b1 = 1.2
            b2 = 1.4
            s1 = 0.9
            s2 = 0.2
        else:  # covers NONE and CUSTOM
            b1 = access.get_active_setting(UI.S_FREEU, UI.F_FREEU_B1, 1.1)
            b2 = access.get_active_setting(UI.S_FREEU, UI.F_FREEU_B2, 1.2)
            s1 = access.get_active_setting(UI.S_FREEU, UI.F_FREEU_S1, 0.6)
            s2 = access.get_active_setting(UI.S_FREEU, UI.F_FREEU_S2, 0.4)

        base_changed = access.setting_changed(UI.S_CHECKPOINTS, UI.F_BASE_CHECKPOINT)
        if base_changed:
            base_name = access.get_active_setting(UI.S_CHECKPOINTS, UI.F_BASE_CHECKPOINT)

            if folder_paths.get_full_path("checkpoints", base_name) is None:
                base_checkpoint = (None, None, None)
            else:
                base_checkpoint = NodeWrapper.checkpoint_loader.load_checkpoint(base_name)

            access.update_in_cache(Names.C_BASE_CHECKPOINT, [base_name], base_checkpoint)
        else:
            base_checkpoint = access.get_from_cache(Names.C_BASE_CHECKPOINT)

        base_model = base_checkpoint[0]
        if freeu_mode is not None and freeu_mode != UI.NONE and freeu_version is not None:
            if freeu_version == UI.FREEU_V1:
                base_model = NodeWrapper.freeu.patch(base_model, b1, b2, s1, s2)[0]
            elif freeu_version == UI.FREEU_V2:
                base_model = NodeWrapper.freeu_v2.patch(base_model, b1, b2, s1, s2)[0]
        base_clip = base_checkpoint[1]
        base_vae = base_checkpoint[2]

        if base_changed or freeu_changed:
            access.update_in_pipeline(Names.P_BASE_MODEL, base_model, True)
            access.update_in_pipeline(Names.P_BASE_CLIP, base_clip, True)
            access.update_in_pipeline(Names.P_BASE_VAE, base_vae, True)
        else:
            access.restore_in_pipeline(Names.P_BASE_MODEL, base_model, True)
            access.restore_in_pipeline(Names.P_BASE_CLIP, base_clip, True)
            access.restore_in_pipeline(Names.P_BASE_VAE, base_vae, True)

        refiner_changed = access.setting_changed(UI.S_CHECKPOINTS, UI.F_REFINER_CHECKPOINT)
        if refiner_changed:
            refiner_name = access.get_active_setting(UI.S_CHECKPOINTS, UI.F_REFINER_CHECKPOINT, UI.NONE)

            if refiner_name == UI.NONE or folder_paths.get_full_path("checkpoints", refiner_name) is None:
                refiner_checkpoint = (None, None, None,)
            else:
                refiner_checkpoint = NodeWrapper.checkpoint_loader.load_checkpoint(refiner_name)

            access.update_in_cache(Names.C_REFINER_CHECKPOINT, [refiner_name], refiner_checkpoint)
        else:
            refiner_checkpoint = access.get_from_cache(Names.C_REFINER_CHECKPOINT)

        refiner_model = refiner_checkpoint[0]
        if freeu_mode is not None and freeu_mode != UI.NONE and freeu_version is not None and refiner_model is not None:
            if freeu_version == UI.FREEU_V1:
                refiner_model = NodeWrapper.freeu.patch(refiner_model, b1, b2, s1, s2)[0]
            elif freeu_version == UI.FREEU_V2:
                refiner_model = NodeWrapper.freeu_v2.patch(refiner_model, b1, b2, s1, s2)[0]
        refiner_clip = refiner_checkpoint[1]
        refiner_vae = refiner_checkpoint[2]

        if refiner_changed or freeu_changed:
            access.update_in_pipeline(Names.P_REFINER_MODEL, refiner_model, True)
            access.update_in_pipeline(Names.P_REFINER_CLIP, refiner_clip, True)
            access.update_in_pipeline(Names.P_REFINER_VAE, refiner_vae, True)
        else:
            access.restore_in_pipeline(Names.P_REFINER_MODEL, refiner_model, True)
            access.restore_in_pipeline(Names.P_REFINER_CLIP, refiner_clip, True)
            access.restore_in_pipeline(Names.P_REFINER_VAE, refiner_vae, True)

        vae_changed = access.setting_changed(UI.S_CHECKPOINTS, UI.F_VAE_CHECKPOINT)
        if vae_changed:
            vae_name = access.get_active_setting(UI.S_CHECKPOINTS, UI.F_VAE_CHECKPOINT, UI.VAE_FROM_BASE_MODEL)

            if vae_name == UI.VAE_FROM_REFINER_MODEL:
                if refiner_vae is not None and folder_paths.get_full_path("vae", vae_name) is not None:
                    vae_checkpoint = refiner_vae
                else:
                    vae_checkpoint = base_vae

            elif vae_name == UI.VAE_FROM_BASE_MODEL:
                vae_checkpoint = base_vae

            else:
                vae_checkpoint = NodeWrapper.vae_loader.load_vae(vae_name)[0]

            access.update_in_cache(Names.C_VAE_CHECKPOINT, [vae_name], vae_checkpoint)
            access.update_in_pipeline(Names.P_VAE_MODEL, vae_checkpoint, True)
        else:
            vae_checkpoint = access.get_from_cache(Names.C_VAE_CHECKPOINT)
            access.restore_in_pipeline(Names.P_VAE_MODEL, vae_checkpoint, True)

        vae_model = vae_checkpoint

        hires_upscaler_changed = access.setting_changed(UI.S_UPSCALE_MODELS, UI.F_HIGH_RES_UPSCALER)
        if hires_upscaler_changed:
            hires_name = access.get_active_setting(UI.S_UPSCALE_MODELS, UI.F_HIGH_RES_UPSCALER, UI.NONE)

            if hires_name != UI.NONE and folder_paths.get_full_path("upscale_models", hires_name) is not None:
                hires_upscaler_model = NodeWrapper.upscale_loader.load_model(hires_name)[0]
            else:
                hires_upscaler_model = None

            access.update_in_cache(Names.C_HIRES_UPSCALE_MODEL, [hires_name], hires_upscaler_model)
            access.update_in_pipeline(Names.P_HIRES_UPSCALER, hires_upscaler_model, True)
        else:
            hires_upscaler_model = access.get_from_cache(Names.C_HIRES_UPSCALE_MODEL)
            access.restore_in_pipeline(Names.P_HIRES_UPSCALER, hires_upscaler_model, True)

        hires_upscaler = hires_upscaler_model

        primary_upscaler_changed = access.setting_changed(UI.S_UPSCALE_MODELS, UI.F_PRIMARY_UPSCALER)
        if primary_upscaler_changed:
            primary_name = access.get_active_setting(UI.S_UPSCALE_MODELS, UI.F_PRIMARY_UPSCALER, UI.NONE)

            if primary_name != UI.NONE and folder_paths.get_full_path("upscale_models", primary_name) is not None:
                primary_upscaler_model = NodeWrapper.upscale_loader.load_model(primary_name)[0]
            else:
                primary_upscaler_model = None

            access.update_in_cache(Names.C_PRIMARY_UPSCALE_MODEL, [primary_name], primary_upscaler_model)
            access.update_in_pipeline(Names.P_PRIMARY_UPSCALER, primary_upscaler_model, True)
        else:
            primary_upscaler_model = access.get_from_cache(Names.C_PRIMARY_UPSCALE_MODEL)
            access.restore_in_pipeline(Names.P_PRIMARY_UPSCALER, primary_upscaler_model, True)

        primary_upscaler = primary_upscaler_model

        secondary_upscaler_changed = access.setting_changed(UI.S_UPSCALE_MODELS, UI.F_SECONDARY_UPSCALER)
        if secondary_upscaler_changed:
            secondary_name = access.get_active_setting(UI.S_UPSCALE_MODELS, UI.F_SECONDARY_UPSCALER, UI.NONE)

            if secondary_name != UI.NONE and folder_paths.get_full_path("upscale_models", secondary_name) is not None:
                secondary_upscaler_model = NodeWrapper.upscale_loader.load_model(secondary_name)[0]
            else:
                secondary_upscaler_model = None

            access.update_in_cache(Names.C_SECONDARY_UPSCALE_MODEL, [secondary_name], secondary_upscaler_model)
            access.update_in_pipeline(Names.P_SECONDARY_UPSCALER, secondary_upscaler_model, True)
        else:
            secondary_upscaler_model = access.get_from_cache(Names.C_SECONDARY_UPSCALE_MODEL)
            access.restore_in_pipeline(Names.P_SECONDARY_UPSCALER, secondary_upscaler_model, True)

        secondary_upscaler = secondary_upscaler_model

        detail_processor_changed = access.setting_changed(UI.S_UPSCALE_MODELS, UI.F_DETAIL_PROCESSOR)
        if detail_processor_changed:
            detailer_name = access.get_active_setting(UI.S_UPSCALE_MODELS, UI.F_DETAIL_PROCESSOR, UI.NONE)

            if detailer_name != UI.NONE and folder_paths.get_full_path("upscale_models", detailer_name) is not None:
                detail_processor_model = NodeWrapper.upscale_loader.load_model(detailer_name)[0]
            else:
                detail_processor_model = None

            access.update_in_cache(Names.C_DETAIL_PROCESSOR_MODEL, [detailer_name], detail_processor_model)
            access.update_in_pipeline(Names.P_DETAIL_PROCESSOR, detail_processor_model, True)
        else:
            detail_processor_model = access.get_from_cache(Names.C_DETAIL_PROCESSOR_MODEL)
            access.restore_in_pipeline(Names.P_DETAIL_PROCESSOR, detail_processor_model, True)

        detail_processor = detail_processor_model

        clip_vision_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_CLIP_VISION_CHECKPOINT)
        if clip_vision_checkpoint_changed:
            clipvision_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_CLIP_VISION_CHECKPOINT, UI.NONE)

            if clipvision_name != UI.NONE and folder_paths.get_full_path("clip_vision", clipvision_name) is not None:
                clip_vision_model = NodeWrapper.clipvision_loader.load_clip(clipvision_name)[0]
            else:
                clip_vision_model = None

            access.update_in_cache(Names.C_CLIP_VISION_MODEL, [clipvision_name], clip_vision_model)
            access.update_in_pipeline(Names.P_CLIP_VISION_MODEL, clip_vision_model, True)
        else:
            clip_vision_model = access.get_from_cache(Names.C_CLIP_VISION_MODEL)
            access.restore_in_pipeline(Names.P_CLIP_VISION_MODEL, clip_vision_model, True)

        clip_vision = clip_vision_model

        canny_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_CANNY_CHECKPOINT)
        if canny_checkpoint_changed:
            canny_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_CANNY_CHECKPOINT, UI.NONE)

            if canny_name != UI.NONE and folder_paths.get_full_path("controlnet", canny_name) is not None:
                canny_model = NodeWrapper.controlnet_loader.load_controlnet(canny_name)[0]
            else:
                canny_model = None

            access.update_in_cache(Names.C_CN_CANNY_MODEL, [canny_name], canny_model)
            access.update_in_pipeline(Names.P_CN_CANNY_MODEL, canny_model, True)
        else:
            canny_model = access.get_from_cache(Names.C_CN_CANNY_MODEL)
            access.restore_in_pipeline(Names.P_CN_CANNY_MODEL, canny_model, True)

        cn_canny = canny_model

        depth_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_DEPTH_CHECKPOINT)
        if depth_checkpoint_changed:
            depth_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_DEPTH_CHECKPOINT, UI.NONE)

            if depth_name != UI.NONE and folder_paths.get_full_path("controlnet", depth_name) is not None:
                depth_model = NodeWrapper.controlnet_loader.load_controlnet(depth_name)[0]
            else:
                depth_model = None

            access.update_in_cache(Names.C_CN_DEPTH_MODEL, [depth_name], depth_model)
            access.update_in_pipeline(Names.P_CN_DEPTH_MODEL, depth_model, True)
        else:
            depth_model = access.get_from_cache(Names.C_CN_DEPTH_MODEL)
            access.restore_in_pipeline(Names.P_CN_DEPTH_MODEL, depth_model, True)

        cn_depth = depth_model

        recolor_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_RECOLOR_CHECKPOINT)
        if recolor_checkpoint_changed:
            recolor_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_RECOLOR_CHECKPOINT, UI.NONE)

            if recolor_name != UI.NONE and folder_paths.get_full_path("controlnet", recolor_name) is not None:
                recolor_model = NodeWrapper.controlnet_loader.load_controlnet(recolor_name)[0]
            else:
                recolor_model = None

            access.update_in_cache(Names.C_CN_RECOLOR_MODEL, [recolor_name], recolor_model)
            access.update_in_pipeline(Names.P_CN_RECOLOR_MODEL, recolor_model, True)
        else:
            recolor_model = access.get_from_cache(Names.C_CN_RECOLOR_MODEL)
            access.restore_in_pipeline(Names.P_CN_RECOLOR_MODEL, recolor_model, True)

        cn_recolor = recolor_model

        sketch_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_SKETCH_CHECKPOINT)
        if sketch_checkpoint_changed:
            sketch_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_SKETCH_CHECKPOINT, UI.NONE)

            if sketch_name != UI.NONE and folder_paths.get_full_path("controlnet", sketch_name) is not None:
                sketch_model = NodeWrapper.controlnet_loader.load_controlnet(sketch_name)[0]
            else:
                sketch_model = None

            access.update_in_cache(Names.C_CN_SKETCH_MODEL, [sketch_name], sketch_model)
            access.update_in_pipeline(Names.P_CN_SKETCH_MODEL, sketch_model, True)
        else:
            sketch_model = access.get_from_cache(Names.C_CN_SKETCH_MODEL)
            access.restore_in_pipeline(Names.P_CN_SKETCH_MODEL, sketch_model, True)

        cn_sketch = sketch_model

        custom_checkpoint_changed = access.setting_changed(UI.S_CONTROLNET_MODELS, UI.F_CUSTOM_CHECKPOINT)
        if custom_checkpoint_changed:
            custom_name = access.get_active_setting(UI.S_CONTROLNET_MODELS, UI.F_CUSTOM_CHECKPOINT, UI.NONE)

            if custom_name != UI.NONE and folder_paths.get_full_path("controlnet", custom_name) is not None:
                custom_model = NodeWrapper.controlnet_loader.load_controlnet(custom_name)[0]
            else:
                custom_model = None

            access.update_in_cache(Names.C_CN_CUSTOM_MODEL, [custom_name], custom_model)
            access.update_in_pipeline(Names.P_CN_CUSTOM_MODEL, custom_model, True)
        else:
            custom_model = access.get_from_cache(Names.C_CN_CUSTOM_MODEL)
            access.restore_in_pipeline(Names.P_CN_CUSTOM_MODEL, custom_model, True)

        cn_custom = custom_model

        loaded_models = {
            Names.F_BASE_MODEL: base_model,
            Names.F_BASE_CLIP: base_clip,
            Names.F_BASE_VAE: base_vae,
            Names.F_REFINER_MODEL: refiner_model,
            Names.F_REFINER_CLIP: refiner_clip,
            Names.F_REFINER_VAE: refiner_vae,
            Names.F_VAE_MODEL: vae_model,
            Names.F_HIRES_UPSCALER: hires_upscaler,
            Names.F_PRIMARY_UPSCALER: primary_upscaler,
            Names.F_SECONDARY_UPSCALER: secondary_upscaler,
            Names.F_DETAIL_PROCESSOR: detail_processor,
            Names.F_CLIP_VISION_MODEL: clip_vision,
            Names.F_CN_CANNY_MODEL: cn_canny,
            Names.F_CN_DEPTH_MODEL: cn_depth,
            Names.F_CN_RECOLOR_MODEL: cn_recolor,
            Names.F_CN_SKETCH_MODEL: cn_sketch,
            Names.F_CN_CUSTOM_MODEL: cn_custom,
        }

        if data is not None:
            data[Names.S_LOADED_MODELS] = loaded_models

        stage_output = {
            Names.S_LOADED_MODELS: loaded_models,
        }

        return (data, stage_output,)
