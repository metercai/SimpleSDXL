import threading
import queue

import args_manager
from extras.inpaint_mask import generate_mask_from_image, SAMOptions
from modules.patch import PatchSettings, patch_settings, patch_all
import modules.config

patch_all()


class AsyncTask:
    def __init__(self, args, task_id=None):
        self.args = args
        self.yields = []
        self.results = []
        from modules.flags import Performance, MetadataScheme, ip_list, disabled, task_class_mapping, default_vae
        from modules.util import get_enabled_loras
        from modules.config import default_max_lora_number
        import uuid
        import args_manager
        import re
        import shared

        from enhanced.simpleai import comfyd

        normalize_lines = lambda text: re.sub(r'[\r\n]+', ' ', text)

        self.args = args.copy()
        self.yields = []
        self.results = []
        self.last_stop = False
        self.processing = False
        self.task_id = str(uuid.uuid4()) if task_id is None else task_id
        self.remote_task = False
        self.img_paths = []

        self.performance_loras = []

        if len(args) == 0:
            return

        args.reverse()
        self.generate_image_grid = args.pop()
        self.prompt = normalize_lines(args.pop())
        self.negative_prompt = args.pop()
        self.style_selections = args.pop()

        self.performance_selection = Performance(args.pop())
        self.steps = self.performance_selection.steps()
        self.original_steps = self.steps

        self.aspect_ratios_selection = args.pop()
        self.image_number = args.pop()
        self.output_format = args.pop()
        self.seed = int(args.pop())
        self.read_wildcards_in_order = args.pop()
        self.sharpness = args.pop()
        self.cfg_scale = args.pop()
        self.base_model_name = args.pop()
        self.refiner_model_name = args.pop()
        self.refiner_switch = args.pop()
        self.loras = get_enabled_loras([(bool(args.pop()), str(args.pop()), float(args.pop())) for _ in
                                        range(default_max_lora_number)])
        self.input_image_checkbox = args.pop()
        self.current_tab = args.pop()
        self.uov_method = args.pop()
        self.uov_input_image = args.pop()
        self.outpaint_selections = args.pop()
        self.inpaint_input_image = args.pop()
        self.inpaint_additional_prompt = args.pop()
        self.inpaint_mask_image_upload = args.pop()

        self.layer_method = args.pop()
        self.layer_input_image = args.pop()
        self.iclight_enable = args.pop()
        self.iclight_source_radio = args.pop()

        self.disable_preview = args.pop()
        self.disable_intermediate_results = args.pop()
        self.disable_seed_increment = args.pop()
        self.black_out_nsfw = args.pop()
        self.adm_scaler_positive = args.pop()
        self.adm_scaler_negative = args.pop()
        self.adm_scaler_end = args.pop()
        self.adaptive_cfg = args.pop()
        self.clip_skip = args.pop()
        self.sampler_name = args.pop()
        self.scheduler_name = args.pop()
        self.vae_name = args.pop()
        self.overwrite_step = args.pop()
        self.overwrite_switch = args.pop()
        self.overwrite_width = args.pop()
        self.overwrite_height = args.pop()
        self.overwrite_vary_strength = args.pop()
        self.overwrite_upscale_strength = args.pop()
        self.mixing_image_prompt_and_vary_upscale = args.pop()
        self.mixing_image_prompt_and_inpaint = args.pop()
        self.debugging_cn_preprocessor = args.pop()
        self.skipping_cn_preprocessor = args.pop()
        self.canny_low_threshold = args.pop()
        self.canny_high_threshold = args.pop()
        self.refiner_swap_method = args.pop()
        self.controlnet_softness = args.pop()
        self.freeu_enabled = args.pop()
        self.freeu_b1 = args.pop()
        self.freeu_b2 = args.pop()
        self.freeu_s1 = args.pop()
        self.freeu_s2 = args.pop()
        self.debugging_inpaint_preprocessor = args.pop()
        self.inpaint_disable_initial_latent = args.pop()
        self.inpaint_engine = args.pop()
        self.inpaint_strength = args.pop()
        self.inpaint_respective_field = args.pop()
        self.inpaint_advanced_masking_checkbox = args.pop()
        self.invert_mask_checkbox = args.pop()
        self.inpaint_erode_or_dilate = args.pop()

        self.params_backend = args.pop().copy()
        self.backfill_prompt = self.params_backend.pop('backfill_prompt')
        self.translation_methods = self.params_backend.pop('translation_methods')
        self.comfyd_active_checkbox = self.params_backend.pop('comfyd_active_checkbox')
        self.nickname = self.params_backend.pop('nickname')
        self.user_did = self.params_backend.pop('user_did')


        self.save_final_enhanced_image_only = args.pop() if not args_manager.args.disable_image_log else False
        self.save_metadata_to_images = args.pop() if not args_manager.args.disable_metadata else False
        self.metadata_scheme = MetadataScheme(
            args.pop()) if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        self.cn_tasks = {x: [] for x in ip_list}
        self.cn_task_num = 0
        for _ in range(modules.config.default_controlnet_image_count):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                self.cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])
                self.cn_task_num += 1

        self.debugging_dino = args.pop()
        self.dino_erode_or_dilate = args.pop()
        self.debugging_enhance_masks_checkbox = args.pop()

        self.enhance_input_image = args.pop()
        self.enhance_checkbox = args.pop()
        self.enhance_uov_method = args.pop()
        self.enhance_uov_processing_order = args.pop()
        self.enhance_uov_prompt_type = args.pop()
        self.enhance_ctrls = []
        for _ in range(modules.config.default_enhance_tabs):
            enhance_enabled = args.pop()
            enhance_mask_dino_prompt_text = args.pop()
            enhance_prompt = args.pop()
            enhance_negative_prompt = args.pop()
            enhance_mask_model = args.pop()
            enhance_mask_cloth_category = args.pop()
            enhance_mask_sam_model = args.pop()
            enhance_mask_text_threshold = args.pop()
            enhance_mask_box_threshold = args.pop()
            enhance_mask_sam_max_detections = args.pop()
            enhance_inpaint_disable_initial_latent = args.pop()
            enhance_inpaint_engine = args.pop()
            enhance_inpaint_strength = args.pop()
            enhance_inpaint_respective_field = args.pop()
            enhance_inpaint_erode_or_dilate = args.pop()
            enhance_mask_invert = args.pop()
            if enhance_enabled:
                self.enhance_ctrls.append([
                    enhance_mask_dino_prompt_text,
                    enhance_prompt,
                    enhance_negative_prompt,
                    enhance_mask_model,
                    enhance_mask_cloth_category,
                    enhance_mask_sam_model,
                    enhance_mask_text_threshold,
                    enhance_mask_box_threshold,
                    enhance_mask_sam_max_detections,
                    enhance_inpaint_disable_initial_latent,
                    enhance_inpaint_engine,
                    enhance_inpaint_strength,
                    enhance_inpaint_respective_field,
                    enhance_inpaint_erode_or_dilate,
                    enhance_mask_invert
                ])
        self.should_enhance = self.enhance_checkbox and (self.enhance_uov_method != disabled.casefold() or len(self.enhance_ctrls) > 0)
        self.images_to_enhance_count = 0
        self.enhance_stats = {}

        self.task_class = self.params_backend.pop('backend_engine', 'Fooocus')
        self.task_name = self.params_backend.pop('preset', 'default')
        self.task_method = self.params_backend.pop('task_method', 'text2image')
        self.preset = self.task_name
        if 'layer' in self.current_tab and self.task_class == 'Fooocus' and self.input_image_checkbox:
            self.task_class = 'Comfy'
            self.task_name = 'default'
            self.task_method = self.layer_method
        self.task_class_full = task_class_mapping[self.task_class]

        if self.task_class in ['Kolors', 'Flux', 'HyDiT', 'SD3x'] and self.task_name not in ['Kolors', 'Flux', 'HyDiT', 'SD3x']:
            self.task_name = self.task_class
        if len(self.loras) > 0:
            if self.task_name in ['Kolors', 'Flux'] and '_aio' not in self.task_method:
                self.params_backend.update({
                    "lora_1": self.loras[0][0],
                    "lora_1_strength": self.loras[0][1],
                    })
            elif len(self.loras) > 1 and (self.task_name in ['Kolors'] or 'base2_gguf' in self.task_method) and '_aio' not in self.task_method:
                self.params_backend.update({
                    "lora_1": self.loras[0][0],
                    "lora_1_strength": self.loras[0][1],
                    "lora_2": self.loras[1][0],
                    "lora_2_strength": self.loras[1][1],
                    })
            else:
                for i, (lora_name, lora_strength) in enumerate(self.loras):
                    self.params_backend.update({
                        f"lora_{i+1}": lora_name,
                        f"lora_{i+1}_strength": lora_strength,
                        })

        ui_options = {
            'iclight_enable': self.iclight_enable,
            'iclight_source_radio': self.iclight_source_radio,
            }
        if self.task_name == 'default' and self.task_class == 'Comfy':
            self.params_backend.update({"ui_options": ui_options})
       
        if self.task_class != 'Fooocus' and self.vae_name != default_vae:
            self.params_backend.update({"vae_model": self.vae_name})
            self.params_backend.update({"is_custom_vae": True})

        if 'display_step' in self.params_backend and self.overwrite_step!=-1:
            self.params_backend.pop('display_step')
            
        if 'scene_frontend' in self.params_backend:
            self.aspect_ratios_selection = self.params_backend.pop('scene_aspect_ratio')
            self.image_number = self.params_backend.pop('scene_image_number')
            self.scene_canvas_image = self.params_backend.pop('scene_canvas_image')
            self.scene_input_image1 = self.params_backend.pop('scene_input_image1')
            self.scene_theme = self.params_backend.pop('scene_theme')
            self.scene_additional_prompt = self.params_backend.pop('scene_additional_prompt')
            self.scene_steps = self.params_backend.pop('scene_steps', None)
            self.scene_frontend = self.params_backend.pop('scene_frontend')

class EarlyReturnException(BaseException):
    pass


def worker():
    global async_tasks, worker_processing, pending_tasks


    import os
    import traceback
    import math
    import numpy as np
    import torch
    import time
    import shared
    import random
    import copy
    import cv2
    import re
    import json
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.patch
    import ldm_patched.modules.model_management
    import extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import extras.ip_adapter as ip_adapter
    import extras.face_crop
    import fooocus_version
    import enhanced.wildcards as wildcards
    import enhanced.version as version
    import enhanced.translator as translator
    import enhanced.all_parameters as ads

    from extras.censor import default_censor
    from modules.sdxl_styles import apply_style, get_random_style, fooocus_expansion, apply_arrays, random_style_name
    from modules.private_logger import log, p2p_log
    from extras.expansion import safe_str
    from modules.util import (remove_empty_str, HWC3, resize_image, get_image_shape_ceil, set_image_shape_ceil,
                              get_shape_ceil, resample_image, erode_or_dilate, parse_lora_references_from_prompt,
                              apply_wildcards)
    from modules.upscaler import perform_upscale
    from modules.flags import Performance
    from modules.meta_parser import get_metadata_parser, MetadataParser
    from modules.model_loader import is_models_file_absent

    from enhanced.simpleai import comfyd, comfyclient_pipeline as comfypipeline, get_echo_off, p2p_task
    from enhanced.comfy_task import get_comfy_task, default_kolors_base_model_name
    from enhanced.all_parameters import default as default_params
    from enhanced.minicpm import MiniCPM
    import logging
    from enhanced.logger import format_name
    logger = logging.getLogger(format_name(__name__))

    minicpm = MiniCPM()
    pid = os.getpid()
    logger.info(f'Started worker with PID {pid}')

    try:
        async_gradio_app = shared.gradio_root
        flag = 'App started successful.'
        if hasattr(async_gradio_app, 'local_url') and async_gradio_app.local_url:
            flag += f' Use the app with {str(async_gradio_app.local_url)}'
        if hasattr(async_gradio_app, 'server_name') and hasattr(async_gradio_app, 'server_port') and async_gradio_app.server_name and async_gradio_app.server_port:
            if 'Use the app with' in flag:
                flag += f' or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'
            else:
                flag += f'Use the app with {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'
        if hasattr(async_gradio_app, 'share') and async_gradio_app.share_url:
            flag += f''' or {async_gradio_app.share_url}'''
        logger.info(flag)
    except Exception as e:
        logger.info(e)
    ldm_patched.modules.model_management.print_memory_info("worker init")

    def progressbar(async_task, number, text, img=None):
        if async_task.remote_task:
            p2p_task.call_remote_progress(async_task, number, text, img)
            return
        if img is None:
            logger.info(f'{text}')
        async_task.yields.append(['preview', (number, text, img)])

    def yield_result(async_task, imgs, progressbar_index, black_out_nsfw, censor=True, do_not_show_finished_images=False):
        if async_task.remote_task:
            p2p_task.call_remote_result(async_task, imgs, progressbar_index, black_out_nsfw, censor, do_not_show_finished_images)
            return
        if imgs is None:
            imgs = async_task.img_paths
            async_task.img_paths = []
        if not isinstance(imgs, list):
            imgs = [imgs]

        if censor and (modules.config.default_black_out_nsfw or black_out_nsfw):
            progressbar(async_task, progressbar_index, '检测NSFW ...')
            imgs = default_censor(imgs)

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        async_task.yields.append(['results', async_task.results])
        return

    def image_log(async_task, img, metadata, metadata_parser: MetadataParser | None = None, output_format=None, task=None, persist_image=True, user_did=None, remote_task=False):
        paths, image, log_item = log(img, metadata, metadata_parser, output_format, task, persist_image, user_did, remote_task)
        async_task.img_paths.append(paths)
        if remote_task:
            p2p_task.call_remote_save_and_log(async_task, image, log_item)

    def p2p_save_and_log(async_task, result_img, result_log):
        paths = p2p_log(result_img, result_log, async_task.output_format, persist_image=True, user_did=async_task.user_did)
        async_task.img_paths.append(paths)
    
    def stop_processing(async_task, processing_start_time, status="Finished"):
        async_task.processing = False
        async_task.img_paths = []
        processing_time = (int(time.time() * 1000) - processing_start_time)/1000.0
        if status=="Finished":
            logger.info(f'Processing time (total): {processing_time:.2f} seconds')
        else:
            if processing_start_time==0:
                logger.info(f'Processing stop, status:{status}')
            else:
                logger.info(f'Processing time (total): {processing_time:.2f} seconds, status:{status}')
        if async_task.task_class in flags.comfy_classes:
            if async_task.comfyd_active_checkbox:
                comfyd.finished()
            else:
                comfyd.stop()
        if async_task.remote_task:
            p2p_task.call_remote_stop(async_task, processing_start_time, status)
        return

    def interrupt_processing():
        comfyd.interrupt()
        ldm_patched.modules.model_management.interrupt_current_processing()

    def build_image_wall(async_task):
        results = []

        if len(async_task.results) < 2:
            return

        for img in async_task.results:
            if isinstance(img, str) and os.path.exists(img):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return
            results.append(img)

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return

    def process_task(all_steps, async_task, callback, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path, current_task_id,
                     denoising_strength, final_scheduler_name, goals, initial_latent, steps, switch, positive_cond,
                     negative_cond, task, loras, tiled, use_expansion, width, height, base_progress, preparation_steps,
                     total_count, show_intermediate_results, persist_image=True):
        if async_task.last_stop is not False:
            interrupt_processing()
        
        if async_task.task_class in flags.comfy_classes:
            default_params = dict(
                prompt=task["positive"][0],
                negative_prompt=task["negative"][0],
                width=width,
                height=height,
                base_model=async_task.base_model_name,
                sampler=async_task.sampler_name,
                scheduler=final_scheduler_name,
                cfg_scale=async_task.cfg_scale,
                steps=steps,
                denoise=denoising_strength,
                seed=task['task_seed'],
                )
            params_backend = async_task.params_backend.copy()
            input_images = params_backend.pop('input_images', None)
            options = params_backend.pop('ui_options', {})
            reserved_vram = params_backend.pop('reserved_vram', 0)
            if reserved_vram > 0:
                comfyd.modify_variable({"reserved_vram": reserved_vram})
            default_params.update(params_backend)
            try:
                #logger.info(f"ready to get register_cert for {async_task.user_did}")
                user_cert = shared.token.get_register_cert(async_task.user_did)
                if user_cert == "Unknown":
                    raise ValueError(f"User cert is not found: {async_task.user_did}")
                if not shared.token.is_registered(async_task.user_did):
                    raise ValueError(f"User is not registered: {async_task.user_did}")
                comfy_task = get_comfy_task(async_task.user_did, async_task.task_name, async_task.task_method, 
                        default_params, input_images, options)
                if async_task.disable_preview:
                    callback = None
                #logger.info(f"ready to process_flow for {comfy_task.name}")
                imgs = comfypipeline.process_flow(async_task.user_did, comfy_task.name, comfy_task.params, comfy_task.images, callback=callback, total_steps=comfy_task.steps, user_cert=user_cert)
                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]
            except ValueError as e:
                logger.info(f"comfy_task_error: {e}")
                empty_path = [np.zeros((width, height), dtype=np.uint8)]
                imgs = empty_path
                current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * steps)
                yield_result(async_task, empty_path, current_progress, async_task.black_out_nsfw, False,
                     do_not_show_finished_images=not show_intermediate_results or async_task.disable_intermediate_results)
                return imgs, [], current_progress
            logger.info(f'comfypipeline.process_flow finished.')
        else:
            if 'cn' in goals:
                for cn_flag, cn_path in [
                    (flags.cn_canny, controlnet_canny_path),
                    (flags.cn_cpds, controlnet_cpds_path),
                    (flags.cn_pose, controlnet_pose_path)
                ]:
                    for cn_img, cn_stop, cn_weight in async_task.cn_tasks[cn_flag]:
                        positive_cond, negative_cond = core.apply_controlnet(
                            positive_cond, negative_cond,
                            pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)
            imgs = pipeline.process_diffusion(
                positive_cond=positive_cond,
                negative_cond=negative_cond,
                steps=steps,
                switch=switch,
                width=width,
                height=height,
                image_seed=task['task_seed'],
                callback=callback,
                sampler_name=async_task.sampler_name,
                scheduler_name=final_scheduler_name,
                latent=initial_latent,
                denoise=denoising_strength,
                tiled=tiled,
                cfg_scale=async_task.cfg_scale,
                refiner_swap_method=async_task.refiner_swap_method,
                disable_preview=async_task.disable_preview
            )
            del positive_cond, negative_cond  # Save memory
            if inpaint_worker.current_task is not None:
                imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]
            logger.info(f'pipeline.process_diffusion finished.')

        current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * int(all_steps/async_task.image_number))
        if modules.config.default_black_out_nsfw or async_task.black_out_nsfw:
            progressbar(async_task, current_progress, '检测NSFW ...')
            imgs = default_censor(imgs)
        progressbar(async_task, current_progress, f'保存已生成图片 {current_task_id + 1}/{total_count} ...')
        save_and_log(async_task, height, imgs, task, use_expansion, width, loras, goals, persist_image)
        yield_result(async_task, None, current_progress, async_task.black_out_nsfw, False,
                     do_not_show_finished_images=not show_intermediate_results or async_task.disable_intermediate_results)
        
        logger.info(f'The process_task end.')
        return imgs, async_task.img_paths, current_progress

    def apply_patch_settings(async_task):
        patch_settings[pid] = PatchSettings(
            async_task.sharpness,
            async_task.adm_scaler_end,
            async_task.adm_scaler_positive,
            async_task.adm_scaler_negative,
            async_task.controlnet_softness,
            async_task.adaptive_cfg
        )

    def save_and_log(async_task, height, imgs, task, use_expansion, width, loras, goals, persist_image=True):
        for x in imgs:
            scene_task = None
            if hasattr(async_task, 'scene_frontend'):
                scene_task = async_task.task_method
            cn_tasks = [ f'{flags.cn_name_map[n]}-{len(v)}' for n, v in async_task.cn_tasks.items() if len(v)>0]
            cn_tasks = '|'.join(cn_tasks)
            if 'vary' in goals and 'hires.fix' in async_task.uov_method:
                goals = [v if v!='vary' else 'hires.fix' for v in goals]
            goals = [v if v!='cn' else f'cn({cn_tasks})' for v in goals]
            d = [('Prompt', 'prompt', task['log_positive_prompt']),
                 ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                 ('Fooocus V2 Expansion', 'prompt_expansion', task['expansion']),
                 ('Styles', 'styles',
                  str(task['styles'] if not use_expansion else [fooocus_expansion] + task['styles'])),
                 ('Performance', 'performance', async_task.performance_selection.value),
                 ('Steps', 'steps', async_task.steps),
                 ('Resolution', 'resolution', str((width, height))),
                 ('Guidance Scale', 'guidance_scale', async_task.cfg_scale),
                 ('Sharpness', 'sharpness', async_task.sharpness),
                 ('ADM Guidance', 'adm_guidance', str((
                     modules.patch.patch_settings[pid].positive_adm_scale,
                     modules.patch.patch_settings[pid].negative_adm_scale,
                     modules.patch.patch_settings[pid].adm_scaler_end))),
                 ('Base Model', 'base_model', async_task.base_model_name),
                 ('Refiner Model', 'refiner_model', async_task.refiner_model_name),
                 ('Refiner Switch', 'refiner_switch', async_task.refiner_switch)]
            if goals:
                d.append(('Image2Image', 'image2image', ",".join(goals)))
            if scene_task:
                d.append(('Scene Task', 'scene_task', scene_task))

            if async_task.refiner_model_name != 'None':
                if async_task.overwrite_switch > 0:
                    d.append(('Overwrite Switch', 'overwrite_switch', async_task.overwrite_switch))
                if async_task.refiner_swap_method != flags.refiner_swap_method:
                    d.append(('Refiner Swap Method', 'refiner_swap_method', async_task.refiner_swap_method))
            if modules.patch.patch_settings[pid].adaptive_cfg != modules.config.default_cfg_tsnr:
                d.append(
                    ('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[pid].adaptive_cfg))

            if async_task.clip_skip > 1 and async_task.task_class == 'Fooocus':
                d.append(('CLIP Skip', 'clip_skip', async_task.clip_skip))
            d.append(('Sampler', 'sampler', async_task.sampler_name))
            d.append(('Scheduler', 'scheduler', async_task.scheduler_name))
            d.append(('VAE', 'vae', async_task.vae_name))
            d.append(('Seed', 'seed', str(task['task_seed'])))

            if async_task.freeu_enabled:
                d.append(('FreeU', 'freeu',
                          str((async_task.freeu_b1, async_task.freeu_b2, async_task.freeu_s1, async_task.freeu_s2))))

            for li, (n, w) in enumerate(loras):
                if n != 'None' and async_task.task_class in ['Fooocus', 'Kolors', 'Flux']:
                    d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))

            metadata_parser = None
            if async_task.save_metadata_to_images:
                styles_name = task['styles'] if not use_expansion else [fooocus_expansion] + task['styles']
                styles_definition = {k: modules.sdxl_styles.styles[k] for k in styles_name if k and k not in ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp', 'Fooocus Masterpiece', 'Fooocus Photograph', 'Fooocus Negative', 'Fooocus Cinematic']}
                metadata_parser = get_metadata_parser(async_task.metadata_scheme)
                metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                                         task['log_negative_prompt'], task['negative'],
                                         async_task.steps, async_task.base_model_name, async_task.refiner_model_name,
                                         loras, async_task.vae_name, '')
            
            d.append(('Backend Engine', 'backend_engine', f'{async_task.task_class_full}:{async_task.task_method}' if async_task.task_method else async_task.task_class_full))
            d.append(('Metadata Scheme', 'metadata_scheme',
                      async_task.metadata_scheme.value if async_task.save_metadata_to_images else async_task.save_metadata_to_images))
            if not shared.token.is_guest(async_task.user_did):
                d.append(('User', 'created_by', f'{async_task.user_did}'))

            d.append(('Version', 'version', f'{version.branch}_{version.get_simplesdxl_ver()}'))
            image_log(async_task, x, d, metadata_parser, async_task.output_format, task, persist_image, async_task.user_did, async_task.remote_task)
            
        return

    def apply_control_nets(async_task, height, ip_adapter_face_path, ip_adapter_path, width, current_progress):
        for task in async_task.cn_tasks[flags.cn_canny]:
            cn_img, cn_stop, cn_weight = task
            cn_img = HWC3(cn_img)
            if not async_task.skipping_cn_preprocessor:
                cn_img = preprocessors.canny_pyramid(cn_img, async_task.canny_low_threshold,
                                                     async_task.canny_high_threshold)
            else:
                cn_img = preprocessors.normalizedBG(cn_img)
            cn_img = resize_image(HWC3(cn_img), width=width, height=height)
            task[0] = core.numpy_to_pytorch(cn_img) if async_task.task_class in ['Fooocus'] else cn_img
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        for task in async_task.cn_tasks[flags.cn_pose]:
            cn_img, cn_stop, cn_weight = task
            cn_img = HWC3(cn_img)
            if not async_task.skipping_cn_preprocessor:
                if async_task.task_class in ['Fooocus']:
                    cn_img = preprocessors.openpose(cn_img, stick_scaling=True)
                else:
                    cn_img = preprocessors.openpose(cn_img)
            cn_img = resize_image(HWC3(cn_img), width=width, height=height)
            task[0] = core.numpy_to_pytorch(cn_img) if async_task.task_class in ['Fooocus'] else cn_img
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        for task in async_task.cn_tasks[flags.cn_cpds]:
            cn_img, cn_stop, cn_weight = task
            cn_img = HWC3(cn_img)
            if not async_task.skipping_cn_preprocessor and async_task.task_class in ['Fooocus']:
                cn_img = preprocessors.cpds(cn_img)

            cn_img = resize_image(HWC3(cn_img), width=width, height=height)
            task[0] = core.numpy_to_pytorch(cn_img) if async_task.task_class in ['Fooocus'] else cn_img
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        for task in async_task.cn_tasks[flags.cn_ip]:
            cn_img, cn_stop, cn_weight = task
            cn_img = HWC3(cn_img)

            if async_task.task_class in ['Fooocus']:
                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
            else:
                task[0] = resize_image(cn_img, min_side=1024, resize_mode=3)
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        for task in async_task.cn_tasks[flags.cn_ip_face]:
            cn_img, cn_stop, cn_weight = task
            cn_img = HWC3(cn_img)

            if not async_task.skipping_cn_preprocessor and async_task.task_class in ['Fooocus']:
                cn_img = extras.face_crop.crop_image(cn_img)

            if async_task.task_class in ['Fooocus']:
                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=1)
                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
            else:
                task[0] = resize_image(cn_img, width=1024, height=1024, resize_mode=2)
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        all_ip_tasks = async_task.cn_tasks[flags.cn_ip] + async_task.cn_tasks[flags.cn_ip_face]
        if len(all_ip_tasks) > 0 and async_task.task_class in ['Fooocus']:
            pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

    def apply_vary(async_task, uov_method, denoising_strength, uov_input_image, switch, current_progress, advance_progress=False):
        if 'subtle' in uov_method:
            denoising_strength = 0.5
        if 'strong' in uov_method:
            denoising_strength = 0.85
        if 'hires.fix' in uov_method:
            denoising_strength = 0.85
        if async_task.overwrite_vary_strength > 0:
            denoising_strength = async_task.overwrite_vary_strength
        shape_ceil = get_image_shape_ceil(uov_input_image)
        if shape_ceil < 1024:
            logger.info(f'[Vary] Image is resized because it is too small.')
            shape_ceil = 1024
        elif shape_ceil > 2048:
            logger.info(f'[Vary] Image is resized because it is too big.')
            shape_ceil = 2048
        
        if async_task.task_class in flags.comfy_classes:
            H, W, C = uov_input_image.shape
            width = W
            height = H
            initial_latent = None
            return uov_input_image, denoising_strength, initial_latent, width, height, current_progress
        
        uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)
        initial_pixels = core.numpy_to_pytorch(uov_input_image)
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'VAE 编码 ...')
        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=async_task.steps,
            switch=switch,
            denoise=denoising_strength,
            refiner_swap_method=async_task.refiner_swap_method
        )
        initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
        B, C, H, W = initial_latent['samples'].shape
        width = W * 8
        height = H * 8
        logger.info(f'Final resolution is {str((width, height))}.')
        return uov_input_image, denoising_strength, initial_latent, width, height, current_progress

    def apply_inpaint(async_task, initial_latent, inpaint_head_model_path, inpaint_image,
                      inpaint_mask, inpaint_parameterized, denoising_strength, inpaint_respective_field, switch,
                      inpaint_disable_initial_latent, current_progress, skip_apply_outpaint=False,
                      advance_progress=False):
        if not skip_apply_outpaint:
            inpaint_image, inpaint_mask = apply_outpaint(async_task, inpaint_image, inpaint_mask)
            if len(async_task.outpaint_selections) > 0:
                denoising_strength = 1.0

        inpaint_worker.current_task = inpaint_worker.InpaintWorker(
            image=inpaint_image,
            mask=inpaint_mask,
            use_fill=denoising_strength > 0.99,
            k=inpaint_respective_field
        )
        if async_task.debugging_inpaint_preprocessor:
            yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(), 100,
                         async_task.black_out_nsfw, do_not_show_finished_images=True)
            raise EarlyReturnException
        
        if async_task.task_class in flags.comfy_classes:
            height, width = inpaint_image.shape[:2] if len(async_task.outpaint_selections) > 0 else inpaint_worker.current_task.interested_image.shape[:2]
            inpaint_mask = inpaint_worker.current_task.interested_mask
            inpaint_image = inpaint_worker.current_task.interested_image
            return denoising_strength, initial_latent, width, height, current_progress

        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'VAE 重绘编码 ...')
        inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
        inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
        inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)
        candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
            steps=async_task.steps,
            switch=switch,
            denoise=denoising_strength,
            refiner_swap_method=async_task.refiner_swap_method
        )
        latent_inpaint, latent_mask = core.encode_vae_inpaint(
            mask=inpaint_pixel_mask,
            vae=candidate_vae,
            pixels=inpaint_pixel_image)
        latent_swap = None
        if candidate_vae_swap is not None:
            if advance_progress:
                current_progress += 1
            progressbar(async_task, current_progress, 'VAE SD15 编码 ...')
            latent_swap = core.encode_vae(
                vae=candidate_vae_swap,
                pixels=inpaint_pixel_fill)['samples']
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'VAE 编码 ...')
        latent_fill = core.encode_vae(
            vae=candidate_vae,
            pixels=inpaint_pixel_fill)['samples']
        inpaint_worker.current_task.load_latent(
            latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)
        if inpaint_parameterized:
            pipeline.final_unet = inpaint_worker.current_task.patch(
                inpaint_head_model_path=inpaint_head_model_path,
                inpaint_latent=latent_inpaint,
                inpaint_latent_mask=latent_mask,
                model=pipeline.final_unet
            )
        if not inpaint_disable_initial_latent:
            initial_latent = {'samples': latent_fill}
        B, C, H, W = latent_fill.shape
        height, width = H * 8, W * 8
        final_height, final_width = inpaint_worker.current_task.image.shape[:2]
        logger.info(f'Final resolution is {str((final_width, final_height))}, latent is {str((width, height))}.')

        return denoising_strength, initial_latent, width, height, current_progress

    def apply_outpaint(async_task, inpaint_image, inpaint_mask):
        if len(async_task.outpaint_selections) > 0:
            H, W, C = inpaint_image.shape
            if 'top' in async_task.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                      constant_values=255)
            if 'bottom' in async_task.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                      constant_values=255)

            H, W, C = inpaint_image.shape
            if 'left' in async_task.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, 0], [int(W * 0.3), 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(W * 0.3), 0]], mode='constant',
                                      constant_values=255)
            if 'right' in async_task.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(W * 0.3)], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(W * 0.3)]], mode='constant',
                                      constant_values=255)

            inpaint_image = np.ascontiguousarray(inpaint_image.copy())
            inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
            async_task.inpaint_strength = 1.0
            async_task.inpaint_respective_field = 1.0
            logger.info(f'in apply_outpaint: (H, W)=({H}, {W}) -> ({inpaint_image.shape[0]}, {inpaint_image.shape[1]})')
        return inpaint_image, inpaint_mask

    def apply_upscale(async_task, uov_input_image, uov_method, switch, current_progress, advance_progress=False):
        H, W, C = uov_input_image.shape
        if async_task.task_class in flags.comfy_classes:
            width = W
            height = H
            initial_latent = None
            direct_return = False
            tiled = True
            denoising_strength = 0.2 #0.382
            if async_task.overwrite_upscale_strength > 0:
                denoising_strength = async_task.overwrite_upscale_strength
            return direct_return, uov_input_image, denoising_strength, initial_latent, tiled, width, height, current_progress

        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, f'放大图片{str((W, H))} ...')
        uov_input_image = perform_upscale(uov_input_image)
        logger.info(f'Image upscaled.')
        if '1.5x' in uov_method:
            f = 1.5
        elif '2x' in uov_method:
            f = 2.0
        else:
            f = 1.0
        shape_ceil = get_shape_ceil(H * f, W * f)
        if shape_ceil < 1024:
            logger.info(f'[Upscale] Image is resized because it is too small.')
            uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
            shape_ceil = 1024
        else:
            uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)
        image_is_super_large = shape_ceil > 2800
        if 'fast' in uov_method:
            direct_return = True
        elif image_is_super_large:
            logger.info('Image is too large. Directly returned the SR image. '
                  'Usually directly return SR image at 4K resolution '
                  'yields better results than SDXL diffusion.')
            direct_return = True
        else:
            direct_return = False
        if direct_return:
            return direct_return, uov_input_image, None, None, None, None, None, current_progress
        
        tiled = True
        denoising_strength = 0.2 #0.382
        if async_task.overwrite_upscale_strength > 0:
            denoising_strength = async_task.overwrite_upscale_strength

        initial_pixels = core.numpy_to_pytorch(uov_input_image)
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'VAE 编码 ...')
        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=async_task.steps,
            switch=switch,
            denoise=denoising_strength,
            refiner_swap_method=async_task.refiner_swap_method
        )
        initial_latent = core.encode_vae(
            vae=candidate_vae,
            pixels=initial_pixels, tiled=True)
        B, C, H, W = initial_latent['samples'].shape
        width = W * 8
        height = H * 8
        logger.info(f'Final resolution is {str((width, height))}.')
        return direct_return, uov_input_image, denoising_strength, initial_latent, tiled, width, height, current_progress

    def apply_overrides(async_task, steps, height, width):
        if async_task.overwrite_step > 0:
            steps = async_task.overwrite_step
        switch = int(round(async_task.steps * async_task.refiner_switch))
        if async_task.overwrite_switch > 0:
            switch = async_task.overwrite_switch
        if async_task.overwrite_width > 0:
            width = async_task.overwrite_width
        if async_task.overwrite_height > 0:
            height = async_task.overwrite_height
        return steps, switch, width, height

    def process_prompt(async_task, prompt, negative_prompt, base_model_additional_loras, image_number, disable_seed_increment, use_expansion, use_style,
                       use_synthetic_refiner, current_progress, advance_progress=False):
        prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')
        prompt = prompts[0]
        negative_prompt = negative_prompts[0]
        if prompt == '':
            # disable expansion when empty since it is not meaningful and influences image prompt
            use_expansion = False
        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []
        
        lora_filenames = modules.util.remove_performance_lora(modules.config.lora_filenames,
                                                              async_task.performance_selection)
        loras, prompt = parse_lora_references_from_prompt(prompt, async_task.loras,
                                                          modules.config.default_max_lora_number,
                                                          lora_filenames=lora_filenames)
        loras += async_task.performance_loras
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, '准备提示词 ...')
        tasks = []
        task_rng = random.Random(async_task.seed % (constants.MAX_SEED + 1))
        prompt, wildcards_arrays, arrays_mult, seed_fixed = wildcards.compile_arrays(prompt, task_rng)
        for i in range(image_number if arrays_mult==0 else arrays_mult):
            if arrays_mult==0 or not seed_fixed or not disable_seed_increment:
                task_seed = (async_task.seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not
            else:
                task_seed = async_task.seed % (constants.MAX_SEED + 1)

            task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
            task_prompt = wildcards.apply_arrays(prompt, i, wildcards_arrays, arrays_mult)
            task_prompt = wildcards.replace_wildcard(task_prompt, task_rng)
            task_negative_prompt = wildcards.apply_wildcards(negative_prompt, task_rng)
            task_extra_positive_prompts = [wildcards.apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
            task_extra_negative_prompts = [wildcards.apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]
           
            if not async_task.task_method.lower().endswith('_cn') and async_task.task_class not in ['Kolors', 'HyDiT']: 
                task_prompt = minicpm.translate(task_prompt, async_task.translation_methods)
                task_negative_prompt = minicpm.translate(task_negative_prompt, async_task.translation_methods)
                task_extra_positive_prompts = [minicpm.translate(pmt, async_task.translation_methods) for pmt in extra_positive_prompts]
                task_extra_negative_prompts = [minicpm.translate(pmt, async_task.translation_methods) for pmt in extra_negative_prompts]

            positive_basic_workloads = []
            negative_basic_workloads = []

            task_styles = async_task.style_selections.copy()
            if use_style:
                placeholder_replaced = False

                for j, s in enumerate(task_styles):
                    if s == random_style_name:
                        s = get_random_style(task_rng)
                        task_styles[j] = s
                    p, n, style_has_placeholder = apply_style(s, positive=task_prompt)
                    if style_has_placeholder:
                        placeholder_replaced = True
                    positive_basic_workloads = positive_basic_workloads + p
                    negative_basic_workloads = negative_basic_workloads + n
                    logger.info(f'use_style: {s}, positive:{p}, negative:{n}')

                if not placeholder_replaced:
                    positive_basic_workloads = [task_prompt] + positive_basic_workloads
            else:
                positive_basic_workloads.append(task_prompt)
            
            logger.info(f'prompt after use_style: {positive_basic_workloads}')
            negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

            positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
            negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

            positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
            negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

            tasks.append(dict(
                task_seed=task_seed,
                task_prompt=task_prompt,
                task_negative_prompt=task_negative_prompt,
                positive=positive_basic_workloads,
                negative=negative_basic_workloads,
                expansion='',
                c=None,
                uc=None,
                positive_top_k=len(positive_basic_workloads),
                negative_top_k=len(negative_basic_workloads),
                log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                styles=task_styles
            ))
        
        if async_task.task_class not in ['Kolors', 'HyDiT']:
            minicpm.free_model()
        if async_task.task_class in ['Fooocus']:
            if advance_progress:
                current_progress += 1
            progressbar(async_task, current_progress, '加载模型 ...')
            pipeline.refresh_everything(refiner_model_name=async_task.refiner_model_name,
                                    base_model_name=async_task.base_model_name,
                                    loras=loras, base_model_additional_loras=base_model_additional_loras,
                                    use_synthetic_refiner=use_synthetic_refiner, vae_name=async_task.vae_name)
            pipeline.set_clip_skip(async_task.clip_skip)
        else:
            pipeline.reload_expansion()

        if use_expansion:
            if advance_progress:
                current_progress += 1
            for i, t in enumerate(tasks):
                progressbar(async_task, current_progress, f'准备提示文本扩展 #{i + 1} ...')
                expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                logger.info(f'[Prompt Expansion] {expansion}')
                t['expansion'] = expansion
                t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.
        if async_task.task_class in ['Fooocus']:
            if advance_progress:
                current_progress += 1
            for i, t in enumerate(tasks):
                progressbar(async_task, current_progress, f'正向提示词编码 #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])
            if advance_progress:
                current_progress += 1
            for i, t in enumerate(tasks):
                if abs(float(async_task.cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, current_progress, f'负向提示词编码 #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])
        return tasks, use_expansion, loras, current_progress

    def apply_freeu(async_task):
        logger.info(f'FreeU is enabled!')
        pipeline.final_unet = core.apply_freeu(
            pipeline.final_unet,
            async_task.freeu_b1,
            async_task.freeu_b2,
            async_task.freeu_s1,
            async_task.freeu_s2
        )

    def patch_discrete(unet, scheduler_name):
        return core.opModelSamplingDiscrete.patch(unet, scheduler_name, False)[0]

    def patch_edm(unet, scheduler_name):
        return core.opModelSamplingContinuousEDM.patch(unet, scheduler_name, 120.0, 0.002)[0]

    def patch_samplers(async_task):
        final_scheduler_name = async_task.scheduler_name

        if async_task.scheduler_name in ['lcm', 'tcd']:
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = patch_discrete(pipeline.final_unet, async_task.scheduler_name)
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = patch_discrete(pipeline.final_refiner_unet, async_task.scheduler_name)

        elif async_task.scheduler_name == 'edm_playground_v2.5':
            final_scheduler_name = 'karras'
            if pipeline.final_unet is not None:
                pipeline.final_unet = patch_edm(pipeline.final_unet, async_task.scheduler_name)
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = patch_edm(pipeline.final_refiner_unet, async_task.scheduler_name)

        return final_scheduler_name

    def set_hyper_sd_defaults(async_task, current_progress, advance_progress=False):
        logger.info('Enter Hyper-SD 8 step mode.')
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, '下载HyperSD超速模型 ...')
        async_task.performance_loras += [(modules.config.downloading_sdxl_hyper_sd_lora(), 1.0)]
        if async_task.refiner_model_name != 'None':
            logger.info(f'Refiner disabled in Hyper-SD mode.')
        async_task.refiner_model_name = 'None'
        async_task.sampler_name = 'euler'
        async_task.scheduler_name = 'sgm_uniform'
        async_task.sharpness = 0.0
        async_task.cfg_scale = 1.0
        async_task.adaptive_cfg = 1.0
        async_task.refiner_switch = 1.0
        async_task.adm_scaler_positive = 1.0
        async_task.adm_scaler_negative = 1.0
        async_task.adm_scaler_end = 0.0
        return current_progress

    def set_lightning_defaults(async_task, current_progress, advance_progress=False):
        logger.info('Enter Lightning mode.')
        if advance_progress:
            current_progress += 1
        progressbar(async_task, 1, '下载闪电模型 ...')
        async_task.performance_loras += [(modules.config.downloading_sdxl_lightning_lora(), 1.0)]
        if async_task.refiner_model_name != 'None':
            logger.info(f'Refiner disabled in Lightning mode.')
        async_task.refiner_model_name = 'None'
        async_task.sampler_name = 'euler'
        async_task.scheduler_name = 'sgm_uniform'
        async_task.sharpness = 0.0
        async_task.cfg_scale = 1.0
        async_task.adaptive_cfg = 1.0
        async_task.refiner_switch = 1.0
        async_task.adm_scaler_positive = 1.0
        async_task.adm_scaler_negative = 1.0
        async_task.adm_scaler_end = 0.0
        return current_progress

    def set_lcm_defaults(async_task, current_progress, advance_progress=False):
        logger.info('Enter LCM mode.')
        if advance_progress:
            current_progress += 1
        progressbar(async_task, 1, 'Downloading LCM components ...')
        async_task.performance_loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]
        if async_task.refiner_model_name != 'None':
            logger.info(f'Refiner disabled in LCM mode.')
        async_task.refiner_model_name = 'None'
        async_task.sampler_name = 'lcm'
        async_task.scheduler_name = 'lcm'
        async_task.sharpness = 0.0
        async_task.cfg_scale = 1.0
        async_task.adaptive_cfg = 1.0
        async_task.refiner_switch = 1.0
        async_task.adm_scaler_positive = 1.0
        async_task.adm_scaler_negative = 1.0
        async_task.adm_scaler_end = 0.0
        return current_progress

    def apply_image_input(async_task, base_model_additional_loras, clip_vision_path, controlnet_canny_path,
                          controlnet_cpds_path, controlnet_pose_path, goals, inpaint_head_model_path, inpaint_image, inpaint_mask,
                          inpaint_parameterized,  ip_adapter_face_path, ip_adapter_path, ip_negative_path,
                          skip_prompt_processing, use_synthetic_refiner):
        if (async_task.current_tab == 'uov' or (
                async_task.current_tab == 'ip' and async_task.mixing_image_prompt_and_vary_upscale)) \
                and async_task.uov_method != flags.disabled.casefold() and async_task.uov_input_image is not None:
            async_task.uov_input_image, skip_prompt_processing, async_task.steps = prepare_upscale(
                async_task, goals, async_task.uov_input_image, async_task.uov_method, async_task.performance_selection,
                async_task.steps, 1, skip_prompt_processing=skip_prompt_processing)
        if (async_task.current_tab == 'inpaint' or (
                async_task.current_tab == 'ip' and async_task.mixing_image_prompt_and_inpaint)) \
                and isinstance(async_task.inpaint_input_image, dict):
            inpaint_image = async_task.inpaint_input_image['image']
            inpaint_mask = async_task.inpaint_input_image['mask'][:, :, 0]

            if async_task.inpaint_advanced_masking_checkbox:
                if isinstance(async_task.inpaint_mask_image_upload, dict):
                    if (isinstance(async_task.inpaint_mask_image_upload['image'], np.ndarray)
                            and isinstance(async_task.inpaint_mask_image_upload['mask'], np.ndarray)
                            and async_task.inpaint_mask_image_upload['image'].ndim == 3):
                        async_task.inpaint_mask_image_upload = np.maximum(
                            async_task.inpaint_mask_image_upload['image'],
                            async_task.inpaint_mask_image_upload['mask'])
                if isinstance(async_task.inpaint_mask_image_upload,
                              np.ndarray) and async_task.inpaint_mask_image_upload.ndim == 3:
                    H, W, C = inpaint_image.shape
                    async_task.inpaint_mask_image_upload = resample_image(async_task.inpaint_mask_image_upload,
                                                                          width=W, height=H)
                    async_task.inpaint_mask_image_upload = np.mean(async_task.inpaint_mask_image_upload, axis=2)
                    async_task.inpaint_mask_image_upload = (async_task.inpaint_mask_image_upload > 127).astype(
                        np.uint8) * 255
                    inpaint_mask = np.maximum(inpaint_mask, async_task.inpaint_mask_image_upload)

            if int(async_task.inpaint_erode_or_dilate) != 0:
                inpaint_mask = erode_or_dilate(inpaint_mask, async_task.inpaint_erode_or_dilate)

            if async_task.invert_mask_checkbox:
                inpaint_mask = 255 - inpaint_mask

            inpaint_image = HWC3(inpaint_image)
            if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                    and (np.any(inpaint_mask > 127) or len(async_task.outpaint_selections) > 0):
                progressbar(async_task, 1, '下载放大模型 ...')
                modules.config.downloading_upscale_model()
                if inpaint_parameterized:
                    progressbar(async_task, 1, '下载重绘模型 ...')
                    inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                        async_task.inpaint_engine)
                    base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                    logger.info(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                    if async_task.refiner_model_name == 'None':
                        use_synthetic_refiner = True
                        async_task.refiner_switch = 0.8
                else:
                    inpaint_head_model_path, inpaint_patch_model_path = None, None
                    logger.info(f'[Inpaint] Parameterized inpaint is disabled.')
                if async_task.inpaint_additional_prompt != '':
                    if async_task.prompt == '':
                        async_task.prompt = async_task.inpaint_additional_prompt
                    else:
                        async_task.prompt = async_task.inpaint_additional_prompt + '\n' + async_task.prompt
                goals.append('inpaint')
        if async_task.current_tab == 'ip' or \
                async_task.mixing_image_prompt_and_vary_upscale or \
                async_task.mixing_image_prompt_and_inpaint:
            goals.append('cn')
            progressbar(async_task, 1, '下载控制模型 ...')
            if len(async_task.cn_tasks[flags.cn_canny]) > 0:
                controlnet_canny_path = modules.config.downloading_controlnet_canny()
            if len(async_task.cn_tasks[flags.cn_cpds]) > 0:
                controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
            if len(async_task.cn_tasks[flags.cn_pose]) > 0:
                controlnet_pose_path = modules.config.downloading_controlnet_pose()
            if len(async_task.cn_tasks[flags.cn_ip]) > 0:
                clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
            if len(async_task.cn_tasks[flags.cn_ip_face]) > 0:
                clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                    'face')
        if async_task.current_tab == 'enhance' and async_task.enhance_input_image is not None and async_task.task_class in ['Fooocus']:
            goals.append('enhance')
            skip_prompt_processing = True
            async_task.enhance_input_image = HWC3(async_task.enhance_input_image)
        return base_model_additional_loras, clip_vision_path, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path, inpaint_head_model_path, inpaint_image, inpaint_mask, ip_adapter_face_path, ip_adapter_path, ip_negative_path, skip_prompt_processing, use_synthetic_refiner

    def prepare_upscale(async_task, goals, uov_input_image, uov_method, performance, steps, current_progress,
                        advance_progress=False, skip_prompt_processing=False):
        uov_input_image = HWC3(uov_input_image)
        if 'vary' in uov_method:
            goals.append('vary')
        elif 'upscale' in uov_method:
            goals.append('upscale')
            if 'fast' in uov_method and async_task.task_class in ['Fooocus']:
                skip_prompt_processing = True
                steps = 0
            else:
                if async_task.task_class in ['Fooocus']:
                    steps = performance.steps_uov()
            
            if advance_progress:
                current_progress += 1
            progressbar(async_task, current_progress, '下载放大模型 ...')
            modules.config.downloading_upscale_model()
        return uov_input_image, skip_prompt_processing, steps

    def prepare_enhance_prompt(prompt: str, fallback_prompt: str):
        if safe_str(prompt) == '' or len(remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')) == 0:
            prompt = fallback_prompt

        return prompt

    def process_enhance(all_steps, async_task, callback, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path, 
                        current_progress, current_task_id, denoising_strength, inpaint_disable_initial_latent,
                        inpaint_engine, inpaint_respective_field, inpaint_strength,
                        prompt, negative_prompt, final_scheduler_name, goals, height, img, mask,
                        preparation_steps, steps, switch, tiled, total_count, use_expansion, use_style,
                        use_synthetic_refiner, width, show_intermediate_results=True, persist_image=True):
        base_model_additional_loras = []
        inpaint_head_model_path = None
        inpaint_parameterized = inpaint_engine != 'None'  # inpaint_engine = None, improve detail
        initial_latent = None

        prompt = prepare_enhance_prompt(prompt, async_task.prompt)
        negative_prompt = prepare_enhance_prompt(negative_prompt, async_task.negative_prompt)

        if 'vary' in goals:
            img, denoising_strength, initial_latent, width, height, current_progress = apply_vary(
                async_task, async_task.enhance_uov_method, denoising_strength, img, switch, current_progress)
        if 'upscale' in goals:
            direct_return, img, denoising_strength, initial_latent, tiled, width, height, current_progress = apply_upscale(
                async_task, img, async_task.enhance_uov_method, switch, current_progress)
            if direct_return:
                d = [('Upscale (Fast)', 'upscale_fast', '2x')]
                if modules.config.default_black_out_nsfw or async_task.black_out_nsfw:
                    progressbar(async_task, current_progress, '检测NSFW ...')
                    img = default_censor(img)
                progressbar(async_task, current_progress, f'保存已生成图片 {current_task_id + 1}/{total_count} ...')
                image_log(async_task, img, d, output_format=async_task.output_format, persist_image=persist_image, user_did=async_task.user_did, remote_task=async_task.remote_task)
                yield_result(async_task, None, current_progress, async_task.black_out_nsfw, False,
                             do_not_show_finished_images=not show_intermediate_results or async_task.disable_intermediate_results)
                return current_progress, img, prompt, negative_prompt

        if 'inpaint' in goals and inpaint_parameterized:
            progressbar(async_task, current_progress, '下载重绘模型 ...')
            inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                inpaint_engine)
            if inpaint_patch_model_path not in base_model_additional_loras:
                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
        progressbar(async_task, current_progress, '准备增强提示词 ...')
        # positive and negative conditioning aren't available here anymore, process prompt again
        tasks_enhance, use_expansion, loras, current_progress = process_prompt(
            async_task, prompt, negative_prompt, base_model_additional_loras, 1, True,
            use_expansion, use_style, use_synthetic_refiner, current_progress)
        task_enhance = tasks_enhance[0]
        # TODO could support vary, upscale and CN in the future
        # if 'cn' in goals:
        #     apply_control_nets(async_task, height, ip_adapter_face_path, ip_adapter_path, width)
        if async_task.freeu_enabled:
            apply_freeu(async_task)
        patch_samplers(async_task)
        if 'inpaint' in goals:
            denoising_strength, initial_latent, width, height, current_progress = apply_inpaint(
                async_task, None, inpaint_head_model_path, img, mask,
                inpaint_parameterized, inpaint_strength,
                inpaint_respective_field, switch, inpaint_disable_initial_latent,
                current_progress, True)
        imgs, img_paths, current_progress = process_task(all_steps, async_task, callback, controlnet_canny_path,
                                                         controlnet_cpds_path, controlnet_pose_path, current_task_id, denoising_strength,
                                                         final_scheduler_name, goals, initial_latent, steps, switch,
                                                         task_enhance['c'], task_enhance['uc'], task_enhance, loras,
                                                         tiled, use_expansion, width, height, current_progress,
                                                         preparation_steps, total_count, show_intermediate_results,
                                                         persist_image)

        del task_enhance['c'], task_enhance['uc']  # Save memory
        return current_progress, imgs[0], prompt, negative_prompt

    def enhance_upscale(all_steps, async_task, base_progress, callback, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path,
                        current_task_id, denoising_strength, done_steps_inpainting, done_steps_upscaling, enhance_steps,
                        prompt, negative_prompt, final_scheduler_name, height, img, preparation_steps, switch, tiled,
                        total_count, use_expansion, use_style, use_synthetic_refiner, width, persist_image=True):
        # reset inpaint worker to prevent tensor size issues and not mix upscale and inpainting
        inpaint_worker.current_task = None

        current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * (done_steps_upscaling + done_steps_inpainting))
        goals_enhance = []
        img, skip_prompt_processing, steps = prepare_upscale(
            async_task, goals_enhance, img, async_task.enhance_uov_method, async_task.performance_selection,
            enhance_steps, current_progress)
        steps, _, _, _ = apply_overrides(async_task, steps, height, width)
        exception_result = ''
        if len(goals_enhance) > 0:
            try:
                current_progress, img, prompt, negative_prompt = process_enhance(
                    all_steps, async_task, callback, controlnet_canny_path, controlnet_cpds_path,
                    controlnet_pose_path, current_progress, current_task_id, denoising_strength, False,
                    'None', 0.0, 0.0, prompt, negative_prompt, final_scheduler_name,
                    goals_enhance, height, img, None, preparation_steps, steps, switch, tiled, total_count,
                    use_expansion, use_style, use_synthetic_refiner, width, persist_image=persist_image)

            except ldm_patched.modules.model_management.InterruptProcessingException:
                if async_task.last_stop == 'skip':
                    logger.info('User skipped')
                    async_task.last_stop = False
                    # also skip all enhance steps for this image, but add the steps to the progress bar
                    if async_task.enhance_uov_processing_order == flags.enhancement_uov_before:
                        done_steps_inpainting += len(async_task.enhance_ctrls) * enhance_steps
                    exception_result = 'continue'
                else:
                    logger.info('User stopped')
                    exception_result = 'break'
            finally:
                done_steps_upscaling += steps
        return current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task: AsyncTask):
        logger.info(f'Task_class:{async_task.task_class}, Task_name:{async_task.task_name}, Task_method:{async_task.task_method}')
        p2p_remote_process = ads.get_admin_default("p2p_remote_process")
        remote_process = True if p2p_remote_process is not None and p2p_remote_process.lower() == 'out' else False
        preparation_start_time = time.perf_counter()
        async_task.processing = True
        if remote_process:
            async_task.method = 'generate_image'
            logger.info(f'Start to remote generate image request...')
            qsize = p2p_task.request_p2p_task(async_task)
            if qsize != "error":
                logger.info(f'Waiting the remote response: task_id={async_task.task_id}, qsize={qsize}')
            else:
                logger.info(f'Remote process request: task_id={async_task.task_id}, error')
                stop_processing(async_task, 0, "Remote process request error!")
            while async_task.processing:
                time.sleep(2)
            return
        if is_models_file_absent(async_task.preset):
            stop_processing(async_task, 0, "Model absent")
            return
        ldm_patched.modules.model_management.print_memory_info("begin at handler")
        async_task.outpaint_selections = [o.lower() for o in async_task.outpaint_selections]
        base_model_additional_loras = []
        async_task.uov_method = async_task.uov_method.casefold()
        async_task.enhance_uov_method = async_task.enhance_uov_method.casefold()

        if fooocus_expansion in async_task.style_selections:
            use_expansion = True
            async_task.style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(async_task.style_selections) > 0

        if async_task.base_model_name == async_task.refiner_model_name:
            logger.info(f'Refiner disabled because base model and refiner are same.')
            async_task.refiner_model_name = 'None'

        current_progress = 0
        if async_task.performance_selection == Performance.EXTREME_SPEED:
            set_lcm_defaults(async_task, current_progress, advance_progress=True)
        elif async_task.performance_selection == Performance.LIGHTNING:
            set_lightning_defaults(async_task, current_progress, advance_progress=True)
        elif async_task.performance_selection == Performance.HYPER_SD:
            set_hyper_sd_defaults(async_task, current_progress, advance_progress=True)

        if async_task.task_class in ['Fooocus']:
            logger.info(f'[Parameters] Adaptive CFG = {async_task.adaptive_cfg}')
            logger.info(f'[Parameters] CLIP Skip = {async_task.clip_skip}')
            logger.info(f'[Parameters] Sharpness = {async_task.sharpness}')
            logger.info(f'[Parameters] ControlNet Softness = {async_task.controlnet_softness}')
            logger.info(f'[Parameters] ADM Scale = '
                  f'{async_task.adm_scaler_positive} : '
                  f'{async_task.adm_scaler_negative} : '
                  f'{async_task.adm_scaler_end}')
            logger.info(f'[Parameters] Seed = {async_task.seed}')

        apply_patch_settings(async_task)

        if async_task.task_class in ['Fooocus']:
            logger.info(f'[Parameters] CFG = {async_task.cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = async_task.aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        width, height = int(width), int(height)
        if async_task.task_class in ['Fooocus']:
            logger.info(f'[Parameters] Aspect Ratios = {width}×{height}')

        skip_prompt_processing = False

        inpaint_worker.current_task = None
        inpaint_parameterized = async_task.inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        controlnet_pose_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        goals = []
        tasks = []
        current_progress = 1

        if async_task.input_image_checkbox:
            base_model_additional_loras, clip_vision_path, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path, inpaint_head_model_path, inpaint_image, inpaint_mask, ip_adapter_face_path, ip_adapter_path, ip_negative_path, skip_prompt_processing, use_synthetic_refiner = apply_image_input(
                async_task, base_model_additional_loras, clip_vision_path, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path,
                goals, inpaint_head_model_path, inpaint_image, inpaint_mask, inpaint_parameterized, ip_adapter_face_path,
                ip_adapter_path, ip_negative_path, skip_prompt_processing, use_synthetic_refiner)

        if async_task.task_class in ['Fooocus']:
            # Load or unload CNs
            progressbar(async_task, current_progress, '加载控制模型 ...')
            pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path])
            ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
            ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        async_task.steps, switch, width, height = apply_overrides(async_task, async_task.steps, height, width)

        if async_task.task_class in ['Fooocus']:
            logger.info(f'[Parameters] Sampler = {async_task.sampler_name} - {async_task.scheduler_name}')
            logger.info(f'[Parameters] Steps = {async_task.steps} - {switch}')

        progressbar(async_task, current_progress, '生图初始化 ...')

        loras = async_task.loras
        if not skip_prompt_processing:
            tasks, use_expansion, loras, current_progress = process_prompt(async_task, async_task.prompt, async_task.negative_prompt,
                                                         base_model_additional_loras, async_task.image_number,
                                                         async_task.disable_seed_increment, use_expansion, use_style,
                                                         use_synthetic_refiner, current_progress, advance_progress=True)
        
        if async_task.task_class in flags.comfy_classes:
            logger.info(f'Enable Comfyd backend.')
            # if "flux_aio" in async_task.task_method and \
            #     (((async_task.current_tab in ['uov', 'inpaint', 'ip'] \
            #             and len(async_task.cn_tasks[flags.cn_pose]) > 0 \
            #             and len(async_task.cn_tasks[flags.cn_ip_face]) == 0) \
            #         or ((async_task.current_tab in ['uov'] and 'hires.fix' in async_task.uov_method) \
            #             and ((len(async_task.cn_tasks[flags.cn_ip_face]) == 0 ) or not async_task.mixing_image_prompt_and_vary_upscale))) \
            #     or (len(async_task.cn_tasks[flags.cn_canny]) > 0 and len(async_task.cn_tasks[flags.cn_cpds]) > 0)):
            #     logger.info(f'Clean the model cache in comfyd.')
            #     comfyd.stop()
            comfyd.start()
        else:
            logger.info(f'Enable Fooocus backend.')
            comfyd.stop()
            pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path])

        if len(goals) > 0:
            current_progress += 1
            progressbar(async_task, current_progress, '图片处理中 ...')
            logger.info(f'Preprocess the image for {",".join(goals)}.')

        should_enhance = async_task.enhance_checkbox and (async_task.enhance_uov_method != flags.disabled.casefold() or len(async_task.enhance_ctrls) > 0)
        
        if 'vary' in goals:
            async_task.uov_input_image, denoising_strength, initial_latent, width, height, current_progress = apply_vary(
                async_task, async_task.uov_method, denoising_strength, async_task.uov_input_image, switch,
                current_progress)

        if 'upscale' in goals:
            direct_return, async_task.uov_input_image, denoising_strength, initial_latent, tiled, width, height, current_progress = apply_upscale(
                async_task, async_task.uov_input_image, async_task.uov_method, switch, current_progress,
                advance_progress=True)
            if direct_return:
                d = [('Upscale (Fast)', 'upscale_fast', '2x')]
                if modules.config.default_black_out_nsfw or async_task.black_out_nsfw:
                    progressbar(async_task, 100, '检测NSFW ...')
                    async_task.uov_input_image = default_censor(async_task.uov_input_image)
                progressbar(async_task, 100, '保存已生成图片 ...')
                image_log(async_task, async_task.uov_input_image, d, output_format=async_task.output_format, user_did=async_task.user_did, remote_task=async_task.remote_task)
                yield_result(async_task, None, 100, async_task.black_out_nsfw, False,
                             do_not_show_finished_images=True)
                return
        
        if 'inpaint' in goals:
            try:
                denoising_strength, initial_latent, width, height, current_progress = apply_inpaint(async_task,
                                                                                                    initial_latent,
                                                                                                    inpaint_head_model_path,
                                                                                                    inpaint_image,
                                                                                                    inpaint_mask,
                                                                                                    inpaint_parameterized,
                                                                                                    async_task.inpaint_strength,
                                                                                                    async_task.inpaint_respective_field,
                                                                                                    switch,
                                                                                                    async_task.inpaint_disable_initial_latent,
                                                                                                    current_progress,
                                                                                                    advance_progress=True)
            except EarlyReturnException:
                return
        
        
        if 'cn' in goals:
            apply_control_nets(async_task, height, ip_adapter_face_path, ip_adapter_path, width, current_progress)
            if async_task.debugging_cn_preprocessor:
                return

        if async_task.freeu_enabled:
            apply_freeu(async_task)

        # async_task.steps can have value of uov steps here when upscale has been applied
        steps, _, _, _ = apply_overrides(async_task, async_task.steps, height, width)

        images_to_enhance = []
        if 'enhance' in goals:
            async_task.image_number = 1
            images_to_enhance += [async_task.enhance_input_image]
            height, width, _ = async_task.enhance_input_image.shape
            # input image already provided, processing is skipped
            steps = 0
            yield_result(async_task, async_task.enhance_input_image, current_progress, async_task.black_out_nsfw, False,
                         async_task.disable_intermediate_results)

        all_steps = steps * async_task.image_number

        if async_task.enhance_checkbox and async_task.enhance_uov_method != flags.disabled.casefold():
            enhance_upscale_steps = async_task.performance_selection.steps()
            if 'upscale' in async_task.enhance_uov_method:
                if 'fast' in async_task.enhance_uov_method:
                    enhance_upscale_steps = 0
                else:
                    enhance_upscale_steps = async_task.performance_selection.steps_uov()
            enhance_upscale_steps, _, _, _ = apply_overrides(async_task, enhance_upscale_steps, height, width)
            enhance_upscale_steps_total = async_task.image_number * enhance_upscale_steps
            all_steps += enhance_upscale_steps_total

        if async_task.enhance_checkbox and len(async_task.enhance_ctrls) != 0:
            enhance_steps, _, _, _ = apply_overrides(async_task, async_task.original_steps, height, width)
            all_steps += async_task.image_number * len(async_task.enhance_ctrls) * enhance_steps

        all_steps = max(all_steps, 1)

        if async_task.task_class in ['Fooocus']:
            logger.info(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        if async_task.task_class in ['Fooocus']:
            logger.info(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - preparation_start_time
        logger.info(f'Preparation time: {preparation_time:.2f} seconds')

        final_scheduler_name = patch_samplers(async_task)
        logger.info(f'Using {final_scheduler_name} scheduler.')

        if async_task.task_class in ['Fooocus']:
            progressbar(async_task, current_progress, '模型移动到GPU ...')
        else:
            progressbar(async_task, current_progress, f'{async_task.task_class} 生图 ...')

        processing_start_time = int(time.time() * 1000)

        preparation_steps = current_progress
        total_count = async_task.image_number

        def callback(step, x0, x, total_steps, y):
            if step == 0:
                async_task.callback_steps = 0
            async_task.callback_steps += (100 - preparation_steps) / float(all_steps)
            percentage = int(current_progress + async_task.callback_steps)
            progressbar(async_task, percentage, f'采样步数 {step + 1}/{total_steps}, 图片 {current_task_id + 1}/{total_count} ...', y)

        def callback_comfytask(step, total_steps, y):
            if step == 1:
                async_task.callback_steps = 0
            async_task.callback_steps += (100 - preparation_steps) / float(all_steps)
            if async_task.task_method == 'sd15_aio' and step % 4 != 1 and step <= 30:
                return
            if async_task.task_method in ('il_v_pre', 'il_v_pre_aio') and step % 2 != 1 and step <= 30:
                return
            percentage = int(current_progress + async_task.callback_steps)
            progressbar(async_task, percentage, f'采样步数 {step}/{total_steps}, 图片 {current_task_id + 1}/{total_count} ...', y)

        callback_function = callback
        i2i_uov_hires_fix_blurred = async_task.params_backend.pop('hires_fix_blurred', 0.0)
        i2i_uov_hires_fix_w = async_task.params_backend.pop('hires_fix_weight', 0.5)
        i2i_uov_hires_fix_s = async_task.params_backend.pop('hires_fix_stop', 0.8)

        if async_task.task_class not in ['Fooocus']:
            pipeline.free_everything()
            #ldm_patched.modules.model_management.unload_and_free_everything()
            async_task.refiner_model_name = ''
            async_task.refiner_switch = 1.0
            callback_function = callback_comfytask
            
            input_images = None
            if async_task.layer_input_image is not None:
                input_images = comfypipeline.ComfyInputImage([])
                input_images.set_image('input_image', HWC3(async_task.layer_input_image))
            if "scene_" in async_task.task_method:
                input_images = comfypipeline.ComfyInputImage([])
                if async_task.scene_input_image1 is not None:
                    input_images.set_image('i2i_ip_image1', async_task.scene_input_image1)
                    if "happy_cn" in async_task.task_method and preprocessors.openpose_have(async_task.scene_input_image1, ['face']):
                        async_task.params_backend['i2i_ip_fn1'] = 4
                if async_task.scene_canvas_image is not None:
                    canvas_image = async_task.scene_canvas_image['image']
                    canvas_mask = async_task.scene_canvas_image['mask']
                    input_images.set_image('i2i_inpaint_image', canvas_image)
                    input_images.set_image('i2i_inpaint_mask', canvas_mask)
                if async_task.scene_steps is not None:
                    async_task.steps = async_task.scene_steps
                    #all_steps = async_task.steps * async_task.image_number
                    async_task.params_backend['display_steps'] = async_task.steps
                if async_task.task_method.lower().endswith('_cn'):
                    async_task.steps = async_task.steps * 3
                    #all_steps = async_task.steps * async_task.image_number
                async_task.params_backend['additional_prompt'] = async_task.scene_additional_prompt
            if "_aio" in async_task.task_method:
                input_images = comfypipeline.ComfyInputImage([])
                if '.gguf' in async_task.base_model_name:
                    async_task.params_backend['base_model_gguf'] = async_task.base_model_name
                else:
                    if async_task.task_class == 'Flux':
                        async_task.params_backend['base_model_dtype'] = 'default'
                if 'cn' in goals:
                    async_task.params_backend['i2i_function'] = 1 # image prompt
                    if async_task.skipping_cn_preprocessor:
                        async_task.params_backend['i2i_skip_preprocessors'] = True
                    def push_cn_task(i, task, cn_type):
                        cn_img, cn_stop, cn_weight = task
                        input_images.set_image(f'i2i_ip_image{i}', cn_img)
                        async_task.params_backend[f'i2i_ip_fn{i}'] = cn_type
                        async_task.params_backend[f'i2i_ip_fn{i}_w'] = cn_weight
                        async_task.params_backend[f'i2i_ip_fn{i}_s'] = cn_stop
                        return 

                    i = 1
                    for task in async_task.cn_tasks[flags.cn_ip]:
                        if i > 4:
                            break
                        push_cn_task(i, task, 1)
                        i = i+1
                    for task in async_task.cn_tasks[flags.cn_canny]:
                        if i > 4:
                            break
                        push_cn_task(i, task, 2)
                        i = i+1
                    for task in async_task.cn_tasks[flags.cn_cpds]:
                        if i > 4:
                            break
                        push_cn_task(i, task, 3)
                        i = i+1
                    for task in async_task.cn_tasks[flags.cn_ip_face]:
                        if i > 4:
                            break
                        if 'inpaint' in goals and async_task.task_class == 'Flux':
                            cn_img, cn_stop, cn_weight = task
                            task = cn_img, cn_stop, cn_weight*2
                        push_cn_task(i, task, 4)
                        i = i+1
                    for task in async_task.cn_tasks[flags.cn_pose]:
                        if i > 4:
                            break
                        push_cn_task(i, task, 5)
                        i = i+1
                if 'vary' in goals or 'upscale' in goals:
                    async_task.params_backend['i2i_function'] = 2 # iamge upscale and vary
                    input_images.set_image(f'i2i_uov_image', async_task.uov_input_image)
                    tiled_block = 1024 if (async_task.task_class == 'Comfy' and async_task.task_method == 'sd15_aio') else 2048
                    tiled_size = lambda x, p: int(x*p+16) if int(x*p) < tiled_block else int(int(x*p)/math.ceil(int(x*p)/tiled_block))+16
                    tiled_steps = [10, 6, 4]
                    match = re.search(r'\((?:fast )?([\d.]+)x\)', async_task.uov_method)
                    match_multiple = 1.0 if not match else float(match.group(1))
                    match_multiple = match_multiple if match_multiple<4.0 else 4.0
                    if 'vary' in async_task.uov_method and 'subtle' in async_task.uov_method:
                        async_task.params_backend['i2i_uov_fn'] = 2
                    elif 'vary' in async_task.uov_method and 'strong' in async_task.uov_method:
                        async_task.params_backend['i2i_uov_fn'] = 3
                    elif 'upscale' in async_task.uov_method and 'fast' in async_task.uov_method and match:
                        async_task.params_backend['i2i_uov_fn'] = 1
                        async_task.params_backend['i2i_uov_multiple'] = match_multiple
                        width = int(width * match_multiple)
                        height = int(height * match_multiple)
                        tasks = tasks[:1]
                    elif 'upscale' in async_task.uov_method and match:
                        async_task.params_backend['i2i_uov_fn'] = 4
                        async_task.params_backend['i2i_uov_multiple'] = match_multiple
                        async_task.params_backend['i2i_uov_tiled_width'] = tiled_size(width, match_multiple)
                        async_task.params_backend['i2i_uov_tiled_height'] = tiled_size(height, match_multiple)
                        width = int(width * match_multiple)
                        height = int(height * match_multiple)
                        if async_task.task_class == 'Flux':
                            async_task.params_backend['i2i_uov_tiled_steps'] = tiled_steps[1 if 'gguf' in async_task.base_model_name else 0]
                        else:
                            async_task.params_backend['i2i_uov_tiled_steps'] = int(async_task.steps * 0.6)
                        async_task.steps = async_task.params_backend['i2i_uov_tiled_steps'] * math.ceil(width/(async_task.params_backend['i2i_uov_tiled_width'])) * math.ceil(height/(async_task.params_backend['i2i_uov_tiled_height']))
                        #all_steps = async_task.steps * async_task.image_number
                    elif 'hires.fix' in async_task.uov_method:
                        async_task.params_backend['i2i_uov_fn'] = 5
                        async_task.params_backend['i2i_uov_hires_fix_blurred'] = i2i_uov_hires_fix_blurred
                        async_task.params_backend['i2i_uov_hires_fix_w'] = i2i_uov_hires_fix_w
                        async_task.params_backend['i2i_uov_hires_fix_s'] = i2i_uov_hires_fix_s
                    else:
                        async_task.params_backend['i2i_uov_fn'] = 0
                    async_task.params_backend['i2i_uov_is_mix_ip'] = True if 'cn' in goals else False
                if 'inpaint' in goals:
                    async_task.params_backend['i2i_inpaint_version'] = async_task.inpaint_engine
                    async_task.params_backend['i2i_function'] = 3 # image inpaint
                    input_images.set_image(f'i2i_inpaint_image', inpaint_worker.current_task.interested_image)
                    input_images.set_image(f'i2i_inpaint_mask', inpaint_worker.current_task.interested_mask)
                    inpaint_engine_model_index = f'{async_task.task_method}_{async_task.inpaint_engine}'
                    if inpaint_engine_model_index in flags.inpaint_engine_model_names:
                        async_task.base_model_name = flags.inpaint_engine_model_names[inpaint_engine_model_index]
                        if async_task.task_class == 'Flux':
                            if 'gguf' in async_task.base_model_name:
                                async_task.params_backend['base_model_gguf'] = async_task.base_model_name
                            else:
                                async_task.params_backend.pop('base_model_gguf', None)
                    if async_task.invert_mask_checkbox:
                        async_task.params_backend['i2i_inpaint_is_invert_mask'] = True
                    if 'cn' in goals:
                        async_task.params_backend['i2i_inpaint_is_mix_ip'] = True
                    if async_task.task_class == 'Flux':
                        async_task.cfg_scale = 30
                    if len(async_task.outpaint_selections)>0:
                        async_task.params_backend['i2i_inpaint_fn'] = 1  # out
                    else:
                        async_task.params_backend['i2i_inpaint_fn'] = 2 # detail, object, general
                if async_task.task_class == 'Flux':
                    async_task.params_backend['i2i_model_type'] = 2 if 'gguf' in async_task.base_model_name else 1
                if async_task.task_class == 'Comfy':
                    if 'i2i_uov_tiled_steps' not in async_task.params_backend and async_task.task_method == "sd15_aio":
                        async_task.params_backend['display_steps'] = int((30 if async_task.steps==-1 else async_task.steps) * 1.6)
                    else:
                        async_task.params_backend['display_steps'] = async_task.steps
                    #all_steps = async_task.params_backend['display_steps'] * async_task.image_number
                elif async_task.task_class == 'Kolors':
                    async_task.params_backend['display_steps'] = async_task.steps + 1
                    #all_steps = async_task.params_backend['display_steps'] * async_task.image_number
            if 'display_steps' not in async_task.params_backend:
                async_task.params_backend['display_steps'] = 30 if async_task.steps==-1 else async_task.steps
            all_steps = async_task.params_backend['display_steps'] * async_task.image_number
            async_task.params_backend['input_images'] = input_images

        ldm_patched.modules.model_management.print_memory_info("begin to process_task")

        show_intermediate_results = len(tasks) > 1 or async_task.should_enhance
        persist_image = not async_task.should_enhance or not async_task.save_final_enhanced_image_only

        for current_task_id, task in enumerate(tasks):
            progressbar(async_task, current_progress, f'进入 {async_task.task_class} 生图 {current_task_id + 1}/{async_task.image_number} ...')
            execution_start_time = time.perf_counter()

            try:
                imgs, img_paths, current_progress = process_task(all_steps, async_task, callback_function, controlnet_canny_path,
                                                                 controlnet_cpds_path, controlnet_pose_path, current_task_id,
                                                                 denoising_strength, final_scheduler_name, goals,
                                                                 initial_latent, async_task.steps, switch, task['c'],
                                                                 task['uc'], task, loras, tiled, use_expansion, width,
                                                                 height, current_progress, preparation_steps,
                                                                 async_task.image_number, show_intermediate_results,
                                                                 persist_image)

                current_progress = int(preparation_steps + (100 - preparation_steps) / float(all_steps) * int(all_steps/async_task.image_number) * (current_task_id + 1))
                images_to_enhance += imgs

            except ldm_patched.modules.model_management.InterruptProcessingException:
                if async_task.last_stop == 'skip':
                    if async_task.task_class in ['Fooocus']:
                        del task['c'], task['uc']  # Save memory
                    logger.info(f'User skipped')
                    async_task.last_stop = False
                    continue
                else:
                    logger.info('User stopped')
                    break

            if async_task.task_class in ['Fooocus']:
                del task['c'], task['uc']  # Save memory
            execution_time = time.perf_counter() - execution_start_time
            logger.info(f'Generating and saving time: {execution_time:.2f} seconds')
            ldm_patched.modules.model_management.print_memory_info("process_task finished")


        if not async_task.should_enhance:
            logger.info(f'[Enhance] Skipping, preconditions aren\'t met')
            stop_processing(async_task, processing_start_time)
            return

        progressbar(async_task, current_progress, '图像增强处理中...')

        active_enhance_tabs = len(async_task.enhance_ctrls)
        should_process_enhance_uov = async_task.enhance_uov_method != flags.disabled.casefold()
        enhance_uov_before = False
        enhance_uov_after = False
        if should_process_enhance_uov:
            active_enhance_tabs += 1
            enhance_uov_before = async_task.enhance_uov_processing_order == flags.enhancement_uov_before
            enhance_uov_after = async_task.enhance_uov_processing_order == flags.enhancement_uov_after
        total_count = len(images_to_enhance) * active_enhance_tabs
        async_task.images_to_enhance_count = len(images_to_enhance)

        base_progress = current_progress
        current_task_id = -1
        done_steps_upscaling = 0
        done_steps_inpainting = 0
        enhance_steps, _, _, _ = apply_overrides(async_task, async_task.original_steps, height, width)
        exception_result = None
        for index, img in enumerate(images_to_enhance):
            async_task.enhance_stats[index] = 0
            enhancement_image_start_time = time.perf_counter()

            last_enhance_prompt = async_task.prompt
            last_enhance_negative_prompt = async_task.negative_prompt

            if enhance_uov_before:
                current_task_id += 1
                persist_image = not async_task.save_final_enhanced_image_only or active_enhance_tabs == 0
                current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result = enhance_upscale(
                    all_steps, async_task, base_progress, callback, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path,
                    current_task_id, denoising_strength, done_steps_inpainting, done_steps_upscaling, enhance_steps,
                    async_task.prompt, async_task.negative_prompt, final_scheduler_name, height, img, preparation_steps,
                    switch, tiled, total_count, use_expansion, use_style, use_synthetic_refiner, width, persist_image)
                async_task.enhance_stats[index] += 1

                if exception_result == 'continue':
                    continue
                elif exception_result == 'break':
                    break

            # inpaint for all other tabs
            for enhance_mask_dino_prompt_text, enhance_prompt, enhance_negative_prompt, enhance_mask_model, enhance_mask_cloth_category, enhance_mask_sam_model, enhance_mask_text_threshold, enhance_mask_box_threshold, enhance_mask_sam_max_detections, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine, enhance_inpaint_strength, enhance_inpaint_respective_field, enhance_inpaint_erode_or_dilate, enhance_mask_invert in async_task.enhance_ctrls:
                current_task_id += 1
                current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * (done_steps_upscaling + done_steps_inpainting))
                progressbar(async_task, current_progress, f'准备增强生图 {current_task_id + 1}/{total_count} ...')
                enhancement_task_start_time = time.perf_counter()
                is_last_enhance_for_image = (current_task_id + 1) % active_enhance_tabs == 0 and not enhance_uov_after
                persist_image = not async_task.save_final_enhanced_image_only or is_last_enhance_for_image

                if async_task.task_class not in ['Kolors', 'HyDiT']:
                    enhance_mask_dino_prompt_text = minicpm.translate(enhance_mask_dino_prompt_text, async_task.translation_methods)
                    enhance_prompt = minicpm.translate(enhance_prompt, async_task.translation_methods)
                    enhance_negative_prompt = minicpm.translate(enhance_negative_prompt, async_task.translation_methods)
                    minicpm.free_model()

                extras = {}
                if enhance_mask_model == 'sam':
                    logger.info(f'[Enhance] Searching for "{enhance_mask_dino_prompt_text}"')
                elif enhance_mask_model == 'u2net_cloth_seg':
                    extras['cloth_category'] = enhance_mask_cloth_category

                mask, dino_detection_count, sam_detection_count, sam_detection_on_mask_count = generate_mask_from_image(
                    img, mask_model=enhance_mask_model, extras=extras, sam_options=SAMOptions(
                        dino_prompt=enhance_mask_dino_prompt_text,
                        dino_box_threshold=enhance_mask_box_threshold,
                        dino_text_threshold=enhance_mask_text_threshold,
                        dino_erode_or_dilate=async_task.dino_erode_or_dilate,
                        dino_debug=async_task.debugging_dino,
                        max_detections=enhance_mask_sam_max_detections,
                        model_type=enhance_mask_sam_model,
                    ))
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]

                if int(enhance_inpaint_erode_or_dilate) != 0:
                    mask = erode_or_dilate(mask, enhance_inpaint_erode_or_dilate)

                if enhance_mask_invert:
                    mask = 255 - mask

                if async_task.debugging_enhance_masks_checkbox:
                    progressbar(async_task, current_progress, 'Loading ...', mask) 
                    yield_result(async_task, mask, current_progress, async_task.black_out_nsfw, False,
                                 async_task.disable_intermediate_results)
                    async_task.enhance_stats[index] += 1

                logger.info(f'[Enhance] {dino_detection_count} boxes detected')
                logger.info(f'[Enhance] {sam_detection_count} segments detected in boxes')
                logger.info(f'[Enhance] {sam_detection_on_mask_count} segments applied to mask')

                if enhance_mask_model == 'sam' and (dino_detection_count == 0 or not async_task.debugging_dino and sam_detection_on_mask_count == 0):
                    logger.info(f'[Enhance] No "{enhance_mask_dino_prompt_text}" detected, skipping')
                    continue

                goals_enhance = ['inpaint']

                try:
                    current_progress, img, enhance_prompt_processed, enhance_negative_prompt_processed = process_enhance(
                        all_steps, async_task, callback, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path,
                        current_progress, current_task_id, denoising_strength, enhance_inpaint_disable_initial_latent,
                        enhance_inpaint_engine, enhance_inpaint_respective_field, enhance_inpaint_strength,
                        enhance_prompt, enhance_negative_prompt, final_scheduler_name, goals_enhance, height, img, mask,
                        preparation_steps, enhance_steps, switch, tiled, total_count, use_expansion, use_style,
                        use_synthetic_refiner, width, persist_image=persist_image)
                    async_task.enhance_stats[index] += 1

                    if (should_process_enhance_uov and async_task.enhance_uov_processing_order == flags.enhancement_uov_after
                            and async_task.enhance_uov_prompt_type == flags.enhancement_uov_prompt_type_last_filled):
                        if enhance_prompt_processed != '':
                            last_enhance_prompt = enhance_prompt_processed
                        if enhance_negative_prompt_processed != '':
                            last_enhance_negative_prompt = enhance_negative_prompt_processed

                except ldm_patched.modules.model_management.InterruptProcessingException:
                    if async_task.last_stop == 'skip':
                        logger.info('User skipped')
                        async_task.last_stop = False
                        continue
                    else:
                        logger.info('User stopped')
                        exception_result = 'break'
                        break
                finally:
                    done_steps_inpainting += enhance_steps

                enhancement_task_time = time.perf_counter() - enhancement_task_start_time
                logger.info(f'Enhancement time: {enhancement_task_time:.2f} seconds')

            if exception_result == 'break':
                break

            if enhance_uov_after:
                current_task_id += 1
                # last step in enhance, always save
                persist_image = True
                current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result = enhance_upscale(
                    all_steps, async_task, base_progress, callback, controlnet_canny_path, controlnet_cpds_path, controlnet_pose_path,
                    current_task_id, denoising_strength, done_steps_inpainting, done_steps_upscaling, enhance_steps,
                    last_enhance_prompt, last_enhance_negative_prompt, final_scheduler_name, height, img,
                    preparation_steps, switch, tiled, total_count, use_expansion, use_style, use_synthetic_refiner,
                    width, persist_image)
                async_task.enhance_stats[index] += 1
                
                if exception_result == 'continue':
                    continue
                elif exception_result == 'break':
                    break

            enhancement_image_time = time.perf_counter() - enhancement_image_start_time
            logger.info(f'Enhancement image time: {enhancement_image_time:.2f} seconds')

        stop_processing(async_task, processing_start_time)
        return

    worker.yield_result = yield_result
    worker.progressbar = progressbar
    worker.p2p_save_and_log = p2p_save_and_log
    worker.stop_processing = stop_processing
    worker.interrupt_processing = interrupt_processing

    last_active = time.time()
    while True:
        try:
            try:
                task = async_tasks.get(block=True, timeout=0.5)
            except queue.Empty:
                time.sleep(0.01)
                continue
            with processing_lock:
                worker_processing = task.task_id

            logger.info(f'Got async_tasks: {task.task_id}')
            try:
                handler(task)
                if task.generate_image_grid:
                    build_image_wall(task)
                task.yields.append(['finish', task.results])
                if task.task_class not in flags.comfy_classes and not args_manager.args.disable_backend:
                    pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                task.yields.append(['finish', task.results])
            finally:
                async_tasks.task_done()
                p2p_task.gc_p2p_task()
                with processing_lock:
                    pending_tasks -= 1
                    worker_processing = None
                if pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[pid]
        except Exception as e:
            logger.error(f"Worker loop error: {str(e)}", exc_info=True)
            time.sleep(1)
        if not get_echo_off() and time.time() - last_active > 10:
            logger.info("Worker is alive and waiting...")
            last_active = time.time()
    logger.info("Unexpected exit in worker thread")

def add_task(task):
    global pending_tasks, async_tasks
    async_tasks.put(task)
    with processing_lock:
        pending_tasks += 1

def get_task_size():
    global pending_tasks
    with processing_lock:
        return pending_tasks

def get_processing_id():
    global worker_processing
    with processing_lock:
        return worker_processing

def restart(task):
    global thread, async_tasks, restart_lock

    if not restart_lock.acquire(blocking=False):
        print("另一个线程正在执行重启操作，跳过此次重启请求")
        return

    try:
        if hasattr(thread, '_restarting') and thread._restarting:
            print("线程已经在重启过程中，跳过此次重启请求")
            return
            
        thread._restarting = True
        
        if thread.is_alive():
            print("Thread is unresponsive! Restarting...")
            stop_event.set()
            thread.join(timeout=2)
            stop_event.clear()
        else:
            print("Thread is not alive! Restarting...")
        
        initialize_worker_resources()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        
        thread._restarting = False
    finally:
        restart_lock.release()

def initialize_worker_resources():
    global async_tasks, worker_processing, processing_lock, pending_tasks
    async_tasks = queue.Queue()  
    worker_processing = None  
    processing_lock = threading.Lock()
    pending_tasks = 0

async_tasks = queue.Queue()
worker_processing = None
processing_lock = threading.Lock()
pending_tasks = 0
restart_lock = threading.Lock()
stop_event = threading.Event()

thread = threading.Thread(target=worker, daemon=True)
thread.start()

