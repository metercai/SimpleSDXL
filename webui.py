import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy
#from launch import reset_env_args 
from extras.inpaint_mask import SAMOptions

from PIL import Image
from modules.sdxl_styles import legal_style_names, fooocus_expansion
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

import enhanced.gallery as gallery_util
import enhanced.topbar  as topbar
import enhanced.toolbox  as toolbox
import enhanced.translator  as translator
import enhanced.enhanced_parameters as enhanced_parameters
import enhanced.version as version
import enhanced.wildcards as wildcards
import enhanced.simpleai as simpleai
import enhanced.comfy_task as comfy_task
from enhanced.simpleai import comfyd 

def get_task(*args):
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)

def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]

def enhance_inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        False, inpaint_engine_version, 1.0, 0.618
    ]


reload_javascript()

title = f'{version.branch} {version.simplesdxl_ver} derived from Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(
    title=title,
    css=topbar.css + toolbox.css).queue()

with shared.gradio_root:
    state_topbar = gr.State({})
    params_backend = gr.State({'translation_methods': modules.config.default_translation_methods})
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                with gr.Row():
                    bar_title = gr.Markdown('<b>Presets:</b>', visible=False, elem_id='bar_title', elem_classes='bar_title')
                    bar_buttons = []
                    for i in range(shared.BUTTON_NUM):
                        bar_buttons.append(gr.Button(value='default' if i==0 else '', size='sm', visible=True, min_width=40, elem_id=f'bar{i}', elem_classes='bar_button'))
                    #bar_dropdown = gr.Dropdown(show_label=False, choices=['self','preset1','preset2','preset3'], value='self')
                with gr.Row():
                    progress_window = grh.Image(label='Preview', show_label=False, visible=True, height=768, elem_id='preview_generating',
                                            elem_classes=['main_view'], value="enhanced/attached/welcome.png")
                    progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain', elem_id='finished_gallery',
                                              height=520, visible=False, elem_classes=['main_view', 'image_gallery'])
                progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
                gallery = gr.Gallery(label='Gallery', show_label=True, object_fit='contain', visible=False, height=768,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery', preview=True )
                prompt_info_box = gr.Markdown(toolbox.make_infobox_markdown(None), visible=False, elem_id='infobox', elem_classes='infobox')
                with gr.Group(visible=False, elem_classes='toolbox_note') as params_note_box:
                    params_note_info = gr.Markdown(elem_classes='note_info')
                    params_note_input_name = gr.Textbox(show_label=False, placeholder="Type preset name here.", min_width=100, elem_classes='preset_input', visible=False)
                    params_note_delete_button = gr.Button(value='Enter', visible=False)
                    params_note_regen_button = gr.Button(value='Enter', visible=False)
                    params_note_preset_button = gr.Button(value='Enter', visible=False)
                with gr.Group(visible=False, elem_classes='identity_note') as identity_dialog:
                    identity_note_info = gr.Markdown(elem_classes='note_info', value=simpleai.identity_note)
                    with gr.Tab(label='New Identity') as new_id_tab:
                        identity_nick_input = gr.Textbox(show_label=False, placeholder="Type nickname here.", min_width=70, elem_classes='identity_input')
                        identity_tele_input = gr.Textbox(show_label=False, placeholder="Type telephone here.", min_width=70, elem_classes='identity_input')
                        identity_getvcode_button = gr.Button(value='Get Verification Code', visible=True)
                        identity_vcode_input = gr.Textbox(show_label=False, placeholder="Type vcode here.", min_width=70, elem_classes='identity_input')
                        identity_bind_button = gr.Button(value='Bind identity', min_width=150, visible=True)
                    with gr.Tab(label='Confirm Identity') as confirm_id_tab:
                        identity_phrase_input = gr.Textbox(show_label=False, type='password', placeholder="Type id phrases here.", min_width=150, elem_classes='identity_input')
                        identity_confirm_button = gr.Button(value='Confirm identity', visible=True)
                    identity_getvcode_button.click(simpleai.get_vcode, inputs=[identity_nick_input, identity_tele_input, state_topbar], outputs=state_topbar)
                    identity_bind_button.click(simpleai.bind_identity, inputs=[identity_nick_input, identity_tele_input, identity_vcode_input, state_topbar], outputs=state_topbar)
                    identity_confirm_button.click(simpleai.confirm_identity, inputs=[identity_phrase_input, state_topbar], outputs=state_topbar)

                with gr.Accordion("Finished Images Index:", open=False, visible=False) as index_radio:
                    gallery_index = gr.Radio(choices=None, label="Gallery_Index", value=None, show_label=False)
                    gallery_index.change(gallery_util.images_list_update, inputs=[gallery_index, state_topbar], outputs=[gallery, index_radio, state_topbar], show_progress=False)
            with gr.Group():
                with gr.Row(elem_classes='type_row'):
                    with gr.Column(scale=12):
                        prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        container=False, autofocus=False, elem_classes='type_row', lines=1024)
                        #prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                        #                autofocus=True, lines=3)

                        def calculateTokenCounter(text, style_selections):
                            if len(text) < 1:
                                return 0
                            num=topbar.prompt_token_prediction(text, style_selections)
                            return str(num)
                        prompt_token_counter = gr.HTML(
                            visible=True,
                            value=0,
                            elem_classes=["tokenCounter"],
                            elem_id='token_counter',
                        )

                        default_prompt = modules.config.default_prompt
                        if isinstance(default_prompt, str) and default_prompt != '':
                            shared.gradio_root.load(lambda: default_prompt, outputs=prompt)
                    with gr.Column(scale=2, min_width=0):
                        random_button = gr.Button(value="RandomPrompt", elem_classes='type_row_third', size="sm", min_width = 70)
                        translator_button = gr.Button(value="Translator", elem_classes='type_row_third', size='sm', min_width = 70)
                        super_prompter = gr.Button(value="SuperPrompt", elem_classes='type_row_third', size="sm", min_width = 70)
                    with gr.Column(scale=2, min_width=0):
                        generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True, min_width = 70)
                        reset_button = gr.Button(label="Reconnect", value="Reconnect", elem_classes='type_row', elem_id='reset_button', visible=False)
                        load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False, min_width = 70)
                        skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False, min_width = 70)
                        stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False, min_width = 70)

                        def stop_clicked(currentTask):
                            import ldm_patched.modules.model_management as model_management
                            currentTask.last_stop = 'stop'
                            if (currentTask.processing):
                                comfyd.interrupt()
                                model_management.interrupt_current_processing()
                            return currentTask

                        def skip_clicked(currentTask):
                            import ldm_patched.modules.model_management as model_management
                            currentTask.last_stop = 'skip'
                            if (currentTask.processing):
                                comfyd.interrupt()
                                model_management.interrupt_current_processing()
                            return currentTask

                        stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')
                        skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)

                with gr.Accordion(label='Wildcards & Batch Prompts', visible=False, open=True) as prompt_wildcards:
                    wildcards_list = gr.Dataset(components=[prompt], label='Wildcards: [__color__:L3:4], take 3 phrases starting from the 4th in color in order. [__color__:3], take 3 randomly. [__color__], take 1 randomly.', samples=wildcards.get_wildcards_samples(), visible=True, samples_per_page=28)
                    with gr.Accordion(label='Words/phrases of wildcard', visible=True, open=False) as words_in_wildcard:
                        wildcard_tag_name_selection = gr.Dataset(components=[prompt], label='Words:', samples=wildcards.get_words_of_wildcard_samples(), visible=True, samples_per_page=30, type='index')
                    wildcards_list.click(wildcards.add_wildcards_and_array_to_prompt, inputs=[wildcards_list, prompt, state_topbar], outputs=[prompt, wildcard_tag_name_selection, words_in_wildcard], show_progress=False, queue=False)
                    wildcard_tag_name_selection.click(wildcards.add_word_to_prompt, inputs=[wildcards_list, wildcard_tag_name_selection, prompt], outputs=prompt, show_progress=False, queue=False)
                    wildcards_array = [prompt_wildcards, words_in_wildcard, wildcards_list, wildcard_tag_name_selection]
                    wildcards_array_show =lambda x: [gr.update(visible=True)] * 2 + [gr.Dataset.update(visible=True, samples=wildcards.get_wildcards_samples()), gr.Dataset.update(visible=True, samples=wildcards.get_words_of_wildcard_samples(x))]
                    wildcards_array_hidden = [gr.update(visible=False)] * 2 + [gr.Dataset.update(visible=False, samples=wildcards.get_wildcards_samples()), gr.Dataset.update(visible=False, samples=wildcards.get_words_of_wildcard_samples())]
                    wildcards_array_hold = [gr.update()] * 4

            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
                prompt_panel_checkbox = gr.Checkbox(label='Prompt Panel', value=False, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced+', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
            with gr.Group(visible=False, elem_classes='toolbox') as image_toolbox:
                image_tools_box_title = gr.Markdown('<b>ToolBox</b>', visible=True)
                prompt_info_button = gr.Button(value='ViewMeta', size='sm', visible=True)
                prompt_regen_button = gr.Button(value='ReGenerate', size='sm', visible=True)
                prompt_delete_button = gr.Button(value='DeleteImage', size='sm', visible=True)
                prompt_info_button.click(toolbox.toggle_prompt_info, inputs=state_topbar, outputs=[prompt_info_box, state_topbar], show_progress=False)
            
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
                    with gr.Tab(label='Upscale or Variation', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale', value=False)
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')
                    with gr.Tab(label='Image Prompt', id='ip_tab') as ip_tab:
                        with gr.Row():
                            ip_advanced = gr.Checkbox(label='Advanced Control', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for image_count in range(modules.config.default_controlnet_image_count):
                                image_count += 1
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False, height=300, value=modules.config.default_ip_images[image_count])
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                        with gr.Row():
                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.Tab(label='Inpaint or Outpaint', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint', value=False, container=False)
                            inpaint_advanced_masking_checkbox = gr.Checkbox(label='Enable Advanced Masking Features', value=modules.config.default_inpaint_advanced_masking_checkbox, container=False)
                            invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox, container=False)
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = grh.Image(label='Image', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)

                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", mask_opacity=1, elem_id='inpaint_mask_canvas')
                                inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_dino_prompt_text = gr.Textbox(label='Detection prompt', value='', visible=False, info='Use singular whenever possible', placeholder='Describe what you want to detect.')
                                example_inpaint_mask_dino_prompt_text = gr.Dataset(
                                    samples=modules.config.example_enhance_detection_prompts,
                                    label='Detection Prompt Quick List',
                                    components=[inpaint_mask_dino_prompt_text],
                                    visible=modules.config.default_inpaint_mask_model == 'sam')
                                example_inpaint_mask_dino_prompt_text.click(lambda x: x[0],
                                                                            inputs=example_inpaint_mask_dino_prompt_text,
                                                                            outputs=inpaint_mask_dino_prompt_text,
                                                                            show_progress=False, queue=False)

                                with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                    inpaint_mask_sam_max_detections = gr.Slider(label="Maximum number of detections", info="Set to 0 to detect all", minimum=0, maximum=10, value=modules.config.default_sam_max_detections, step=1, interactive=True)
                                generate_mask_button = gr.Button(value='Generate mask from image')

                                def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug, params_extra):
                                    from extras.inpaint_mask import generate_mask_from_image

                                    extras = {}
                                    sam_options = None
                                    if mask_model == 'u2net_cloth_seg':
                                        extras['cloth_category'] = cloth_category
                                    elif mask_model == 'sam':
                                        sam_options = SAMOptions(
                                            dino_prompt=translator.convert(dino_prompt_text, params_extra['translation_methods']),
                                            dino_box_threshold=box_threshold,
                                            dino_text_threshold=text_threshold,
                                            dino_erode_or_dilate=dino_erode_or_dilate,
                                            dino_debug=dino_debug,
                                            max_detections=sam_max_detections,
                                            model_type=sam_model
                                        )

                                    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

                                    return mask


                                inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                                    [gr.update(visible=x == 'sam')] * 2 +
                                                                    [gr.Dataset.update(visible=x == 'sam',
                                                                                       samples=modules.config.example_enhance_detection_prompts)],
                                                          inputs=inpaint_mask_model,
                                                          outputs=[inpaint_mask_cloth_category,
                                                                   inpaint_mask_dino_prompt_text,
                                                                   inpaint_mask_advanced_options,
                                                                   example_inpaint_mask_dino_prompt_text],
                                                          queue=False, show_progress=False)
                    
                    with gr.TabItem(label='Layer_iclight') as layer_tab:
                        with gr.Row():
                            layer_method = gr.Radio(choices=comfy_task.default_method_names, value=comfy_task.default_method_names[0], container=False)
                        with gr.Row():
                            with gr.Column():
                                layer_input_image = grh.Image(label='Drag given image to here', source='upload', type='numpy', visible=True)
                            with gr.Column():
                                with gr.Group():
                                    iclight_enable = gr.Checkbox(label='Enable IC-Light', value=True)
                                    iclight_source_radio = gr.Radio(show_label=False, choices=comfy_task.iclight_source_names, value=comfy_task.iclight_source_names[0], elem_classes='iclight_source', elem_id='iclight_source')
                                gr.HTML('* The module derived from <a href="https://github.com/lllyasviel/IC-Light" target="_blank">IC-Light</a> <a href="https://github.com/layerdiffusion/LayerDiffuse" target="_blank">LayerDiffuse</a>')
                        with gr.Row():
                            example_quick_subjects = gr.Dataset(samples=comfy_task.quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
                        with gr.Row():
                            example_quick_prompts = gr.Dataset(samples=comfy_task.quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])
                    example_quick_prompts.click(lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), inputs=[example_quick_prompts, prompt], outputs=prompt, show_progress=False, queue=False)
                    example_quick_subjects.click(lambda x: x[0], inputs=example_quick_subjects, outputs=prompt, show_progress=False, queue=False)

                    with gr.TabItem(label='Enhance+', id='enhance_tab') as enhance_tab:
                        with gr.Row():
                            with gr.Column():
                                enhance_checkbox = gr.Checkbox(label='Enhance', value=modules.config.default_enhance_checkbox, container=False)
                                enhance_input_image = grh.Image(label='Use with Enhance, skips image generation', source='upload', type='numpy')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                            with gr.Column():
                                with gr.Row(visible=True) as enhance_input_panel:
                                    with gr.Tabs():
                                        with gr.Tab(label='Upscale  or  Variation'):
                                            with gr.Row():
                                                with gr.Column():
                                                    enhance_uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                              value=modules.config.default_enhance_uov_method)
                                                    enhance_uov_processing_order = gr.Radio(label='Order of Processing',
                                                                        info='Use before to enhance small details and after to enhance large areas.',
                                                                        choices=flags.enhancement_uov_processing_order,
                                                                        value=modules.config.default_enhance_uov_processing_order)
                                                    enhance_uov_prompt_type = gr.Radio(label='Prompt',
                                                                   info='Choose which prompt to use for Upscale or Variation.',
                                                                   choices=flags.enhancement_uov_prompt_types,
                                                                   value=modules.config.default_enhance_uov_prompt_type,
                                                                   visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after)

                                                    enhance_uov_processing_order.change(lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                                                                    inputs=enhance_uov_processing_order,
                                                                    outputs=enhance_uov_prompt_type,
                                                                    queue=False, show_progress=False)
                                                    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                                        enhance_ctrls = []
                                        enhance_inpaint_mode_ctrls = []
                                        enhance_inpaint_engine_ctrls = []
                                        enhance_inpaint_update_ctrls = []
                                        for index in range(modules.config.default_enhance_tabs):
                                            with gr.Tab(label=f'Region#{index + 1}') as enhance_tab_item:
                                                enhance_enabled = gr.Checkbox(label='Enable', value=False if index not in [0,1] else True, 
                                                        elem_classes='min_check', container=False)

                                                enhance_mask_dino_prompt_text = gr.Textbox(label='Detection prompt',
                                                                       info='Use singular whenever possible',
                                                                       placeholder='Describe what you want to detect.',
                                                                       interactive=True,
                                                                       value = '' if index not in [0,1] else 'face' if index==0 else 'hand',
                                                                       visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                                                example_enhance_mask_dino_prompt_text = gr.Dataset(
                                                    samples=modules.config.example_enhance_detection_prompts,
                                                    label='Detection Prompt Quick List',
                                                    components=[enhance_mask_dino_prompt_text],
                                                    visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                                                example_enhance_mask_dino_prompt_text.click(lambda x: x[0],
                                                                        inputs=example_enhance_mask_dino_prompt_text,
                                                                        outputs=enhance_mask_dino_prompt_text,
                                                                        show_progress=False, queue=False)

                                                enhance_prompt = gr.Textbox(label="Enhancement positive prompt",
                                                        placeholder="Uses original prompt instead if empty.",
                                                        elem_id='enhance_prompt')
                                                enhance_negative_prompt = gr.Textbox(label="Enhancement negative prompt",
                                                                 placeholder="Uses original negative prompt instead if empty.",
                                                                 elem_id='enhance_negative_prompt')

                                                with gr.Accordion("Detection", open=False):
                                                    enhance_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_enhance_inpaint_mask_model)
                                                    enhance_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                                          choices=flags.inpaint_mask_cloth_category,
                                                                          value=modules.config.default_inpaint_mask_cloth_category,
                                                                          visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                                                                          interactive=True)

                                                    with gr.Accordion("SAM Options",
                                                                    visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                                                                    open=False) as sam_options:
                                                        enhance_mask_sam_model = gr.Dropdown(label='SAM model',
                                                                         choices=flags.inpaint_mask_sam_model,
                                                                         value=modules.config.default_inpaint_mask_sam_model,
                                                                         interactive=True)
                                                        enhance_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0,
                                                                           maximum=1.0, value=0.3, step=0.05,
                                                                           interactive=True)
                                                        enhance_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0,
                                                                            maximum=1.0, value=0.25, step=0.05,
                                                                            interactive=True)
                                                        enhance_mask_sam_max_detections = gr.Slider(label="Maximum number of detections",
                                                                                info="Set to 0 to detect all",
                                                                                minimum=0, maximum=10,
                                                                                value=modules.config.default_sam_max_detections,
                                                                                step=1, interactive=True)

                                                with gr.Accordion("Inpaint", visible=True, open=False):
                                                    enhance_inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                                   value=modules.config.default_inpaint_method if index not in [0,1] else modules.flags.inpaint_option_detail,
                                                                   label='Method', interactive=True)
                                                    enhance_inpaint_disable_initial_latent = gr.Checkbox(
                                                        label='Disable initial latent in inpaint', value=False)
                                                    enhance_inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                                     value=modules.config.default_inpaint_engine_version,
                                                                     choices=flags.inpaint_engine_versions,
                                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                                                    enhance_inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                                     minimum=0.0, maximum=1.0, step=0.001,
                                                                     value=1.0,
                                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                                          'Only used in inpaint, not used in outpaint. '
                                                                          '(Outpaint always use 1.0)')
                                                    enhance_inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                                             minimum=0.0, maximum=1.0, step=0.001,
                                                                             value=0.618,
                                                                             info='The area to inpaint. '
                                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                                  'Only used in inpaint, not used in outpaint. '
                                                                                  '(Outpaint always use 1.0)')
                                                    enhance_inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                                            minimum=-64, maximum=64, step=1, value=0,
                                                                            info='Positive value will make white area in the mask larger, '
                                                                                 'negative value will make white area smaller. '
                                                                                 '(default is 0, always processed before any mask invert)')
                                                    enhance_mask_invert = gr.Checkbox(label='Invert Mask', value=False)

                                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                                            enhance_ctrls += [
                                                enhance_enabled,
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
                                            ]

                                            enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
                                            enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]

                                            enhance_inpaint_update_ctrls += [[
                                                enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                                                enhance_inpaint_strength, enhance_inpaint_respective_field
                                            ]]

                                            enhance_inpaint_mode.change(enhance_inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state], outputs=[
                                                enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                                                enhance_inpaint_strength, enhance_inpaint_respective_field
                                            ], show_progress=False, queue=False)

                                            enhance_mask_model.change(
                                                lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                        [gr.update(visible=x == 'sam')] * 2 +
                                                        [gr.Dataset.update(visible=x == 'sam',
                                                            samples=modules.config.example_enhance_detection_prompts)],
                                                inputs=enhance_mask_model,
                                                outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                                                        example_enhance_mask_dino_prompt_text],
                                                queue=False, show_progress=False)

                    with gr.Tab(label='Describe', id='describe_tab') as describe_tab:
                        with gr.Row():
                            with gr.Column():
                                describe_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                describe_methods = gr.CheckboxGroup(
                                    label='Content Type',
                                    choices=flags.describe_types,
                                    value=modules.config.default_describe_content_type)
                                describe_apply_styles = gr.Checkbox(label='Apply Styles', value=modules.config.default_describe_apply_prompts_checkbox)
                                describe_btn = gr.Button(value='Describe this Image into Prompt')
                                describe_image_size = gr.Textbox(label='Original Size / Recommended Size', elem_id='describe_image_size', visible=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Documentation</a>')

                                def trigger_show_image_properties(image):
                                    image_size = modules.util.get_image_size_info(image, modules.flags.available_aspect_ratios[0])
                                    return gr.update(value=image_size, visible=True)

                                describe_input_image.upload(trigger_show_image_properties, inputs=describe_input_image,
                                                            outputs=describe_image_size, show_progress=False, queue=False)

                    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                        with gr.Column():
                            metadata_input_image = grh.Image(label='Drag any image generated by Fooocus here', source='upload', type='pil')
                            with gr.Accordion("Preview Metadata", open=True, visible=True) as metadata_preview:
                                metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata', interactive=False)

                        def trigger_metadata_preview(file):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return [results, gr.update(interactive=parameters is not None)]

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=[metadata_json, metadata_import_button], queue=False, show_progress=True)


            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)

            current_tab = gr.Textbox(value='uov', visible=False)

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            with gr.Tab(label='Setting'):
                preset_instruction = gr.HTML(visible=False, value=topbar.preset_instruction())
                if not args_manager.args.disable_preset_selection:
                    preset_selection = gr.Radio(label='Preset',
                                                choices=modules.config.available_presets,
                                                value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                visible=False, interactive=True)
                performance_selection = gr.Radio(label='Performance',
                                                 choices=flags.Performance.list(),
                                                 value=modules.config.default_performance)
                with gr.Group():
                    image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)
                    with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
                        aspect_ratios_selection = gr.Textbox(value='', visible=False) 
                        aspect_ratios_selections = []
                        for template in flags.aspect_ratios_templates:
                            aspect_ratios_selections.append(gr.Radio(label='Aspect Ratios', choices=flags.available_aspect_ratios_list[template], value=flags.default_aspect_ratios[template], visible= template=='SDXL', info='Vertical(9:16), Portrait(4:5), Photo(4:3), Landscape(3:2), Widescreen(16:9), Cinematic(21:9)', elem_classes='aspect_ratios'))
                        
                        for aspect_ratios_select in aspect_ratios_selections:
                            aspect_ratios_select.change(lambda x: x, inputs=aspect_ratios_select, outputs=aspect_ratios_selection, queue=False, show_progress=False).then(lambda x: None, inputs=aspect_ratios_select, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                        shared.gradio_root.load(lambda x: x, inputs=aspect_ratios_selections[0], outputs=aspect_ratios_selection, queue=False, show_progress=False).then(lambda x: None, inputs=aspect_ratios_selections[0], queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                    output_format = gr.Radio(label='Output Format',
                                         choices=flags.OutputFormat.list(),
                                         value=modules.config.default_output_format)

                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                             info='Describing what you do not want to see.', lines=2,
                                             elem_id='negative_prompt',
                                             value=modules.config.default_prompt_negative)
                seed_random = gr.Checkbox(label='Random', value=True)
                image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, seed_string):
                    if r:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        try:
                            seed_value = int(seed_string)
                            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                return seed_value
                        except ValueError:
                            pass
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                def update_history_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')

                    return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">\U0001F4DA History Log</a>')

                history_link = gr.HTML()
                shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)

            with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
                style_sorter.try_load_sorted_styles(
                    style_names=legal_style_names,
                    default_selected=modules.config.default_styles)

                style_search_bar = gr.Textbox(show_label=False, container=False,
                                              placeholder="\U0001F50E Type here to search styles ...",
                                              value="",
                                              label='Search Styles')
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(modules.config.default_styles),
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                        outputs=style_selections)

                style_search_bar.change(style_sorter.search_styles,
                                        inputs=[style_selections, style_search_bar],
                                        outputs=style_selections,
                                        queue=False,
                                        show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')
                prompt.change(lambda x,y: calculateTokenCounter(x,y), inputs=[prompt, style_selections], outputs=prompt_token_counter)

            with gr.Tab(label='Models'):
                with gr.Group():
                    with gr.Row():
                        base_model = gr.Dropdown(label='Base Model (SDXL only)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
                        refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)

                    refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Use 0.4 for SD1.5 realistic models; '
                                                    'or 0.667 for SD1.5 anime models; '
                                                    'or 0.8 for XL-refiners; '
                                                    'or any value for switching two SDXL models.',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Group():
                    lora_ctrls = []

                    for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                                       elem_classes=['lora_enable', 'min_check'], scale=1)
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=filename,
                                                     elem_classes='lora_model', scale=5)
                            lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                                    maximum=modules.config.default_loras_max_weight, step=0.01, value=weight,
                                                    elem_classes='lora_weight', scale=5)
                            lora_ctrls += [lora_enabled, lora_model, lora_weight]

                with gr.Row():
                    refresh_files = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
                #with gr.Row():
                #    sync_model_info = gr.Checkbox(label='Sync model info', interactive=False, info='Improve usability and transferability for preset and embedinfo.', value=False, container=False)
                #    with gr.Column(visible=False) as info_sync_texts:
                #        models_infos = []
                #info_sync_button = gr.Button(label='Sync', value='\U0001f504 Sync Remote Info', visible=False, variant='secondary', elem_classes='refresh_button')
                #info_progress = gr.Markdown('Note: If MUID is not obtained after synchronizing, it means that it is a new model file. You need to add an available download URL before.', visible=False)
                #sync_model_info.change(lambda x: (gr.update(visible=x), gr.update(visible=x),  gr.update(visible=x)), inputs=sync_model_info, outputs=[info_sync_texts, info_sync_button, info_progress], queue=False, show_progress=False)
                #info_sync_button.click(toolbox.sync_model_info_click, inputs=models_infos, outputs=models_infos, queue=False, show_progress=False)

            with gr.Tab(label='Advanced'):
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=0.01, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')
                sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='Higher value means image and texture are sharper.')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Documentation</a>')
                dev_mode = gr.Checkbox(label='Developer Debug Mode', value=modules.config.default_developer_debug_mode_checkbox, container=False)

                with gr.Column(visible=modules.config.default_developer_debug_mode_checkbox) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                        adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='When to end the guidance from positive/negative ADM. ')

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
                        clip_skip = gr.Slider(label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
                                                 value=modules.config.default_clip_skip,
                                                 info='Bypass CLIP layers to avoid overfitting (use 1 to not skip any layers, 2 is recommended).')
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                   value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                     value=modules.config.default_scheduler)
                        vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                                     value=modules.config.default_vae, show_label=True)

                        generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                          info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                          value=False)

                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='Set as -1 to disable. For developer debugging.')
                        overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch,
                                                     info='Set as -1 to disable. For developer debugging.')
                        overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='Set as -1 to disable. For developer debugging. '
                                                         'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging. '
                                                          'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='Set as negative number to disable. For developer debugging.')
                        overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                               minimum=-1, maximum=1.0, step=0.001,
                                                               value=modules.config.default_overwrite_upscale,
                                                               info='Set as negative number to disable. For developer debugging.')

                        disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                                      interactive=not modules.config.default_black_out_nsfw,
                                                      info='Disable preview during generation.')
                        disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                      value=flags.Performance.has_restricted_features(modules.config.default_performance),
                                                      info='Disable intermediate results during generation, only show final gallery.')

                        disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                             info='Disable automatic seed increment when image number is > 1.',
                                                             value=False)
                        read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False, visible=False)

                        black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                                     interactive=not modules.config.default_black_out_nsfw,
                                                     info='Use black image if NSFW is detected.')

                        black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                              inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                              show_progress=False)

                        if not args_manager.args.disable_image_log:
                            save_final_enhanced_image_only = gr.Checkbox(label='Save only final enhanced image',
                                                                         value=modules.config.default_save_only_final_enhanced_image)

                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                  info='Adds parameters to generated images allowing manual regeneration.')
                            metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                                       visible=modules.config.default_save_metadata_to_images)

                            save_metadata_to_images.change(lambda x: [gr.update(visible=x)], inputs=[save_metadata_to_images], outputs=[metadata_scheme], queue=False, show_progress=False)

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')


                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='Canny'):
                            canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                             step=1, value=128)

                    with gr.Tab(label='Inpaint'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                        debugging_enhance_masks_checkbox = gr.Checkbox(label='Debug Enhance Masks', value=False,
                                                                       info='Show enhance masks in preview and final results')
                        debugging_dino = gr.Checkbox(label='Debug GroundingDINO', value=False,
                                                     info='Use GroundingDINO boxes instead of more detailed SAM masks')
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller. '
                                                                 '(default is 0, always processed before any mask invert)')
                        dino_erode_or_dilate = gr.Slider(label='GroundingDINO Box Erode or Dilate',
                                                         minimum=-64, maximum=64, step=1, value=0,
                                                         info='Positive value will make white area in the mask larger, '
                                                              'negative value will make white area smaller. '
                                                              '(default is 0, processed before SAM)')

                        inpaint_mask_color = gr.ColorPicker(label='Inpaint brush color', value='#FFFFFF', elem_id='inpaint_brush_color')

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                                 inputs=inpaint_advanced_masking_checkbox,
                                                                 outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                                 queue=False, show_progress=False)

                        inpaint_mask_color.change(lambda x: gr.update(brush_color=x), inputs=inpaint_mask_color,
                                                  outputs=inpaint_input_image,
                                                  queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                def dev_mode_checked(r):
                    return gr.update(visible=r)

                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def refresh_files_clicked(state_params):
                    engine = state_params.get('engine', 'Fooocus')
                    model_filenames, lora_filenames, vae_filenames = modules.config.update_files(engine)
                    results = [gr.update(choices=model_filenames)]
                    results += [gr.update(choices=['None'] + model_filenames)]
                    results += [gr.update(choices=[flags.default_vae] + vae_filenames)]
                    if not args_manager.args.disable_preset_selection:
                        results += [gr.update(choices=modules.config.available_presets)]
                    for i in range(modules.config.default_max_lora_number):
                        results += [gr.update(interactive=True),
                                    gr.update(choices=['None'] + lora_filenames), gr.update()]
                    return results

                refresh_files_output = [base_model, refiner_model, vae_name]
                if not args_manager.args.disable_preset_selection:
                    refresh_files_output += [preset_selection]
                refresh_files.click(refresh_files_clicked, [state_topbar], refresh_files_output + lora_ctrls,
                                    queue=False, show_progress=False)

            with gr.Tab(label='Enhanced'):
                with gr.Row(visible=False):
                    binding_id_button = gr.Button(value='Binding Identity', visible=True)
                with gr.Row():
                    language_ui = gr.Radio(label='Language of UI', choices=['En', '中文'], value=modules.flags.language_radio(args_manager.args.language), interactive=(args_manager.args.language in ['default', 'cn', 'en']))
                    background_theme = gr.Radio(label='Theme of background', choices=['light', 'dark'], value=args_manager.args.theme, interactive=True)
                with gr.Group():
                    prompt_preset_button = gr.Button(value='Save the current parameters as a preset package')
                    comfyd_active_checkbox = gr.Checkbox(label='Enable Comfyd always active', value=not args_manager.args.disable_comfyd, info='Enabling will improve execution speed but occupy some memory.')
                    image_tools_checkbox = gr.Checkbox(label='Enable ParamsTools', value=True, info='Management of published image sets, located in the middle toolbox on the right side of the image set.')
                    backfill_prompt = gr.Checkbox(label='Backfill prompt while switching images', value=modules.config.default_backfill_prompt, interactive=True, info='Extract and backfill prompt and negative prompt while switching historical gallery images.')
                    translation_methods = gr.Radio(label='Translation methods', choices=modules.flags.translation_methods, value=modules.config.default_translation_methods, info='\'Model\' requires more GPU/CPU and \'APIs\' rely on third.')
                    mobile_url = gr.Checkbox(label=f'http://{args_manager.args.listen}:{args_manager.args.port}{args_manager.args.webroot}/', value=True, info='Mobile phone access address within the LAN. If you want WAN access, consulting QQ group: 938075852.', interactive=False)
                    
                    def sync_params_backend(v, params):
                        params.update({'translation_methods':v})
                        return params
                    translation_methods.change(sync_params_backend, inputs=[translation_methods, params_backend], outputs=params_backend)
                # custom plugin "OneButtonPrompt"
                import custom.OneButtonPrompt.ui_onebutton as ui_onebutton
                run_event = gr.Number(visible=False, value=0)
                ui_onebutton.ui_onebutton(prompt, run_event, random_button)
                with gr.Tab(label="SuperPrompter"):
                    #super_prompter = gr.Button(value="<<SuperPrompt", size="sm", min_width = 70)
                    super_prompter_prompt = gr.Textbox(label='Prompt prefix', value='Expand the following prompt to add more detail:', lines=1)
                with gr.Row():
                    gr.Markdown(value=f'OS: {shared.sysinfo["os_name"]}, {shared.sysinfo["cpu_arch"]}, {shared.sysinfo["cuda_version"]}, Torch{shared.sysinfo["torch_version"]}, XF{shared.sysinfo["xformers_version"]}<br>Ver: {version.branch} {version.simplesdxl_ver} / Fooocus {fooocus_version.version}<br>PyHash: {shared.sysinfo["pyhash"]}, UIHash: {shared.sysinfo["uihash"]}')

            iclight_enable.change(lambda x: [gr.update(interactive=x, value='' if not x else comfy_task.iclight_source_names[0]), gr.update(value=modules.config.add_ratio('1024*1024') if not x else modules.config.default_aspect_ratio)], inputs=iclight_enable, outputs=[iclight_source_radio, aspect_ratios_selections[0]], queue=False, show_progress=False)
            layout_image_tab = [performance_selection, style_selections, freeu_enabled, refiner_model, refiner_switch] + lora_ctrls
            def toggle_image_tab(tab, styles):
                result = []
                if 'layer' in tab:
                    result = [gr.update(choices=flags.Performance.list()[:2]), gr.update(value=[s for s in styles if s!=fooocus_expansion])]
                    result += [gr.update(value=False, interactive=False)]
                    result += [gr.update(interactive=False)] * 17
                else:
                    result = [gr.update(choices=flags.Performance.list()), gr.update()]
                    result += [gr.update(interactive=True)] * 18
                return result
            
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            layer_tab.select(lambda: 'layer', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            
            input_image_checkbox.change(lambda x: [gr.update(visible=x), gr.update(choices=flags.Performance.list()), 
                gr.update()] + [gr.update(interactive=True)]*18, inputs=input_image_checkbox,
                outputs=[image_input_panel] + layout_image_tab, queue=False, show_progress=False, _js=switch_js)
            prompt_panel_checkbox.change(lambda x: gr.update(visible=x, open=x if x else True), inputs=prompt_panel_checkbox, outputs=prompt_wildcards, queue=False, show_progress=False, _js=switch_js).then(lambda x,y: wildcards_array_show(y['wildcard_in_wildcards']) if x else wildcards_array_hidden, inputs=[prompt_panel_checkbox, state_topbar], outputs=wildcards_array, queue=False, show_progress=False)

            image_tools_checkbox.change(lambda x,y: gr.update(visible=x) if "gallery_state" in y and y["gallery_state"] == 'finished_index' else gr.update(visible=False), inputs=[image_tools_checkbox,state_topbar], outputs=image_toolbox, queue=False, show_progress=False)
            comfyd_active_checkbox.change(lambda x: comfyd.active(x), inputs=comfyd_active_checkbox, queue=False, show_progress=False)
            import enhanced.superprompter
            super_prompter.click(lambda x, y, z: enhanced.superprompter.answer(input_text=translator.convert(f'{y}{x}', z), seed=image_seed), inputs=[prompt, super_prompter_prompt, translation_methods], outputs=prompt, queue=False, show_progress=True)
            ehps = [backfill_prompt, translation_methods, comfyd_active_checkbox]
            language_ui.select(None, inputs=language_ui, _js="(x) => set_language_by_ui(x)")
            background_theme.select(None, inputs=background_theme, _js="(x) => set_theme_by_ui(x)")

            gallery_index.select(gallery_util.select_index, inputs=[gallery_index, image_tools_checkbox, state_topbar], outputs=[gallery, image_toolbox, progress_window, progress_gallery, prompt_info_box, params_note_box, params_note_info, params_note_input_name, params_note_regen_button, params_note_preset_button, state_topbar], show_progress=False)
            gallery.select(gallery_util.select_gallery, inputs=[gallery_index, state_topbar, backfill_prompt], outputs=[prompt_info_box, prompt, negative_prompt, params_note_info, params_note_input_name, params_note_regen_button, params_note_preset_button, state_topbar], show_progress=False)
            progress_gallery.select(gallery_util.select_gallery_progress, inputs=state_topbar, outputs=[prompt_info_box, params_note_info, params_note_input_name, params_note_regen_button, params_note_preset_button, state_topbar], show_progress=False)

            #with gr.Row():
            #    manual_link = gr.HTML(value='<a href="https://github.com/metercai/UseCaseGuidance/blob/main/UseCaseGuidanceForSimpleSDXL.md">SimpleSDXL创意生图场景应用指南</a>')
        state_is_generating = gr.State(False)

        load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                             base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                             seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                             inpaint_mode] + enhance_inpaint_mode_ctrls + [generate_button,
                             load_parameter_button] + freeu_ctrls + lora_ctrls

        if not args_manager.args.disable_preset_selection:
            def preset_selection_change(preset, is_generating, inpaint_mode):
                preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

                default_model = preset_prepared.get('base_model')
                previous_default_models = preset_prepared.get('previous_default_models', [])
                checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
                embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
                lora_downloads = preset_prepared.get('lora_downloads', {})
                vae_downloads = preset_prepared.get('vae_downloads', {})

                preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = topbar.download_models(
                    default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads,
                    vae_downloads)

                if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                    del preset_prepared['prompt']

                return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating, inpaint_mode)


        def inpaint_engine_state_change(inpaint_engine_version, *args):
            if inpaint_engine_version == 'empty':
                inpaint_engine_version = modules.config.default_inpaint_engine_version

            result = []
            for inpaint_mode in args:
                if inpaint_mode != modules.flags.inpaint_option_detail:
                    result.append(gr.update(value=inpaint_engine_version))
                else:
                    result.append(gr.update())

            return result

        if not args_manager.args.disable_preset_selection:
            preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}') \
                .then(inpaint_engine_state_change, inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)

        performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                               [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                               [gr.update(value=flags.Performance.has_restricted_features(x))] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                     ], queue=False, show_progress=False)

        
        def reset_aspect_ratios(aspect_ratios):
            if len(aspect_ratios.split(','))>1:
                template = aspect_ratios.split(',')[1]
                aspect_ratios = aspect_ratios.split(',')[0]
                if template=='HyDiT':
                    results = [gr.update(visible=False), gr.update(value=aspect_ratios, visible=True)] + [gr.update(visible=False)] * 2
                elif template=='Common':
                    results = [gr.update(visible=False)] * 2 + [gr.update(value=aspect_ratios, visible=True), gr.update(visible=False)]
                elif template=='Flux':
                    results = [gr.update(visible=False)] * 3 + [gr.update(value=aspect_ratios, visible=True)]
                else:
                    results = [gr.update(value=aspect_ratios, visible=True)] + [gr.update(visible=False)] * 3
            else:
                results = [gr.update()] * 4
            return results


        aspect_ratios_selection.change(reset_aspect_ratios, inputs=aspect_ratios_selection, outputs=aspect_ratios_selections, queue=False, show_progress=False).then(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')


        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)

        inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        # load configured default_inpaint_method
        default_inpaint_ctrls = [inpaint_mode, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field]
        for mode, disable_initial_latent, engine, strength, respective_field in [default_inpaint_ctrls] + enhance_inpaint_update_ctrls:
            shared.gradio_root.load(inpaint_mode_change, inputs=[mode, inpaint_engine_state], outputs=[
                inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, disable_initial_latent,
                engine, strength, respective_field
            ], show_progress=False, queue=False)

        generate_mask_button.click(fn=generate_mask,
                                   inputs=[inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
                                           inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                           inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                           inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino, params_backend],
                                   outputs=inpaint_mask_image, show_progress=True, queue=True)

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
            read_wildcards_in_order, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [layer_method, layer_input_image, iclight_enable, iclight_source_radio]
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
        ctrls += [sampler_name, scheduler_name, vae_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls
        ctrls += [params_backend]

        if not args_manager.args.disable_image_log:
            ctrls += [save_final_enhanced_image_only]

        if not args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        ctrls += ip_ctrls
        
        ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
                  enhance_input_image, enhance_checkbox, enhance_uov_method, enhance_uov_processing_order,
                  enhance_uov_prompt_type]
        ctrls += enhance_ctrls

        system_params = gr.JSON({}, visible=False)
        def parse_meta(raw_prompt_txt, is_generating, state_params, panel_status):
            loaded_json = None
            if len(raw_prompt_txt)>=1 and (raw_prompt_txt[-1]=='[' or raw_prompt_txt[-1]=='_'):
                return [gr.update()] * 3 + [True]
            try:
                if '{' in raw_prompt_txt:
                    if '}' in raw_prompt_txt:
                        if ':' in raw_prompt_txt:
                            loaded_json = json.loads(raw_prompt_txt)
                            assert isinstance(loaded_json, dict)
            except:
                loaded_json = None

            if loaded_json is None:
                if is_generating:
                    return [gr.update()] * 4
                else:
                    return [gr.update(), gr.update(visible=True), gr.update(visible=False), gr.update()]

            return [json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True), gr.update()]

        prompt.input(parse_meta, inputs=[prompt, state_is_generating, state_topbar, prompt_panel_checkbox], outputs=[prompt, generate_button, load_parameter_button, prompt_panel_checkbox], queue=False, show_progress=False)
        
        translator_button.click(lambda x, y: translator.convert(x, y), inputs=[prompt, translation_methods], outputs=prompt, queue=False, show_progress=True)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(file, state_is_generating, state_params):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
            if parameters is None:
                print('Could not find metadata in the image!')
            return toolbox.reset_params_by_image_meta(parameters, state_params, state_is_generating, inpaint_mode)

        reset_preset_layout = [params_backend, performance_selection, scheduler_name, sampler_name, input_image_checkbox, enhance_checkbox, base_model, refiner_model, overwrite_step, guidance_scale, negative_prompt, preset_instruction] + lora_ctrls
        reset_preset_func = [output_format, inpaint_advanced_masking_checkbox, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, backfill_prompt, translation_methods, input_image_checkbox, state_topbar]

        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating, state_topbar], outputs=reset_preset_layout + reset_preset_func + load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

        model_check = [prompt, negative_prompt, base_model, refiner_model] + lora_ctrls
        nav_bars = [bar_title] + bar_buttons
        protections = [prompt, random_button, translator_button, super_prompter, background_theme, image_tools_checkbox] + nav_bars[1:]
        generate_button.click(topbar.process_before_generation, inputs=[state_topbar, params_backend] + ehps, outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating, index_radio, image_toolbox, prompt_info_box] + protections + [params_backend], show_progress=False) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
            .then(fn=enhanced_parameters.set_all_enhanced_parameters, inputs=ehps) \
            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
            .then(topbar.process_after_generation, inputs=state_topbar, outputs=[generate_button, stop_button, skip_button, state_is_generating, gallery_index, index_radio] + protections, show_progress=False) \
            .then(fn=update_history_link, outputs=history_link) \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        reset_button.click(lambda: [worker.AsyncTask(args=[]), False, gr.update(visible=True, interactive=True)] +
                                   [gr.update(visible=False)] * 6 +
                                   [gr.update(visible=True, value=[])],
                           outputs=[currentTask, state_is_generating, generate_button,
                                    reset_button, stop_button, skip_button,
                                    progress_html, progress_window, progress_gallery, gallery],
                           queue=False)

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(modes, img, apply_styles):
            describe_prompts = []
            styles = set()

            if flags.describe_type_photo in modes:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                describe_prompts.append(default_interrogator_photo(img))
                styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

            if flags.describe_type_anime in modes:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                describe_prompts.append(default_interrogator_anime(img))
                styles.update(["Fooocus V2", "Fooocus Masterpiece"])

            if len(styles) == 0 or not apply_styles:
                styles = gr.update()
            else:
                styles = list(styles)

            if len(describe_prompts) == 0:
                describe_prompt = gr.update()
            else:
                describe_prompt = ', '.join(describe_prompts)

            return describe_prompt, styles

        describe_btn.click(trigger_describe, inputs=[describe_methods, describe_input_image, describe_apply_styles],
                           outputs=[prompt, style_selections], show_progress=True, queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')

        if args_manager.args.enable_auto_describe_image:
            def trigger_auto_describe(mode, img, prompt, apply_styles):
                # keep prompt if not empty
                if prompt == '':
                    return trigger_describe(mode, img, apply_styles)
                return gr.update(), gr.update()

            uov_input_image.upload(trigger_auto_describe, inputs=[describe_methods, uov_input_image, prompt, describe_apply_styles],
                                   outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

            enhance_input_image.upload(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
                .then(trigger_auto_describe, inputs=[describe_methods, enhance_input_image, prompt, describe_apply_styles],
                      outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

    prompt_delete_button.click(toolbox.toggle_note_box_delete, inputs=state_topbar, outputs=[params_note_info, params_note_delete_button, params_note_box, state_topbar], show_progress=False)
    params_note_delete_button.click(toolbox.delete_image, inputs=state_topbar, outputs=[gallery, gallery_index, params_note_delete_button, params_note_box, state_topbar], show_progress=False)
    
    prompt_regen_button.click(toolbox.toggle_note_box_regen, inputs=model_check + [state_topbar], outputs=[params_note_info, params_note_regen_button, params_note_box, state_topbar], show_progress=False)
    params_note_regen_button.click(toolbox.reset_image_params, inputs=[state_topbar, state_is_generating, inpaint_mode], outputs=reset_preset_layout + reset_preset_func + load_data_outputs + [params_note_regen_button, params_note_box], show_progress=False)

    prompt_preset_button.click(toolbox.toggle_note_box_preset, inputs=model_check + [state_topbar], outputs=[params_note_info, params_note_input_name, params_note_preset_button, params_note_box, state_topbar], show_progress=False)
    params_note_preset_button.click(toolbox.save_preset, inputs=[params_note_input_name, params_backend] + reset_preset_func + load_data_outputs, outputs=[params_note_input_name, params_note_preset_button, params_note_box, state_topbar] + nav_bars, show_progress=False) \
        .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, queue=False, show_progress=False) \
        .then(fn=lambda x: None, inputs=system_params, _js=topbar.refresh_topbar_status_js)

    

    reset_layout_params = nav_bars + reset_preset_layout + reset_preset_func + load_data_outputs
    reset_preset_inputs = [prompt, negative_prompt, state_topbar, state_is_generating, inpaint_mode, comfyd_active_checkbox]

    
    binding_id_button.click(simpleai.toggle_identity_dialog, inputs=state_topbar, outputs=identity_dialog, show_progress=False)

    for i in range(shared.BUTTON_NUM):
        bar_buttons[i].click(topbar.check_absent_model, inputs=[bar_buttons[i], state_topbar], outputs=[state_topbar]) \
               .then(topbar.reset_layout_params, inputs=reset_preset_inputs, outputs=reset_layout_params, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(lambda: None, _js='()=>{refresh_style_localization();}') \
               .then(inpaint_engine_state_change, inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)


    shared.gradio_root.load(fn=lambda x: x, inputs=system_params, outputs=state_topbar, _js=topbar.get_system_params_js, queue=False, show_progress=False) \
                      .then(topbar.init_nav_bars, inputs=state_topbar, outputs=nav_bars + [progress_window, language_ui, background_theme, gallery_index, index_radio, inpaint_advanced_masking_checkbox, preset_instruction], show_progress=False) \
                      .then(topbar.reset_layout_params, inputs=reset_preset_inputs, outputs=reset_layout_params, show_progress=False) \
                      .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
                      .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
                      .then(topbar.sync_message, inputs=state_topbar, outputs=[state_topbar]) \
                      .then(fn=lambda: None, _js='refresh_grid_delayed')


def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


#dump_default_english_config()
import logging
import httpx
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

import logging
import httpx
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
hydit_logger = logging.getLogger("hydit")
hydit_logger.setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#reset_env_args()

if not args_manager.args.disable_comfyd:
    comfyd.active(True)

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    root_path=args_manager.args.webroot,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)

