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
import launch

from PIL import Image
from modules.sdxl_styles import legal_style_names
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
import enhanced.location as location
import enhanced.wildcards as wildcards
from enhanced.models_info import models_info, sync_model_info_click 

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


reload_javascript()

title = f'{version.branch} {version.simplesdxl_ver} derived from Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(
    title=title,
    css=topbar.css + toolbox.css).queue()

with shared.gradio_root:
    state_topbar = gr.State({})
    currentTask = gr.State(worker.AsyncTask(args=[]))
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                with gr.Row():
                    bar_title = gr.Markdown('<b>Presets:</b>', visible=False, elem_id='bar_title', elem_classes='bar_title')
                    bar0_button = gr.Button(value='default', size='sm', visible=True, min_width=70, elem_id='bar0', elem_classes='bar_button')
                    bar1_button = gr.Button(value='', size='sm', visible=True, min_width=70, elem_id='bar1', elem_classes='bar_button')
                    bar2_button = gr.Button(value='', size='sm', visible=True, min_width=70, elem_id='bar2', elem_classes='bar_button')
                    bar3_button = gr.Button(value='', size='sm', visible=True, min_width=70, elem_id='bar3', elem_classes='bar_button')
                    bar4_button = gr.Button(value='', size='sm', visible=True, min_width=70, elem_id='bar4', elem_classes='bar_button')
                    bar5_button = gr.Button(value='', size='sm', visible=True, min_width=70, elem_id='bar5', elem_classes='bar_button')
                    bar6_button = gr.Button(value='', size='sm', visible=True, min_width=70, elem_id='bar6', elem_classes='bar_button')
                    bar7_button = gr.Button(value='', size='sm', visible=True, min_width=70, elem_id='bar7', elem_classes='bar_button')
                    bar8_button = gr.Button(value='', size='sm', visible=True, min_width=70, elem_id='bar8', elem_classes='bar_button')
                with gr.Row():
                    progress_window = grh.Image(label='Preview', show_label=True, visible=True, height=768, elem_id='preview_generating',
                                            elem_classes=['main_view'], value="enhanced/attached/welcome.jpg")
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
                    params_note_embed_button = gr.Button(value='Enter', visible=False)
                with gr.Group(visible=False, elem_classes='toolbox_note') as preset_down_note:
                    preset_down_note_info = gr.Markdown(elem_classes='note_info', value=topbar.preset_down_note_info)
                    preset_down_confirm_button = gr.Button(value='Continue', visible=True)
                    preset_down_cancel_button = gr.Button(value='Cancel', visible=True)
                    preset_down_confirm_button.click(fn=lambda x: x.update({"load_preset":1}), inputs=state_topbar, outputs=state_topbar)
                    preset_down_cancel_button.click(fn=lambda x: x.update({"load_preset":0}), inputs=state_topbar, outputs=state_topbar)
                with gr.Accordion("Finished Images Index:", open=False, visible=False) as index_radio:
                    gallery_index = gr.Radio(choices=None, label="Gallery_Index", value=None, show_label=False)
                    gallery_index.change(gallery_util.images_list_update, inputs=[gallery_index, state_topbar], outputs=[gallery, index_radio, state_topbar], show_progress=False)
            with gr.Group():
                with gr.Row(elem_classes='type_row'):
                    with gr.Column(scale=12):
                        prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        container=False, autofocus=False, elem_classes='type_row', lines=1024)
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
                        random_button = gr.Button(value="RandomPrompt", elem_classes='type_row_third', size="sm", min_width = 1)
                        translator_button = gr.Button(value="Translator", elem_classes='type_row_third', size='sm', min_width = 1)
                        super_prompter = gr.Button(value="SuperPrompt", elem_classes='type_row_third', size="sm", min_width = 1)
                    with gr.Column(scale=2, min_width=0):
                        generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                        load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                        skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', visible=False)
                        stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                        def stop_clicked(currentTask):
                            import ldm_patched.modules.model_management as model_management
                            currentTask.last_stop = 'stop'
                            if (currentTask.processing):
                                model_management.interrupt_current_processing()
                            return currentTask

                        def skip_clicked(currentTask):
                            import ldm_patched.modules.model_management as model_management
                            currentTask.last_stop = 'skip'
                            if (currentTask.processing):
                                model_management.interrupt_current_processing()
                            return currentTask

                        stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')
                        skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)

                with gr.Accordion(label='Wildcards & Batch Prompts', visible=False, open=True) as prompt_wildcards:
                    wildcards_list = gr.Dataset(components=[prompt], label='Wildcards: [__color__:L3:4], take 3 phrases starting from the 4th in color in order. [__color__:3], take 3 randomly. [__color__], take 1 randomly.', samples=wildcards.get_wildcards_samples(), visible=False, samples_per_page=14 if args_manager.args.language=='cn' else 20)
                    with gr.Accordion(label='Words/phrases of wildcard', visible=False, open=False) as words_in_wildcard:
                        tag_name_selection = gr.Dataset(components=[prompt], label='Words:', samples=wildcards.get_words_of_wildcard_samples(), visible=False, samples_per_page=30)
                    wildcards_list.click(wildcards.add_wildcards_and_array_to_prompt, inputs=[wildcards_list, prompt, state_topbar], outputs=[prompt, tag_name_selection, words_in_wildcard], show_progress=False, queue=False)
                    wildcards_array = [prompt_wildcards, words_in_wildcard, wildcards_list, tag_name_selection]
                    wildcards_array_show =lambda x: [gr.update(visible=True)] * 2 + [gr.Dataset.update(visible=True, samples=wildcards.get_wildcards_samples()), gr.Dataset.update(visible=True, samples=wildcards.get_words_of_wildcard_samples(x["wildcard_in_wildcards"]))]
                    wildcards_array_hidden = [gr.update(visible=False)] * 2 + [gr.Dataset.update(visible=False, samples=wildcards.get_wildcards_samples()), gr.Dataset.update(visible=False, samples=wildcards.get_words_of_wildcard_samples())]

            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=False, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced+', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
                image_tools_checkbox = gr.Checkbox(label='ParamsTools', value=False, container=False, elem_classes='min_check')
            
            with gr.Group(visible=False, elem_classes='toolbox') as image_toolbox:
                image_tools_box_title = gr.Markdown('<b>ToolBox</b>', visible=True)
                prompt_info_button = gr.Button(value='ViewMeta', size='sm', visible=True)
                prompt_regen_button = gr.Button(value='ReGenerate', size='sm', visible=True)
                prompt_embed_button = gr.Button(value='EmbedMeta', size='sm', visible=not modules.config.default_save_metadata_to_images)
                prompt_delete_button = gr.Button(value='DeleteImage', size='sm', visible=True)
                image_tools_checkbox.change(toolbox.toggle_toolbox, inputs=[image_tools_checkbox, state_topbar], outputs=[image_toolbox, prompt_info_box, params_note_info, params_note_input_name, params_note_regen_button, params_note_preset_button, state_topbar], queue=False, show_progress=False)
                prompt_info_button.click(toolbox.toggle_prompt_info, inputs=state_topbar, outputs=[prompt_info_box, state_topbar], show_progress=False)
            
            with gr.Row(visible=False) as image_input_panel:
                with gr.Tabs():
                    with gr.TabItem(label='Upscale or Variation') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Drag above image to here', source='upload', type='numpy')
                            with gr.Column():
                                mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale', value=False)
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=flags.disabled)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Document</a> ')
                    with gr.TabItem(label='Image Prompt') as ip_tab:
                        with gr.Row():
                            ip_advanced = gr.Checkbox(label='Advanced Control', value=False, container=False)
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for _ in range(flags.controlnet_image_count):
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False, height=300)
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=False) as ad_col:
                                        with gr.Row():
                                            default_end, default_weight = flags.default_parameters[flags.default_ip]

                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=default_end)
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=default_weight)
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=flags.default_ip, container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        #ip_advanced = gr.Checkbox(label='Advanced Control', value=False, container=False)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Document</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)
                    with gr.TabItem(label='Inpaint or Outpaint') as inpaint_tab:
                        with gr.Row():
                            mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint', value=False)
                            inpaint_mask_upload_checkbox = gr.Checkbox(label='Enable Mask Upload', value=True)
                            invert_mask_checkbox = gr.Checkbox(label='Invert Mask', value=False)
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = grh.Image(label='Drag inpaint or outpaint image to here', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas')
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.flags.inpaint_option_default, label='Method')
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Document</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)

                            with gr.Column(visible=True) as inpaint_mask_generation_col:
                                inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy',
                                                               height=500)
                                inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_sam_prompt_text = gr.Textbox(label='Segmentation prompt', value='', visible=False)
                                with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_sam_quant = gr.Checkbox(label='Quantization', value=False)
                                    inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                generate_mask_button = gr.Button(value='Generate mask from image')


                                def generate_mask(image, mask_model, cloth_category, sam_prompt_text, sam_model, sam_quant, box_threshold, text_threshold):
                                    from extras.inpaint_mask import generate_mask_from_image

                                    extras = {}
                                    if mask_model == 'u2net_cloth_seg':
                                        extras['cloth_category'] = cloth_category
                                    elif mask_model == 'sam':
                                        extras['sam_prompt_text'] = sam_prompt_text
                                        extras['sam_model'] = sam_model
                                        extras['sam_quant'] = sam_quant
                                        extras['box_threshold'] = box_threshold
                                        extras['text_threshold'] = text_threshold

                                    return generate_mask_from_image(image, mask_model, extras)

                            inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg'), gr.update(visible=x == 'sam'), gr.update(visible=x == 'sam')],
                                                      inputs=inpaint_mask_model,
                                                      outputs=[inpaint_mask_cloth_category, inpaint_mask_sam_prompt_text, inpaint_mask_advanced_options],
                                                      queue=False, show_progress=False)

                    with gr.TabItem(label='Describe') as desc_tab:
                        with gr.Row():
                            with gr.Column():
                                desc_input_image = grh.Image(label='Drag any image to here', source='upload', type='filepath')
                            with gr.Column():
                                desc_method = gr.Radio(
                                    label='Content Type',
                                    choices=[flags.desc_type_photo, flags.desc_type_anime],
                                    value=flags.desc_type_photo)
                                desc_btn = gr.Button(value='Describe this Image into Prompt')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Document</a>')

                    with gr.TabItem(label='Metadata') as load_tab:
                        with gr.Column():
                            metadata_input_image = grh.Image(label='Drag any image generated by Fooocus here', source='upload', type='filepath')
                            with gr.Accordion("Preview Metadata", open=True, visible=True) as metadata_preview:
                                metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')

                        def trigger_metadata_preview(filepath):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(filepath)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, _js=switch_js)
            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            desc_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            with gr.Tab(label='Setting'):
                preset_instruction = gr.HTML(visible=False, value=topbar.preset_instruction())
                if not args_manager.args.disable_preset_selection:
                    preset_selection = gr.Radio(label='Preset',
                                                choices=modules.config.available_presets,
                                                value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                interactive=True)
                backend_selection = gr.Radio(label='Generate Engine', choices=flags.backend_engine_list, value=modules.config.default_backend)
                performance_selection = gr.Radio(label='Performance',
                                                 choices=flags.Performance.list(),
                                                 value=modules.config.default_performance)
                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)
                aspect_ratios_selection = gr.Radio(label='Aspect Ratios', choices=modules.config.available_aspect_ratios, value=modules.config.default_aspect_ratio, info='Vertical(9:16), Portrait(4:5), Photo(4:3), Landscape(3:2), Widescreen(16:9), Cinematic(21:9)', elem_classes='aspect_ratios')
                sd3_aspect_ratios_selection = gr.Radio(label='Aspect Ratios', choices=modules.config.sd3_available_aspect_ratios, visible=False, value=modules.config.sd3_default_aspect_ratio, info='Vertical(9:16), Portrait(4:5), Photo(4:3), Landscape(3:2), Widescreen(16:9), Cinematic(21:9)', elem_classes='aspect_ratios_sd3')
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

            with gr.Tab(label='Style', elem_classes=['style_selections_tab']):
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

            with gr.Tab(label='Model'):
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
                with gr.Row():
                    sync_model_info = gr.Checkbox(label='Sync model info', info='Improve usability and transferability for preset and embedinfo.', value=False, container=False)
                    with gr.Column(visible=False) as info_sync_texts:
                        models_infos = []
                        keylist = sorted(models_info.keys())
                        with gr.Tab(label='Checkpoints'):
                            for k in keylist:
                                if k.startswith('checkpoints'):
                                    muid = '' if not models_info[k]['muid'] else models_info[k]['muid']
                                    durl = None if not models_info[k]['url'] else models_info[k]['url']
                                    downURL = gr.Textbox(label=k.split('/')[1], info=f'MUID={muid}', value=durl, placeholder="Type Download URL here.", max_lines=1)
                                    models_infos += [downURL]
                        with gr.Tab(label='LoRAs'):
                            for k in keylist:
                                if k.startswith('loras'):
                                    muid = '' if not models_info[k]['muid'] else models_info[k]['muid']
                                    durl = None if not models_info[k]['url'] else models_info[k]['url']
                                    downURL = gr.Textbox(label=k.split('/')[1], info=f'MUID={muid}', value=durl, placeholder="Type Download URL here.", max_lines=1)
                                    models_infos += [downURL]
                        with gr.Tab(label='Others'):
                            for k in keylist:
                                if not k.startswith('checkpoints') and not k.startswith('loras'):
                                    muid = '' if not models_info[k]['muid'] else models_info[k]['muid']
                                    durl = None if not models_info[k]['url'] else models_info[k]['url']
                                    downURL = gr.Textbox(label=k.split('/')[1], info=f'MUID={muid}', value=durl, placeholder="Type Download URL here.", max_lines=1)
                                    models_infos += [downURL]
                info_sync_button = gr.Button(label='Sync', value='\U0001f504 Sync Remote Info', visible=False, variant='secondary', elem_classes='refresh_button')
                info_progress = gr.Markdown('Note: If MUID is not obtained after synchronizing, it means that it is a new model file. You need to add an available download URL before.', visible=False)
                sync_model_info.change(lambda x: (gr.update(visible=x), gr.update(visible=x),  gr.update(visible=x)), inputs=sync_model_info, outputs=[info_sync_texts, info_sync_button, info_progress], queue=False, show_progress=False)
                info_sync_button.click(sync_model_info_click, inputs=models_infos, outputs=models_infos, queue=False, show_progress=False)

            with gr.Tab(label='Advanced'):
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')
                sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='Higher value means image and texture are sharper.')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Document</a>')
                dev_mode = gr.Checkbox(label='Developer Debug Mode', value=True, container=False)

                with gr.Column(visible=True) as dev_tools:
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
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                   value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                     value=modules.config.default_scheduler)

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
                                                               minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                               info='Set as negative number to disable. For developer debugging.')
                        disable_preview = gr.Checkbox(label='Disable Preview', value=False,
                                                      info='Disable preview during generation.')
                        disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results', 
                                                      value=modules.config.default_performance == flags.Performance.EXTREME_SPEED.value,
                                                      interactive=modules.config.default_performance != flags.Performance.EXTREME_SPEED.value,
                                                      info='Disable intermediate results during generation, only show final gallery.')
                        disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                             info='Disable automatic seed increment when image number is > 1.',
                                                             value=False)
                        read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False, visible=False)

                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                  info='Adds parameters to generated images allowing manual regeneration.')
                            metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                                       visible=modules.config.default_save_metadata_to_images)

                            save_metadata_to_images.change(lambda x: [gr.update(visible=x), gr.update(visible=not x)], inputs=[save_metadata_to_images], outputs=[metadata_scheme, prompt_embed_button], queue=False, show_progress=False)

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        #mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                        #                                                   value=False)
                        #mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                        #                                              value=False)

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
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model')
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
                                                                 'negative value will make white area smaller.'
                                                                 '(default is 0, always process before any mask invert)')
                        #inpaint_mask_upload_checkbox = gr.Checkbox(label='Enable Mask Upload', value=True)
                        #invert_mask_checkbox = gr.Checkbox(label='Invert Mask', value=False)

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_mask_upload_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                            inputs=inpaint_mask_upload_checkbox,
                                                            outputs=[inpaint_mask_image, inpaint_mask_generation_col],
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

                def refresh_files_clicked():
                    modules.config.update_files()
                    results = [gr.update(choices=modules.config.model_filenames)]
                    results += [gr.update(choices=['None'] + modules.config.model_filenames)]
                    if not args_manager.args.disable_preset_selection:
                        results += [gr.update(choices=modules.config.available_presets)]
                    for i in range(modules.config.default_max_lora_number):
                        results += [gr.update(interactive=True),
                                    gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
                    return results

                refresh_files_output = [base_model, refiner_model]
                if not args_manager.args.disable_preset_selection:
                    refresh_files_output += [preset_selection]
                refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                                    queue=False, show_progress=False)

            with gr.Tab(label='Enhanced'):
                with gr.Row():
                    language_ui = gr.Radio(label='Language of UI', choices=['En', '中文'], value=location.language_radio(args_manager.args.language), interactive=(args_manager.args.language in ['default', 'cn', 'en']))
                    background_theme = gr.Radio(label='Theme of background', choices=['light', 'dark'], value=args_manager.args.theme, interactive=True)
                with gr.Group():
                    backfill_prompt = gr.Checkbox(label='Backfill prompt while switching images', value=modules.config.default_backfill_prompt, interactive=True, info='Extract and backfill prompt and negative prompt while switching historical gallery images.')
                    prompt_preset_button = gr.Button(value='Save the current parameters as a preset package')
                    translation_timing = gr.Radio(label='Timing of translation', choices=modules.flags.translation_timing, value=modules.config.default_translation_timing, info='The selection of timing for prompt translation.')
                    translation_methods = gr.Radio(label='Translation methods', choices=modules.flags.translation_methods, value=modules.config.default_translation_methods, info='\'Model\' requires more GPU/CPU and \'APIs\' rely on third.')
                    mobile_url = gr.Checkbox(label=f'http://{args_manager.args.listen}:{args_manager.args.port}{args_manager.args.webroot}/', value=True, info='Mobile phone access address within the LAN. If you want WAN access, consulting QQ group: 938075852.', interactive=False)
                
                # custom plugin "OneButtonPrompt"
                import custom.OneButtonPrompt.ui_onebutton as ui_onebutton
                run_event = gr.Number(visible=False, value=0)
                ui_onebutton.ui_onebutton(prompt, run_event, random_button)
                with gr.Tab(label="SuperPrompter"):
                    #super_prompter = gr.Button(value="<<SuperPrompt", size="sm", min_width = 70)
                    super_prompter_prompt = gr.Textbox(label='Prompt prefix', value='Expand the following prompt to add more detail:', lines=1)
                with gr.Row():
                    gr.Markdown(value=f'VERSION: {version.branch} {version.simplesdxl_ver} / Fooocus {fooocus_version.version}')

            translation_timing.change(lambda x: gr.update(interactive=not (x=='No translate')), inputs=translation_timing, outputs=translation_methods, queue=False, show_progress=False)
            import enhanced.superprompter
            super_prompter.click(lambda x, y, z: enhanced.superprompter.answer(input_text=translator.convert(f'{y}{x}', z), seed=image_seed), inputs=[prompt, super_prompter_prompt, translation_methods], outputs=prompt, queue=False, show_progress=True)
            ehps = [backfill_prompt, translation_timing, translation_methods, backend_selection, sd3_aspect_ratios_selection]
            language_ui.select(None, inputs=language_ui, _js="(x) => set_language_by_ui(x)")
            background_theme.select(None, inputs=background_theme, _js="(x) => set_theme_by_ui(x)")
           
            generate_mask_button.click(enhanced_parameters.set_all_enhanced_parameters, inputs=ehps) \
                                 .then(fn=generate_mask,inputs=[
                                                               inpaint_input_image, inpaint_mask_model,
                                                               inpaint_mask_cloth_category,
                                                               inpaint_mask_sam_prompt_text,
                                                               inpaint_mask_sam_model,
                                                               inpaint_mask_sam_quant,
                                                               inpaint_mask_box_threshold,
                                                               inpaint_mask_text_threshold
                                                           ], outputs=inpaint_mask_image, show_progress=True, queue=True)

            gallery_index.select(gallery_util.select_index, inputs=[gallery_index, state_topbar], outputs=[gallery, progress_window, progress_gallery, prompt_info_box, params_note_box, image_tools_checkbox, state_topbar], show_progress=False)
            gallery.select(gallery_util.select_gallery, inputs=[gallery_index, state_topbar, backfill_prompt], outputs=[prompt_info_box, prompt, negative_prompt, params_note_info, params_note_input_name, params_note_regen_button, params_note_preset_button, state_topbar], show_progress=False)
            progress_gallery.select(gallery_util.select_gallery_progress, inputs=state_topbar, outputs=[prompt_info_box, params_note_info, params_note_input_name, params_note_regen_button, params_note_preset_button, state_topbar], show_progress=False)

        state_is_generating = gr.State(False)

        load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, base_model,
                             refiner_model, refiner_switch, sampler_name, scheduler_name, seed_random, image_seed,
                             generate_button, load_parameter_button] + freeu_ctrls + lora_ctrls

        if not args_manager.args.disable_preset_selection:
            def preset_selection_change(preset, is_generating):
                preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

                default_model = preset_prepared.get('base_model')
                previous_default_models = preset_prepared.get('previous_default_models', [])
                checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
                embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
                lora_downloads = preset_prepared.get('lora_downloads', {})

                preset_prepared['base_model'], preset_prepared['lora_downloads'] = launch.download_models(
                    default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads)

                if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                    del preset_prepared['prompt']

                return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating)

            preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \

        performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                               [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                               [gr.update(interactive=not flags.Performance.has_restricted_features(x), value=flags.Performance.has_restricted_features(x))] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                     ], queue=False, show_progress=False)
        
        def toggle_engine(x):
            if x=='SD3 Api' or x=='SD3Turbo Api':
                results = [gr.update(value=False, interactive=False), gr.update(visible=False), gr.update(value=1, interactive=False), gr.update(visible=False), gr.update(visible=True), gr.update(choices=['png', 'jpeg'], value='png')]
            else:
                results = [gr.update(interactive=True), gr.update(visible=True), gr.update(value=modules.config.default_image_number, interactive=True), gr.update(visible=True), gr.update(visible=False), gr.update(choices=flags.OutputFormat.list(), value=modules.config.default_output_format)]
            return results

        backend_selection.change(toggle_engine, inputs=backend_selection, outputs = [input_image_checkbox, performance_selection, image_number, aspect_ratios_selection, sd3_aspect_ratios_selection, output_format], queue=False, show_progress=False)

        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)
        
        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)

        def inpaint_mode_change(mode):
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

            if mode == modules.flags.inpaint_option_modify:
                return [
                    gr.update(visible=True), gr.update(visible=False, value=[]),
                    gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
                    True, modules.config.default_inpaint_engine_version, 1.0, 0.0
                ]

            return [
                gr.update(visible=False, value=''), gr.update(visible=True),
                gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
                False, modules.config.default_inpaint_engine_version, 1.0, 0.618
            ]

        inpaint_mode.input(inpaint_mode_change, inputs=inpaint_mode, outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

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
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg]
        ctrls += [sampler_name, scheduler_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls

        if not args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        ctrls += ip_ctrls
        
        system_params = gr.JSON({}, visible=False)
        state_is_generating = gr.State(False)
        def parse_meta(raw_prompt_txt, is_generating, state_params):
            loaded_json = None
            if len(raw_prompt_txt)>=1 and (raw_prompt_txt[-1]=='[' or raw_prompt_txt[-1]=='_'):
                return [gr.update()] * 4 + wildcards_array_show(state_params)
            matchs = wildcards.array_regex.findall(raw_prompt_txt)
            matchs2 = wildcards.tag_regex2.findall(raw_prompt_txt)
            if len(matchs)>0 or len(matchs2)>0:
                wildcards_array_results =  wildcards_array_show(state_params)
            else:
                wildcards_array_results =  wildcards_array_hidden
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
                    return [gr.update()] * 4 + wildcards_array_results
                else:
                    return [gr.update(), gr.update(visible=True), gr.update(visible=False)] + wildcards_array_results

            return [json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)] + wildcards_array_results

        prompt.input(parse_meta, inputs=[prompt, state_is_generating, state_topbar], outputs=[prompt, generate_button, load_parameter_button] + wildcards_array, queue=False, show_progress=False)
        
        translator_button.click(lambda x, y: translator.convert(x, y), inputs=[prompt, translation_methods], outputs=prompt, queue=False, show_progress=True)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(filepath, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(filepath)
            if parameters is None:
                print('Could not find metadata in the image!')
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.parse_json(parameters)
            return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating)

        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

        reset_params = [prompt, negative_prompt, style_selections, performance_selection, aspect_ratios_selection, sharpness, guidance_scale, base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, adaptive_cfg, overwrite_step, overwrite_switch, inpaint_engine] + lora_ctrls + [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, seed_random, image_seed] + freeu_ctrls
        reset_params_in = ctrls[2:] + [seed_random]
        model_check = [prompt, negative_prompt, base_model, refiner_model] + lora_ctrls
        nav_bars = [bar_title, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button]

        generate_button.click(topbar.process_before_generation, inputs=state_topbar, outputs=[prompt, stop_button, skip_button, generate_button, gallery, state_is_generating, index_radio, image_tools_checkbox, background_theme, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button], show_progress=False) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
            .then(enhanced_parameters.set_all_enhanced_parameters, inputs=ehps) \
            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
            .then(topbar.process_after_generation, inputs=state_topbar, outputs=[generate_button, prompt, stop_button, skip_button, state_is_generating, gallery_index, index_radio, background_theme, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button], show_progress=False) \
            .then(fn=update_history_link, outputs=history_link) \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(mode, img):
            if mode == flags.desc_type_photo:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                return default_interrogator_photo(img)
            if mode == flags.desc_type_anime:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                return default_interrogator_anime(img)
            return mode
        
        def trigger_describe_pre(mode, img_path):
            img_rgb = Image.open(img_path).convert('RGB')
            return trigger_describe(mode, img_rgb)

        desc_btn.click(trigger_describe_pre, inputs=[desc_method, desc_input_image],
                       outputs=[prompt], show_progress=True, queue=True)

        def trigger_uov_describe(mode, img, prompt):
            # keep prompt if not empty
            if prompt == '':
                return trigger_describe(mode, img)
            return prompt

        uov_input_image.upload(trigger_uov_describe, inputs=[desc_method, uov_input_image, prompt],
                       outputs=[prompt], show_progress=False, queue=True)


    prompt_delete_button.click(toolbox.toggle_note_box_delete, inputs=state_topbar, outputs=[params_note_info, params_note_delete_button, params_note_box, state_topbar], show_progress=False)
    params_note_delete_button.click(toolbox.delete_image, inputs=state_topbar, outputs=[gallery, gallery_index, params_note_delete_button, params_note_box, state_topbar], show_progress=False)
    
    prompt_regen_button.click(toolbox.toggle_note_box_regen, inputs=model_check + [state_topbar], outputs=[params_note_info, params_note_regen_button, params_note_box, state_topbar], show_progress=False)
    params_note_regen_button.click(toolbox.reset_image_params, inputs=state_topbar, outputs=reset_params + [params_note_regen_button, params_note_box, state_topbar], show_progress=False)

    prompt_preset_button.click(toolbox.toggle_note_box_preset, inputs=model_check + [state_topbar], outputs=[params_note_info, params_note_input_name, params_note_preset_button, params_note_box, state_topbar], show_progress=False)
    params_note_preset_button.click(toolbox.save_preset, inputs= reset_params_in + [params_note_input_name, state_topbar], outputs=[params_note_input_name, params_note_preset_button, params_note_box, state_topbar] + nav_bars, show_progress=False) \
        .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, queue=False, show_progress=False) \
        .then(fn=lambda x: None, inputs=system_params, _js=topbar.refresh_topbar_status_js)

    prompt_embed_button.click(toolbox.toggle_note_box_embed, inputs=model_check + [state_topbar], outputs=[params_note_info, params_note_embed_button, params_note_box, state_topbar], show_progress=False)
    params_note_embed_button.click(toolbox.embed_params, inputs=state_topbar, outputs=[params_note_embed_button, params_note_box, state_topbar], show_progress=False)
    
    reset_preset_fun = [preset_instruction, image_number, inpaint_mask_upload_checkbox, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, backfill_prompt, translation_timing, translation_methods]
    reset_preset_all = [backend_selection] + reset_params + reset_preset_fun + nav_bars + [output_format] + [state_topbar]
    
    preset_down_confirm_button.click(topbar.down_absent_model, inputs=state_topbar, outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=state_topbar, outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    preset_down_cancel_button.click(fn=lambda: gr.update(visible=False), outputs=preset_down_note, show_progress=False)

    bar0_button.click(topbar.check_absent_model, inputs=[bar0_button, state_topbar], outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar1_button.click(topbar.check_absent_model, inputs=[bar1_button, state_topbar], outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar2_button.click(topbar.check_absent_model, inputs=[bar2_button, state_topbar], outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar3_button.click(topbar.check_absent_model, inputs=[bar3_button, state_topbar], outputs=[preset_down_note, state_topbar], show_progress=False) \
               .then(topbar.reset_params_for_preset, inputs=state_topbar, outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar4_button.click(topbar.check_absent_model, inputs=[bar4_button, state_topbar], outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar5_button.click(topbar.check_absent_model, inputs=[bar5_button, state_topbar], outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar6_button.click(topbar.check_absent_model, inputs=[bar6_button, state_topbar], outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar7_button.click(topbar.check_absent_model, inputs=[bar7_button, state_topbar], outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar8_button.click(topbar.check_absent_model, inputs=[bar8_button, state_topbar], outputs=[preset_down_note, state_topbar]) \
               .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')

    shared.gradio_root.load(fn=lambda x: x, inputs=system_params, outputs=state_topbar, _js=topbar.get_system_params_js, queue=False, show_progress=False) \
                      .then(topbar.init_nav_bars, inputs=state_topbar, outputs=nav_bars + [progress_window, language_ui, background_theme, gallery_index, index_radio, image_tools_checkbox, inpaint_mask_upload_checkbox, preset_instruction], show_progress=False) \
                      .then(topbar.reset_params_for_preset, inputs=[state_topbar], outputs=reset_preset_all, show_progress=False) \
                      .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
                      .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
                      .then(topbar.sync_message, inputs=state_topbar, outputs=[state_topbar]).then(fn=lambda: None, _js='refresh_grid_delayed')


def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


#dump_default_english_config()

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

