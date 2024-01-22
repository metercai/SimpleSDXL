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
import modules.advanced_parameters as advanced_parameters
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth

import enhanced.gallery as gallery_util
import enhanced.topbar  as topbar
import enhanced.toolbox  as toolbox
import enhanced.translator  as translator
import enhanced.enhanced_parameters as enhanced_parameters
import enhanced.version as version
import enhanced.location as location
from enhanced.models_info import models_info, sync_model_info_click 

def generate_clicked(*args):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    execution_start_time = time.perf_counter()
    task = worker.AsyncTask(args=list(args))
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

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


reload_javascript()

title = f'{version.branch} {version.simplesdxl_ver} derived from Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(
    title=title,
    css=modules.html.css + topbar.css + toolbox.css).queue()

with shared.gradio_root:
    state_topbar = gr.State({})
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
                    progress_window = grh.Image(label='Preview', show_label=True, visible=False, height=520, elem_id='preview_generating',
                                            elem_classes=['main_view'])
                    progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain', elem_id='finished_gallery',
                                              height=520, visible=False, elem_classes=['main_view', 'image_gallery'])
                progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
                gallery = gr.Gallery(label='Gallery', show_label=True, object_fit='contain', visible=True, height=520,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery', preview=True)
                prompt_info_box = gr.Markdown(toolbox.make_infobox_markdown(None), visible=False, elem_id='infobox', elem_classes='infobox')
                with gr.Group(visible=False, elem_classes='toolbox_note') as params_note_box:
                    params_note_info = gr.Markdown(elem_classes='note_info')
                    params_note_input_name = gr.Textbox(show_label=False, placeholder="Type preset name here.", min_width=100, elem_classes='preset_input', visible=False)
                    params_note_delete_button = gr.Button(value='Enter', visible=False)
                    params_note_regen_button = gr.Button(value='Enter', visible=False)
                    params_note_preset_button = gr.Button(value='Enter', visible=False)
                    params_note_embed_button = gr.Button(value='Enter', visible=False)
                with gr.Accordion("Finished Images Index:", open=False, visible=False) as index_radio:
                    gallery_index = gr.Radio(choices=None, label="Gallery_Index", value=None, show_label=False)
                    gallery_index.change(gallery_util.images_list_update, inputs=[gallery_index, state_topbar], outputs=[gallery, index_radio, state_topbar], show_progress=False)
            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        container=False, autofocus=False, elem_classes='type_row', lines=1024)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    translator_button = gr.Button(label="Translator", value="Translator", elem_classes='type_row', elem_id='translator_button', visible=False)
                    skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', visible=False)
                    stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked():
                        import ldm_patched.modules.model_management as model_management
                        shared.last_stop = 'stop'
                        model_management.interrupt_current_processing()
                        return [gr.update(interactive=False)] * 2

                    def skip_clicked():
                        import ldm_patched.modules.model_management as model_management
                        shared.last_stop = 'skip'
                        model_management.interrupt_current_processing()
                        return

                    stop_button.click(stop_clicked, outputs=[skip_button, stop_button],
                                      queue=False, show_progress=False, _js='cancelGenerateForever')
                    skip_button.click(skip_clicked, queue=False, show_progress=False)
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=False, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced+', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
                image_tools_checkbox = gr.Checkbox(label='ParamsTools', value=False, container=False, elem_classes='min_check')
            with gr.Group(visible=False, elem_classes='toolbox') as image_toolbox:
                image_tools_box_title = gr.Markdown('<b>ToolBox</b>', visible=True)
                prompt_info_button = gr.Button(value='ViewMeta', size='sm', visible=True)
                prompt_regen_button = gr.Button(value='ReGenerate', size='sm', visible=True)
                prompt_preset_button = gr.Button(value='SavePreset', size='sm', visible=True)
                prompt_embed_button = gr.Button(value='EmbedMeta', size='sm', visible=True)
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
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=flags.disabled)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Document</a>')
                    with gr.TabItem(label='Image Prompt') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for _ in range(4):
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
                        ip_advanced = gr.Checkbox(label='Advanced Control', value=False, container=False)
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
                            inpaint_input_image = grh.Image(label='Drag inpaint or outpaint image to here', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas')
                            inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', height=500, visible=True)

                        with gr.Row():
                            inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                            outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                            inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.flags.inpaint_option_default, label='Method')
                        example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts, label='Additional Prompt Quick List', components=[inpaint_additional_prompt], visible=False)
                        gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Document</a>')
                        example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)
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
                                params_btn = gr.Button(value='Extract params from Image and Reset')
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
                preset_instruction = gr.HTML(visible=False)
                performance_selection = gr.Radio(label='Performance',
                                                 choices=modules.flags.performance_selections,
                                                 value=modules.config.default_performance)
                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)
                aspect_ratios_selection = gr.Radio(label='Aspect Ratios', choices=modules.config.available_aspect_ratios,
                                                   value=modules.config.default_aspect_ratio, info='Vertical(9:16), Portrait(4:5), Photo(4:3), Landscape(3:2), Widescreen(16:9), Cinematic(21:9)',
                                                   elem_classes='aspect_ratios')
                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type negative prompt here.",
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

                if not args_manager.args.disable_image_log:
                    gr.HTML(f'<a href="{args_manager.args.webroot}/file={get_current_html_path()}" target="_blank">\U0001F4DA History Log</a>')

            with gr.Tab(label='Style'):
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

                    for i, (n, v) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=n)
                            lora_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.01, value=v,
                                                    elem_classes='lora_weight')
                            lora_ctrls += [lora_model, lora_weight]

                with gr.Row():
                    model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
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

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value='joint',
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

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

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
                        inpaint_mask_upload_checkbox = gr.Checkbox(label='Enable Mask Upload', value=True)
                        invert_mask_checkbox = gr.Checkbox(label='Invert Mask', value=False)
                        
                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_mask_upload_checkbox.change(lambda x: gr.update(visible=x),
                                                           inputs=inpaint_mask_upload_checkbox,
                                                           outputs=inpaint_mask_image, queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                adps = [disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name,
                        scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height,
                        overwrite_vary_strength, overwrite_upscale_strength,
                        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint,
                        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness,
                        canny_low_threshold, canny_high_threshold, refiner_swap_method]
                adps += freeu_ctrls
                adps += inpaint_ctrls

                def dev_mode_checked(r):
                    return gr.update(visible=r)


                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def model_refresh_clicked():
                    modules.config.update_all_model_names()
                    results = []
                    results += [gr.update(choices=modules.config.model_filenames), gr.update(choices=['None'] + modules.config.model_filenames)]
                    for i in range(5):
                        results += [gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
                    return results

                model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls,
                                    queue=False, show_progress=False)

            with gr.Tab(label='Enhanced'):
                with gr.Row():
                    language_ui = gr.Radio(label='Language of UI', choices=['En', '中文'], value=location.language_radio(args_manager.args.language), interactive=(args_manager.args.language in ['default', 'cn', 'en']))
                    background_theme = gr.Radio(label='Theme of background', choices=['light', 'dark'], value=args_manager.args.theme, interactive=True)
                with gr.Group():
                    backfill_prompt = gr.Checkbox(label='Backfill prompt while switching images', value=False, interactive=True, info='Extract and backfill prompt and negative prompt while switching historical gallery images.')
                    translation_timing = gr.Radio(label='Timing of translation', choices=['No translation', 'Translate then generate', 'Modify after translation'], value='Translate then generate', info='The selection of timing for prompt translation.')
                    translation_methods = gr.Radio(label='Translation methods', choices=['Model in local','APIs on third'], value='APIs on third', info='\'Model\' requires more GPU/CPU and \'APIs\' rely on third.')
                    mobile_url = gr.Checkbox(label=f'http://{args_manager.args.listen}:{args_manager.args.port}{args_manager.args.webroot}/', value=True, info='Mobile phone access address within the LAN. If you want WAN access, consulting QQ group: 938075852.', interactive=False)

            ehps = [backfill_prompt, translation_timing, translation_methods]
            language_ui.select(None, inputs=language_ui, _js="(x) => set_language_by_ui(x)")
            background_theme.select(None, inputs=background_theme, _js="(x) => set_theme_by_ui(x)")
            
            gallery_index.select(gallery_util.select_index, inputs=[gallery_index, state_topbar], outputs=[gallery, progress_gallery, prompt_info_box, params_note_box, image_tools_checkbox, state_topbar], show_progress=False)
            gallery.select(gallery_util.select_gallery, inputs=[gallery_index, state_topbar, backfill_prompt], outputs=[prompt_info_box, prompt, negative_prompt, params_note_info, params_note_input_name, params_note_regen_button, params_note_preset_button, state_topbar], show_progress=False)
            progress_gallery.select(gallery_util.select_gallery_progress, inputs=state_topbar, outputs=[prompt_info_box, params_note_info, params_note_input_name, params_note_regen_button, params_note_preset_button, state_topbar], show_progress=False)

        performance_selection.change(lambda x: [gr.update(interactive=x != 'Extreme Speed')] * 11 +
                                               [gr.update(visible=x != 'Extreme Speed')] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt
                                     ], queue=False, show_progress=False)

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

        ctrls = [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, image_seed, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += ip_ctrls
        
        system_params = gr.JSON({}, visible=False)
        state_is_generating = gr.State(False)
        def parse_meta(raw_prompt_txt, is_generating, timing):
            loaded_json = None
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
                    return gr.update(), gr.update(), gr.update(), gr.update()
                else:
                    flag = (timing=='Modify after translation' and translator.is_chinese(raw_prompt_txt))
                    return gr.update(), gr.update(visible=not flag), gr.update(visible=False), gr.update(visible=flag)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

        prompt.input(parse_meta, inputs=[prompt, state_is_generating, translation_timing], outputs=[prompt, generate_button, load_parameter_button, translator_button], queue=False, show_progress=False)
        
        translator_button.click(lambda x: [gr.update(value=translator.convert(x)), gr.update(visible=True), gr.update(visible=False)], inputs=prompt, outputs=[prompt, generate_button, translator_button], queue=False, show_progress=False)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating], outputs=[
            advanced_checkbox,
            image_number,
            prompt,
            negative_prompt,
            style_selections,
            performance_selection,
            aspect_ratios_selection,
            overwrite_width,
            overwrite_height,
            sharpness,
            guidance_scale,
            adm_scaler_positive,
            adm_scaler_negative,
            adm_scaler_end,
            base_model,
            refiner_model,
            refiner_switch,
            sampler_name,
            scheduler_name,
            seed_random,
            image_seed,
            generate_button,
            load_parameter_button
        ] + lora_ctrls, queue=False, show_progress=False)

        reset_preset = [prompt, negative_prompt, style_selections, performance_selection, aspect_ratios_selection, sharpness, guidance_scale, base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, adaptive_cfg, overwrite_step, overwrite_switch, inpaint_engine] + lora_ctrls
        reset_params = reset_preset + [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, seed_random, image_seed]
        model_check = [prompt, negative_prompt, base_model, refiner_model] + lora_ctrls
        nav_bars = [bar_title, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button]

        generate_button.click(topbar.process_before_generation, inputs=state_topbar, outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating, index_radio, image_tools_checkbox, background_theme, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button], show_progress=False) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(advanced_parameters.set_all_advanced_parameters, inputs=adps) \
            .then(enhanced_parameters.set_all_enhanced_parameters, inputs=ehps) \
            .then(fn=generate_clicked, inputs=ctrls, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
            .then(topbar.process_after_generation, inputs=state_topbar, outputs=[generate_button, stop_button, skip_button, state_is_generating, gallery_index, index_radio, background_theme, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button], show_progress=False) \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(mode, img):
            if mode == flags.desc_type_photo:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                return default_interrogator_photo(img), ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]
            if mode == flags.desc_type_anime:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                return default_interrogator_anime(img), ["Fooocus V2", "Fooocus Masterpiece"]
            return mode, ["Fooocus V2"]

        desc_btn.click(trigger_describe, inputs=[desc_method, desc_input_image],
                       outputs=[prompt, style_selections], show_progress=True, queue=True)
        params_btn.click(toolbox.extract_reset_image_params, inputs=desc_input_image, outputs=reset_params, show_progress=False, queue=True) \
                   .then(fn=lambda: None, _js=toolbox.extract_reset_image_params_js)

    prompt_delete_button.click(toolbox.toggle_note_box_delete, inputs=state_topbar, outputs=[params_note_info, params_note_delete_button, params_note_box, state_topbar], show_progress=False)
    params_note_delete_button.click(toolbox.delete_image, inputs=state_topbar, outputs=[gallery, gallery_index, params_note_delete_button, params_note_box, state_topbar], show_progress=False)
    
    prompt_regen_button.click(toolbox.toggle_note_box_regen, inputs=model_check + [state_topbar], outputs=[params_note_info, params_note_regen_button, params_note_box, state_topbar], show_progress=False)
    params_note_regen_button.click(toolbox.reset_image_params, inputs=state_topbar, outputs=reset_params + [params_note_regen_button, params_note_box, state_topbar], show_progress=False)

    prompt_preset_button.click(toolbox.toggle_note_box_preset, inputs=model_check + [state_topbar], outputs=[params_note_info, params_note_input_name, params_note_preset_button, params_note_box, state_topbar], show_progress=False)
    params_note_preset_button.click(toolbox.save_preset, inputs= [params_note_input_name, state_topbar] + reset_params, outputs=[params_note_input_name, params_note_preset_button, params_note_box, state_topbar] + nav_bars, show_progress=False) \
        .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, queue=False, show_progress=False) \
        .then(fn=lambda x: None, inputs=system_params, _js=topbar.refresh_topbar_status_js)

    prompt_embed_button.click(toolbox.toggle_note_box_embed, inputs=model_check + [state_topbar], outputs=[params_note_info, params_note_embed_button, params_note_box, state_topbar], show_progress=False)
    params_note_embed_button.click(toolbox.embed_params, inputs=state_topbar, outputs=[params_note_embed_button, params_note_box, state_topbar], show_progress=False)

    reset_preset_all = reset_params + [gallery, gallery_index, preset_instruction, state_topbar, image_number, inpaint_mask_upload_checkbox]
    bar0_button.click(topbar.reset_params_for_preset, inputs=[bar0_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar1_button.click(topbar.reset_params_for_preset, inputs=[bar1_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar2_button.click(topbar.reset_params_for_preset, inputs=[bar2_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar3_button.click(topbar.reset_params_for_preset, inputs=[bar3_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar4_button.click(topbar.reset_params_for_preset, inputs=[bar4_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar5_button.click(topbar.reset_params_for_preset, inputs=[bar5_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar6_button.click(topbar.reset_params_for_preset, inputs=[bar6_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar7_button.click(topbar.reset_params_for_preset, inputs=[bar7_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')
    bar8_button.click(topbar.reset_params_for_preset, inputs=[bar8_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
               .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
               .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
               .then(fn=lambda: None, _js='refresh_grid_delayed')

    shared.gradio_root.load(fn=lambda x: x, inputs=system_params, outputs=state_topbar, _js=topbar.get_system_params_js, queue=False, show_progress=False) \
                      .then(topbar.init_nav_bars, inputs=state_topbar, outputs=nav_bars + [language_ui, background_theme, gallery_index, index_radio], show_progress=False) \
                      .then(topbar.reset_params_for_preset, inputs=[bar0_button, state_topbar], outputs=reset_preset_all, show_progress=False) \
                      .then(fn=lambda x: x, inputs=state_topbar, outputs=system_params, show_progress=False) \
                      .then(fn=lambda x: {}, inputs=system_params, outputs=system_params, _js=topbar.refresh_topbar_status_js) \
                      .then(topbar.sync_message, inputs=state_topbar, outputs=state_topbar).then(fn=lambda: None, _js='refresh_grid_delayed')


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
    auth=check_auth if auth_enabled else None,
    blocked_paths=[constants.AUTH_FILENAME]
)

