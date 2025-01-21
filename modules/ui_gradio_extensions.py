# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import gradio as gr
import args_manager
import modules.config

from modules.localization import localization_js


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'

def load_tips_text():
    tips_path = os.path.join(script_path, 'tips.txt')
    tips_text = ''
    if os.path.exists(tips_path):
        try:
            with open(tips_path, encoding='utf-8') as f:
                tips_text = f.readlines()
            tips_text = [line.strip() for line in tips_text if line.strip()]
            tips_text = ','.join([f'"{line}"' for line in tips_text])
        except Exception as e:
            logger.info(str(e))
            logger.info(f'Failed to load tips file {tips_path}')
    return f'let tips = [{tips_text}];'

def javascript_html():
    script_js_path = webpath('javascript/script.js')
    context_menus_js_path = webpath('javascript/contextMenus.js')
    localization_js_path = webpath('javascript/localization.js')
    zoom_js_path = webpath('javascript/zoom.js')
    edit_attention_js_path = webpath('javascript/edit-attention.js')
    viewer_js_path = webpath('javascript/viewer.js')
    image_viewer_js_path = webpath('javascript/imageviewer.js')
    topbar_js_path = webpath('javascript/topbar.js')
    canvg_min_js_path = webpath('javascript/umd.min.js') #canvg.min.js')
    samples_path = webpath(os.path.abspath('./sdxl_styles/samples/fooocus_v2.jpg'))
    preset_samples_path = webpath(os.path.abspath('./presets/samples/default.jpg'))
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    head += f'<script type="text/javascript">{load_tips_text()}</script>\n'
    head += f'<script type="text/javascript" src="{script_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{context_menus_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{localization_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{zoom_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{edit_attention_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{viewer_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{image_viewer_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{topbar_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{canvg_min_js_path}"></script>\n'
    head += f'<meta name="samples-path" content="{samples_path}">\n'
    head += f'<meta name="preset-samples-path" content="{preset_samples_path}">\n'

    theme = args_manager.args.theme if args_manager.args.theme else "light"
    head += f'<script type="text/javascript">set_theme(\"{theme}\");</script>\n'

    return head


def css_html():
    style_css_path = webpath('css/style.css')
    head = f'<link rel="stylesheet" property="stylesheet" href="{style_css_path}">'
    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
