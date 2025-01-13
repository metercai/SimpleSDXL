import os
import args_manager
import modules.config
import json
import urllib.parse
import enhanced.all_parameters as ads
import shared

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from modules.flags import OutputFormat
from modules.meta_parser import MetadataParser, get_exif
from modules.util import generate_temp_filename
import logging
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

log_cache = {}


def get_current_html_path(output_format=None, user_did=None):

    if not user_did:
        user_did = shared.token.get_guest_did()
    user_path_outputs = modules.config.get_user_path_outputs(user_did)
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=user_path_outputs,
                                                                         extension=output_format)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')
    return html_name


def log(img, metadata, metadata_parser: MetadataParser | None = None, output_format=None, task=None, persist_image=True, user_did=None) -> str:
    if not user_did:
        user_did = shared.token.get_guest_did()
    user_path_outputs = modules.config.get_user_path_outputs(user_did)
    
    path_outputs = modules.config.temp_path if args_manager.args.disable_image_log or not persist_image else user_path_outputs
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=path_outputs, extension=output_format)
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)

    parsed_parameters = metadata_parser.to_string(metadata.copy()) if metadata_parser is not None else ''
    metadata_scheme = metadata_parser.get_scheme().value if metadata_parser is not None else ''

    image = Image.fromarray(img)

    if output_format == OutputFormat.PNG.value or (image.mode == 'RGBA' and output_format == OutputFormat.JPEG.value):
        if metadata_scheme == 'simple':
            pnginfo = PngInfo()
            pnginfo.add_text("Comment", parsed_parameters)
        elif metadata_parser:
            pnginfo = PngInfo()
            pnginfo.add_text('parameters', parsed_parameters)
            pnginfo.add_text('fooocus_scheme', metadata_parser.get_scheme().value)
        else:
            pnginfo = None
        if output_format == OutputFormat.JPEG.value:
            local_temp_filename = local_temp_filename[:-4] + "png"
        image.save(local_temp_filename, pnginfo=pnginfo)
    elif output_format == OutputFormat.JPEG.value and image.mode != 'RGBA':
        image.save(local_temp_filename, quality=95, optimize=True, progressive=True, exif=get_exif(parsed_parameters, metadata_scheme) if metadata_parser else Image.Exif())
    elif output_format == OutputFormat.WEBP.value:
        image.save(local_temp_filename, quality=95, lossless=False, exif=get_exif(parsed_parameters, metadata_scheme) if metadata_parser else Image.Exif())
    else:
        image.save(local_temp_filename)

    if args_manager.args.disable_image_log:
        return local_temp_filename


    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')

    css_styles = (
        "<style>"
        "body { background-color: #121212; color: #E0E0E0; } "
        "a { color: #BB86FC; } "
        ".metadata { border-collapse: collapse; width: 100%; } "
        ".metadata .label { width: 15%; } "
        ".metadata .value { width: 85%; font-weight: bold; } "
        ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
        ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
        ".image-container div { text-align: center; padding: 4px; } "
        "hr { border-color: gray; } "
        "button { background-color: black; color: white; border: 1px solid grey; border-radius: 5px; padding: 5px 10px; text-align: center; display: inline-block; font-size: 16px; cursor: pointer; }"
        "button:hover {background-color: grey; color: black;}"
        "</style>"
    )

    js = (
        """<script>
        function to_clipboard(txt) { 
        txt = decodeURIComponent(txt);
        if (navigator.clipboard && navigator.permissions) {
            navigator.clipboard.writeText(txt)
        } else {
            const textArea = document.createElement('textArea')
            textArea.value = txt
            textArea.style.width = 0
            textArea.style.position = 'fixed'
            textArea.style.left = '-999px'
            textArea.style.top = '10px'
            textArea.setAttribute('readonly', 'readonly')
            document.body.appendChild(textArea)

            textArea.select()
            document.execCommand('copy')
            document.body.removeChild(textArea)
        }
        alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.\\nCurrent clipboard content is:\\n\\n' + txt);
        }
        </script>"""
    )

    begin_part = f"<!DOCTYPE html><html><head><title>Fooocus Log {date_string}</title>{css_styles}</head><body>{js}<p>Fooocus Log {date_string} (private)</p>\n<p>Metadata is embedded if enabled in the config or developer debug mode. You can find the information for each image in line Metadata Scheme.</p><!--fooocus-log-split-->\n\n"
    end_part = f'\n<!--fooocus-log-split--></body></html>'

    middle_part = log_cache.get(html_name, "")

    if middle_part == "":
        if os.path.exists(html_name):
            existing_split = open(html_name, 'r', encoding='utf-8').read().split('<!--fooocus-log-split-->')
            if len(existing_split) == 3:
                middle_part = existing_split[1]
            else:
                middle_part = existing_split[0]

    div_name = only_name.replace('.', '_')
    item = f"<div id=\"{div_name}\" class=\"image-container\"><hr><table><tr>\n"
    item += f"<td><a href=\"{only_name}\" target=\"_blank\"><img src='{only_name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'/></a><div>{only_name}</div></td>"
    item += "<td><table class='metadata'>"
    for label, key, value in metadata:
        value_txt = str(value).replace('\n', ' </br> ')
        item += f"<tr><td class='label'>{label}</td><td class='value'>{value_txt}</td></tr>\n"

    if task is not None and 'positive' in task and 'negative' in task:
        full_prompt_details = f"""<details><summary>Positive</summary>{', '.join(task['positive'])}</details>
        <details><summary>Negative</summary>{', '.join(task['negative'])}</details>"""
        item += f"<tr><td class='label'>Full raw prompt</td><td class='value'>{full_prompt_details}</td></tr>\n"

    item += "</table>"

    js_txt = urllib.parse.quote(json.dumps({k: v for _, k, v, in metadata}, indent=0), safe='')
    item += f"</br><button onclick=\"to_clipboard('{js_txt}')\" disabled>Copy to Clipboard</button>"

    item += "</td>"
    item += "</tr></table></div>\n\n"

    middle_part = item + middle_part

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(begin_part + middle_part + end_part)

    logger.info(f'Image generated with private log at: {html_name}')

    log_cache[html_name] = middle_part
    
    log_ext(local_temp_filename)

    return local_temp_filename


def log_ext(file_name):
    dirname, filename = os.path.split(file_name)
    log_name = os.path.join(dirname, "log_ads.json")
    
    log_ext = {}
    if os.path.exists(log_name):
        with open(log_name, "r", encoding="utf-8") as log_file:
            log_ext.update(json.load(log_file))
    
    ads_ext = {} #ads.get_diff_for_log_ext()
    if len(ads_ext.keys())==0:
        return

    log_ext.update({filename: ads_ext})

    with open(log_name, 'w', encoding='utf-8') as log_file:
        json.dump(log_ext, log_file)

    logger.info(f'Image generated with advanced params log at: {log_name}')
    return 
