import gradio as gr
import os
import math
import json
import copy
import modules.util as util
import modules.config as config
import enhanced.toolbox as toolbox
import re
import shared
import shutil
from lxml import etree


# app context
images_list = {}
images_list_keys = {} #[]
images_prompt = {}
images_prompt_keys = {} #[]
images_ads = {}


image_types = ['.png', '.jpg', '.jpeg', '.webp']
output_images_regex = re.compile(r'\d{4}-\d{2}-\d{2}')

def refresh_output_list(max_per_page, max_catalog, user_did=None):
    global image_types, images_list, images_list_keys, images_prompt, images_prompt_keys, images_ads

    path_outputs = config.path_outputs
    guest_did = shared.token.get_guest_did()
    guest_path_outputs = os.path.join(path_outputs, guest_did)
    if not os.path.exists(guest_path_outputs):
        os.rename(path_outputs, guest_did)
        os.makedirs(path_outputs, exist_ok=True)
        shutil.move(guest_did, os.path.join(path_outputs, guest_did))
    if not user_did:
        user_did = guest_did
    user_path_outputs = os.path.join(path_outputs, user_did)
    if not os.path.exists(user_path_outputs):
        print(f'[Gallery] Makedirs for new user: {user_path_outputs}')
        os.makedirs(user_path_outputs, exist_ok=True)
    listdirs = [f for f in os.listdir(user_path_outputs) if output_images_regex.findall(f) and os.path.isdir(os.path.join(user_path_outputs,f))]
    if listdirs is None:
        return None
    listdirs1 = listdirs.copy()
    total_nums = 0
    for index in listdirs:
        path_gallery = os.path.join(user_path_outputs, index)
        nums = len(util.get_files_from_folder(path_gallery, image_types, None))
        total_nums += nums
        if nums > max_per_page:
            max_page_no = math.ceil(nums/max_per_page)
            for i in range(1,max_page_no+1):
                listdirs1.append("{}/{}".format(index, str(i).zfill(len(str(max_page_no)))))
            listdirs1.remove(index)
    output_list = sorted([f[2:] for f in listdirs1], reverse=True)
    pages = len(output_list)
    display_max_pages = max_catalog
    print(f'[Gallery] Refresh_output_catalog: A total of {total_nums} images and {pages} pages, displaying the latest {pages if pages<display_max_pages else display_max_pages} pages.')
    output_list = output_list[:display_max_pages]
    if user_did not in images_list:
        images_list[user_did]={}
    if user_did not in images_list_keys:
        images_list_keys[user_did]=[]
    if user_did not in images_prompt:
        images_prompt[user_did]={}
    if user_did not in images_prompt_keys:
        images_prompt_keys[user_did]=[]
    if user_did not in images_ads:
        images_ads[user_did]={}

    return output_list, total_nums, pages


def images_list_update(choice, state_params):
    if "__output_list" not in state_params.keys():
        return  gr.update(), gr.update(), state_params
    output_list = state_params["__output_list"]
    if choice is None and len(output_list) > 0:
        choice = output_list[0]
    user_did = state_params["user_did"]
    images_gallery = get_images_from_gallery_index(choice, state_params["__max_per_page"], user_did)
    state_params.update({"prompt_info": [choice, 0]})
    return gr.update(value=images_gallery), gr.update(open=False, visible=len(output_list)>0)


def select_index(choice, image_tools_checkbox, state_params, evt: gr.SelectData):
    if "__output_list" in state_params.keys():
        state_params.update({"infobox_state": 0})
        state_params.update({"note_box_state": ['',0,0]})
    print(f'[Gallery] Selected_gallery_catalog: change image catalog:{choice}.')
    state_params.update({"gallery_state": 'finished_index'})
    state_params['identity_dialog'] = False
    return [gr.update(visible=True)] + [gr.update(visible=image_tools_checkbox)] + [gr.update(visible=False)] * 9


def select_gallery(choice, state_params, backfill_prompt, evt: gr.SelectData):
    if "__output_list" not in state_params.keys():
        return  [gr.update()] * 7 + [state_params]
    state_params.update({"note_box_state": ['',0,0]})
    state_params.update({"prompt_info": [choice, evt.index]})
    if choice is None and len(state_params["__output_list"]) > 0:
        choice = state_params["__output_list"][0]
    result = get_images_prompt(choice, evt.index, state_params["__max_per_page"], True, state_params["user_did"])
    #print(f'[Gallery] Selected_gallery: selected index {evt.index} of {choice} images_list:{result["Filename"]}.')
    if backfill_prompt and 'Prompt' in result:
        return [gr.update(value=toolbox.make_infobox_markdown(result, state_params['__theme'])), gr.update(value=result["Prompt"]), gr.update(value=result["Negative Prompt"])] + [gr.update(visible=False)] * 4 + [state_params]
    else:
        return [gr.update(value=toolbox.make_infobox_markdown(result, state_params['__theme'])), gr.update(), gr.update()] + [gr.update(visible=False)] * 4 + [state_params]

def select_gallery_progress(state_params, evt: gr.SelectData):
    #if "__output_list" not in state_params.keys():
    #    return  [gr.update()] * 5 + [state_params]
    state_params.update({"note_box_state": ['',0,0]})
    state_params.update({"prompt_info": [None, evt.index]})
    result = get_images_prompt(state_params["__output_list"][0], evt.index, state_params["__max_per_page"], user_did=state_params["user_did"])
    return [gr.update(value=toolbox.make_infobox_markdown(result, state_params['__theme']), visible=False)] + [gr.update(visible=False)] * 4


def get_images_from_gallery_index(choice, max_per_page, user_did=None):
    global images_list

    if choice is None:
        return []
    if not user_did:
        user_did = shared.token.get_guest_did()
    if user_did not in images_list:
        images_list[user_did]={}

    page = 0
    _page = choice.split("/")
    if len(_page) > 1:
        choice = _page[0]
        page = int(_page[1])

    images_gallery = refresh_images_catalog(choice, user_did=user_did)
    nums = len(images_gallery)
    if page > 0:
        page = abs(page-math.ceil(nums/max_per_page))+1
        if page*max_per_page < nums:
            images_gallery = images_list[user_did][choice][(page-1)*max_per_page:page*max_per_page]
        else:
            images_gallery = images_list[user_did][choice][nums-max_per_page:]
    user_path_outputs = os.path.join(config.path_outputs, user_did)
    images_gallery = [os.path.join(os.path.join(user_path_outputs, "20{}".format(choice)), f) for f in images_gallery]
    #print(f'[Gallery]Get images from index: choice={choice}, page={page}, images_gallery={images_gallery}')
    return images_gallery


def refresh_images_catalog(choice: str, passthrough = False, user_did=None):
    global images_list, images_list_keys, image_types

    if not user_did:
        user_did = shared.token.get_guest_did()
    if user_did not in images_list:
        images_list[user_did]={}
    if user_did not in images_list_keys:
        images_list_keys[user_did]=[]

    if not passthrough and choice in images_list_keys[user_did]:
        images_list_keys[user_did].remove(choice)
        images_list_keys[user_did].append(choice)
        #print(f'[Gallery] Refresh_images_list: hit cache {len(images_list[user_did][choice])} image_items of {choice}.')
        return images_list[user_did][choice]
    user_path_outputs = os.path.join(config.path_outputs, user_did)
    images_list_new = sorted([f for f in util.get_files_from_folder(os.path.join(user_path_outputs, "20{}".format(choice)), image_types, None)], reverse=True)
    if len(images_list_new)==0:
        parse_html_log(choice, passthrough, user_did)
        if choice in images_list_keys[user_did]:
            images_list_keys[user_did].pop(images_list_keys[user_did].index(choice))
            images_list[user_did].pop(choice)
        return []
    if choice in images_list_keys[user_did]:
        images_list_keys[user_did].pop(images_list_keys[user_did].index(choice))
    if len(images_list[user_did].keys())>15:
        images_list[user_did].pop(images_list_keys[user_did].pop(0))
    images_list[user_did].update({choice: images_list_new})
    images_list_keys[user_did].append(choice)
    parse_html_log(choice, passthrough, user_did)
    print(f'[Gallery] Refresh_images_catalog: loaded {len(images_list[user_did][choice])} image_items of {choice}.')
    return images_list[user_did][choice]


def get_images_prompt(choice, selected, max_per_page, display_index=False, user_did=None):
    global images_list, images_prompt, images_prompt_keys, images_ads

    if choice is None:
        return None
    if not user_did:
        user_did = shared.token.get_guest_did()
    if user_did not in images_list:
        images_list[user_did]={}
    if user_did not in images_prompt:
        images_prompt[user_did]={}
    if user_did not in images_prompt_keys:
        images_prompt_keys[user_did]=[]
    if user_did not in images_ads:
        images_ads[user_did]={}
    
    page = 0
    _page = choice.split("/")
    if len(_page) > 1:
        choice = _page[0]
        page = int(_page[1])
    page_choice = page
    page_index = selected
    parse_html_log(choice, user_did=user_did)
    if choice not in images_list[user_did].keys():
        nums = len(refresh_images_catalog(choice, user_did=user_did))
    else:
        nums = len(images_list[user_did][choice])
    if page > 0:
        page = abs(page-math.ceil(nums/max_per_page))+1
        if page*max_per_page < nums:
            selected = (page-1)*max_per_page + selected
        else:
            selected = nums-max_per_page + selected
    images_prompt_keys[user_did].remove(choice)
    images_prompt_keys[user_did].append(choice)
    filename = images_list[user_did][choice][selected]
    metainfo = {"Filename": filename} if filename not in images_prompt[user_did][choice].keys() else images_prompt[user_did][choice][filename]
    if display_index:
        print(f'[Gallery] The image selected: catalog={choice}, page={page_choice}, in_page={page_index}, in_catalog={selected}, filename={filename}')
    if choice in images_ads[user_did].keys() and filename in images_ads[user_did][choice].keys():
        metainfo.update({"Advanced_parameters": images_ads[user_did][choice][metainfo['Filename']]})
    return metainfo


def parse_html_log(choice: str, passthrough = False, user_did=None):
    global images_prompt, images_prompt_keys, images_ads
    
    if not user_did:
        user_did = shared.token.get_guest_did()
    if user_did not in images_prompt:
        images_prompt[user_did]={}
    if user_did not in images_prompt_keys:
        images_prompt_keys[user_did]=[]
    if user_did not in images_ads:
        images_ads[user_did]={}

    choice = choice.split('/')[0]
    if not passthrough and choice in images_prompt_keys[user_did] and images_prompt[user_did][choice]:
        images_prompt_keys[user_did].remove(choice)
        images_prompt_keys[user_did].append(choice)
        #print(f'[Gallery] Parse_html_log: hit cache {len(images_prompt[user_did][choice])} image_infos of {choice}.')
        return
    user_path_outputs = os.path.join(config.path_outputs, user_did)
    html_file = os.path.join(os.path.join(user_path_outputs, "20{}".format(choice)), 'log.html')
    if not os.path.exists(html_file):
        return
    html = etree.parse(html_file, etree.HTMLParser(encoding='utf-8'))
    prompt_infos = html.xpath('/html/body/div')
    images_prompt_list = {}
    images_prompt_list[user_did] = {}
    for info in prompt_infos:
        text = info.xpath('.//p//text()')
        #print(f'log_parse_text1:{text}')
        if len(text)>20:
            def standardized(x):
                if x.startswith(', '):
                    x=x[2:]
                if x.endswith(': '):
                    x=x[:-2]
                if x==' ':
                    x=''
                return x
            text = list(map(standardized, info.xpath('.//p//text()')))
            if text[6]!='':
                text.insert(6, '')
            if text[8]=='':
                text.insert(8, '')
            info_dict={"Filename":text[0]}
            if text[3]=='':
                info_dict[text[1]] = text[2]
                info_dict[text[4]] = text[5]
                info_dict[text[7]] = text[8]
                for i in range(0,int(len(text)/2)-5):
                    info_dict[text[10+i*2]] = text[11+i*2]
            else:
                if text[4]!='Fooocus V2 Expansion':
                    del text[6]
                else:
                    text.insert(4, '')
                    if text[6]=='Styles':
                        text.insert(6, '')
                        del text[8]
                    else:
                        del text[7]
                for i in range(0,int(len(text)/2)-1):
                    info_dict[text[1+i*2]] = text[2+i*2]
        else:
            text = info.xpath('.//td//text()')
            #print(f'log_parse_text2:{text}')
            if len(text)>10:
                if text[2]=='\n' or text[2]=='\r\n':
                    text.insert(2, '')
                if text[5]=='\n' or text[5]=='\r\n':
                    text.insert(5, '')
                if text[8]=='\n' or text[8]=='\r\n':
                    text.insert(8, '')
                if text[29]=='\n' or text[29]=='\r\n':
                    text.insert(29, '')
                if text[32]=='\n' or text[32]=='\r\n':
                    text.insert(32, '')
                if text[35]=='\n' or text[35]=='\r\n':
                    text.insert(35, '')
                if text[41]=='\n' or text[41]=='\r\n':
                    text.insert(41, '')
                info_dict={"Filename":text[0]}
                for i in range(0,int(len(text)/3)):
                    key = text[1+i*3].strip()
                    value = text[2+i*3].strip()
                    if key == '' or key is None or key == 'Full raw prompt' or key == 'Positive' or key == 'Negative':
                        continue
                    info_dict[key] = value
            else:
                if 'Upscale (Fast)' not in text:
                    print(f'[Gallery] Parse_html_log: Parse error for {choice}, file={html_file}\ntext:{info.xpath(".//text()")}')
                info_dict={"Filename":text[1]}
                info_dict[text[2]] = text[3]
        #print(f'{len(text)},info_dict={info_dict}')
        images_prompt_list[user_did].update({info_dict["Filename"]: info_dict})
    if len(images_prompt_list[user_did].keys())==0:
        if choice in images_prompt[user_did].keys():
            images_prompt_keys[user_did].pop(images_prompt_keys[user_did].index(choice))
            images_prompt[user_did].pop(choice)
            if choice in images_ads[user_did].keys():
                images_ads[user_did].pop(choice)
        return
    if choice in images_prompt_keys[user_did]:
        images_prompt_keys[user_did].pop(images_prompt_keys[user_did].index(choice))
    if len(images_prompt[user_did].keys())>15:
        key = images_prompt_keys[user_did].pop(0)
        images_prompt[user_did].pop(key)
        if key in images_ads[user_did].keys():
            images_ads[user_did].pop(key)
    images_prompt[user_did].update({choice: images_prompt_list[user_did]})
    images_prompt_keys[user_did].append(choice)
    
    dirname, filename = os.path.split(html_file)
    log_name = os.path.join(dirname, "log_ads.json")
    log_ext = {}
    if os.path.exists(log_name):
        with open(log_name, "r", encoding="utf-8") as log_file:
            log_ext.update(json.load(log_file))
    images_ads[user_did].update({choice: log_ext})
    
    print(f'[Gallery] Parse_html_log: loaded {len(images_prompt[user_did][choice])} image_infos of {choice}.')
    return


