import gradio as gr
import os
import math
import json
import copy
import modules.util as util
import modules.config as config
import enhanced.toolbox as toolbox
from lxml import etree


# app context
output_list = []
max_per_page = 28

images_list = {}
images_list_keys = []
images_prompt = {}
images_prompt_keys = []

def refresh_output_list():
    global output_list, max_per_page

    listdirs = [f for f in os.listdir(config.path_outputs) if os.path.isdir(os.path.join(config.path_outputs,f))]
    if listdirs is None:
        return
    listdirs1 = listdirs.copy()
    for index in listdirs:
        path_gallery = os.path.join(config.path_outputs, index)
        nums = len(util.get_files_from_folder(path_gallery, ['.png'], None))
        if nums > max_per_page:
            for i in range(1,math.ceil(nums/max_per_page)+1):
                listdirs1.append(index + "/" + str(i))
            listdirs1.remove(index)
    output_list = sorted([f[2:] for f in listdirs1], reverse=True)
    print(f'[Gallery] Refresh_output_list: loaded {len(output_list)} images_lists.')
    return


def images_list_update(choice, state_params):
    global output_list
    
    images_gallery = get_images_from_gallery_index(choice)
    state_params.update({"prompt_info": [choice, 0]})
    return gr.update(value=images_gallery), gr.update(open=False, visible=len(output_list)>0), state_params


def select_index(choice, state_params, evt: gr.SelectData):
    state_params.update({"infobox_state": 0})
    state_params.update({"note_box_state": ['',0,0]})
    print(f'[Gallery] Selected_gallery_index: change image catalog:{choice}.')
    return [gr.update(visible=True)] + [gr.update(visible=False)] * 3 + [state_params]


def select_gallery(choice, state_params, backfill_prompt, evt: gr.SelectData):
    state_params.update({"note_box_state": ['',0,0]})
    state_params.update({"prompt_info": [choice, evt.index]})
    result = get_images_prompt(choice, evt.index)
    #print(f'[Gallery] Selected_gallery: selected index {evt.index} of {choice} images_list:{result["Filename"]}.')
    if backfill_prompt:
        return [gr.update(value=toolbox.make_infobox_markdown(result)), gr.update(value=result["Prompt"]), gr.update(value=result["Negative Prompt"])] + [gr.update(visible=False)] * 4 + [state_params]
    else:
        return [gr.update(value=toolbox.make_infobox_markdown(result)), gr.update(), gr.update()] + [gr.update(visible=False)] * 4 + [state_params]

def select_gallery_progress(state_params, evt: gr.SelectData):
    global output_list
    state_params.update({"note_box_state": ['',0,0]})
    state_params.update({"prompt_info": [None, evt.index]})
    result = get_images_prompt(None, evt.index)
    #print(f'[Gallery] Selected_gallery_progress: selected index {evt.index} of {output_list[0]} images_list.')
    return [gr.update(value=toolbox.make_infobox_markdown(result), visible=False)] + [gr.update(visible=False)] * 4 + [state_params]


def get_images_from_gallery_index(choice):
    global output_list, max_per_page, images_list

    if choice is None:
        if len(output_list) == 0:
            return None
        choice = output_list[0]
    page = 0
    _page = choice.split("/")
    if len(_page) > 1:
        choice = _page[0]
        page = int(_page[1])

    images_gallery = refresh_images_list(choice)
    nums = len(images_gallery)
    if page > 0:
        page = abs(page-math.ceil(nums/max_per_page))+1
        if page*max_per_page < nums:
            images_gallery = images_list[choice][(page-1)*max_per_page:page*max_per_page-1]
        else:
            images_gallery = images_list[choice][nums-max_per_page:]
    images_gallery = [os.path.join(os.path.join(config.path_outputs, '20' + choice), f) for f in images_gallery]
    #print(f'[Gallery]Get images from index: choice={choice}, page={page}, images_gallery={images_gallery}')
    return images_gallery


def refresh_images_list(choice: str, passthrough = False):
    global output_list, images_list, images_list_keys

    if not passthrough and choice in images_list_keys:
        images_list_keys.remove(choice)
        images_list_keys.append(choice)
        #print(f'[Gallery] Refresh_images_list: hit cache {len(images_list[choice])} image_items of {choice}.')
        return images_list[choice]

    images_list_new = sorted([f for f in util.get_files_from_folder(os.path.join(config.path_outputs, '20' + choice), ['.png'], None)], reverse=True)
    if len(images_list_new)==0:
        parse_html_log(choice, passthrough)
        if choice in images_list_keys:
            images_list_keys.pop(images_list_keys.index(choice))
            images_list.pop(choice)
        return []
    if choice in images_list_keys:
        images_list_keys.pop(images_list_keys.index(choice))
    if len(images_list.keys())>15:
        images_list.pop(images_list_keys.pop(0))
    images_list.update({choice: images_list_new})
    images_list_keys.append(choice)
    parse_html_log(choice, passthrough)
    print(f'[Gallery] Refresh_images_list: loaded {len(images_list[choice])} image_items of {choice}.')
    return images_list[choice]


def get_images_prompt(choice, selected):
    global max_per_page, images_prompt, images_prompt_keys

    if choice is None:
        choice = output_list[0]
    page = 0
    _page = choice.split("/")
    if len(_page) > 1:
        choice = _page[0]
        page = int(_page[1])

    parse_html_log(choice)
    nums = len(images_prompt[choice])
    if page > 0:
        page = abs(page-math.ceil(nums/max_per_page))+1
        if page*max_per_page < nums:
            selected = (page-1)*max_per_page + selected
        else:
            selected = nums-max_per_page + selected
    images_prompt_keys.remove(choice)
    images_prompt_keys.append(choice)
    return images_prompt[choice][selected]


def parse_html_log(choice: str, passthrough = False):
    global images_prompt, images_prompt_keys
    
    choice = choice.split('/')[0]
    if not passthrough and choice in images_prompt_keys and images_prompt[choice]:
        images_prompt_keys.remove(choice)
        images_prompt_keys.append(choice)
        #print(f'[Gallery] Parse_html_log: hit cache {len(images_prompt[choice])} image_infos of {choice}.')
        return
    html_file = os.path.join(os.path.join(config.path_outputs, '20' + choice), 'log.html')
    html = etree.parse(html_file, etree.HTMLParser(encoding='utf-8'))
    prompt_infos = html.xpath('/html/body/div')
    images_prompt_list = []
    for info in prompt_infos:
        text = info.xpath('.//p//text()')
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
            if len(text)>10:
                if text[2]=='\n':
                    text.insert(2, '')
                if text[5]=='\n':
                    text.insert(5, '')
                if text[8]=='\n':
                    text.insert(8, '')
                info_dict={"Filename":text[0]}
                for i in range(0,int(len(text)/3)):
                    info_dict[text[1+i*3]] = text[2+i*3]
            else:
                print(f'[Gallery] Parse_html_log: Parse error for {choice}, file={html_file}\ntext:{info.xpath(".//text()")}')
        #print(f'{len(text)},info_dict={info_dict}')
        images_prompt_list.append(info_dict)
    if len(images_prompt_list)==0:
        if choice in images_prompt.keys():
            images_prompt_keys.pop(images_prompt_keys.index(choice))
            images_prompt.pop(choice)
        return
    if choice in images_prompt_keys:
        images_prompt_keys.pop(images_prompt_keys.index(choice))
    if len(images_prompt.keys())>15:
        images_prompt.pop(images_prompt_keys.pop(0))
    images_prompt.update({choice: images_prompt_list})
    images_prompt_keys.append(choice)
    print(f'[Gallery] Parse_html_log: loaded {len(images_prompt[choice])} image_infos of {choice}.')
    return


refresh_output_list()
