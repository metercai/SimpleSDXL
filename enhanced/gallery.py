import gradio as gr
import os
import math
import json
import copy
import modules.util as util
import modules.config as config
import enhanced.toolbox as toolbox
from lxml import etree

output_list = []
max_per_page = 28

images_list=['',[]]
images_prompt=['',[]]


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


def refresh_images_list(choice):
    global output_list, images_list

    if choice == images_list[0]:
        if images_list[0] == output_list[0].split('/')[0]:
            images_list[1] = sorted([f for f in util.get_files_from_folder(os.path.join(config.path_outputs, '20' + images_list[0]), ['.png'], None)], reverse=True) 
        return
    images_list[0] = choice
    images_list[1] = sorted([f for f in util.get_files_from_folder(os.path.join(config.path_outputs, '20' + images_list[0]), ['.png'], None)], reverse=True)
    parse_html_log(choice)
    print(f'[Gallery] Refresh_images_list: loaded {len(images_list[1])} image_items of {images_list[0]}.')
    return


def get_images_from_gallery_index(choice):
    global output_list, images_list, max_per_page

    if choice is None:
#        refresh_output_list()
        if len(output_list) == 0:
            return None
        choice = output_list[0]
    page = 0
    _page = choice.split("/")
    if len(_page) > 1:
        choice = _page[0]
        page = int(_page[1])

    refresh_images_list(choice)
    images_gallery = images_list[1]
    nums = len(images_gallery)
    if page > 0:
        page = abs(page-math.ceil(nums/max_per_page))+1
        if page*max_per_page < nums:
            images_gallery = images_list[1][(page-1)*max_per_page:page*max_per_page-1]
        else:
            images_gallery = images_list[1][nums-max_per_page:]
    images_gallery = [os.path.join(os.path.join(config.path_outputs, '20' + choice), f) for f in images_gallery]
    #print(f'[Gallery]Get images from index: choice={choice}, page={page}, images_gallery={images_gallery}')
    return images_gallery


refresh_output_list()


def get_images_prompt(choice, selected):
    global images_list, images_prompt

    if choice is None:
        choice = output_list[0]
    page = 0
    _page = choice.split("/")
    if len(_page) > 1:
        choice = _page[0]
        page = int(_page[1])

    if choice != images_prompt[0] or images_prompt[1] is None:
        parse_html_log(choice)
    nums = len(images_prompt[1])
    if page > 0:
        page = abs(page-math.ceil(nums/max_per_page))+1
        if page*max_per_page < nums:
            selected = (page-1)*max_per_page + selected
        else:
            selected = nums-max_per_page + selected
    return images_prompt[1][selected]


def parse_html_log(choice):
    global images_prompt
    
    choice = choice.split('/')[0]
    html_file = os.path.join(os.path.join(config.path_outputs, '20' + choice), 'log.html')
    html = etree.parse(html_file, etree.HTMLParser(encoding='utf-8'))
    prompt_infos = html.xpath('/html/body/div')
    images_prompt[1] = []
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
            if text[2]=='\n':
                text.insert(2, '')
            if text[5]=='\n':
                text.insert(5, '')
            if text[8]=='\n':
                text.insert(8, '')
            info_dict={"Filename":text[0]}
            for i in range(0,int(len(text)/3)):
                    info_dict[text[1+i*3]] = text[2+i*3]
        #print(f'{len(text)},info_dict={info_dict}')
        images_prompt[1].append(info_dict)
    images_prompt[0] = choice
    print(f'[Gallery] Parse_html_log: loaded {len(images_prompt[1])} image_infos of {choice}.')
    

def select_gallery(choice, evt: gr.SelectData):

    result = get_images_prompt(choice, evt.index)
    print(f'[Gallery] Selected_gallery: selected index {evt.index} of {choice} images_list.')

    return result, gr.update(value=toolbox.make_infobox_markdown(result))


def select_gallery_progress(evt: gr.SelectData):
    global output_list

    result = get_images_prompt(None, evt.index)
    print(f'[Gallery] Selected_gallery_progress: selected index {evt.index} of {output_list[0]} images_list.')

    return result, gr.update(value=toolbox.make_infobox_markdown(result))
