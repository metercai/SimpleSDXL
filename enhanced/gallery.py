import gradio as gr
import os
import math
import modules.util as util

from modules.config import path_outputs


output_list = []
max_per_page = 28

def get_listdir_output(num_max):
    global max_per_page

    listdirs = [f for f in os.listdir(path_outputs) if os.path.isdir(os.path.join(path_outputs,f))]
    listdirs1 = listdirs.copy()
    for index in listdirs:
        path_gallery = os.path.join(path_outputs, index)
        nums = len(util.get_files_from_folder(path_gallery, ['.png'], None))
        if nums > max_per_page:
            for i in range(1,math.ceil(nums/max_per_page)+1):
                listdirs1.append(index + "/" + str(i))
            listdirs1.remove(index)
    listdirs = sorted([f[2:] for f in listdirs1], reverse=True)
    if num_max is None:
        return listdirs
    elif num_max < len(listdirs) and num_max > 0:
        return listdirs[:num_max-1]
    return listdirs


def get_images_from_gallery_index(choice):
    global output_list, max_per_page

    if choice is None:
        output_list = get_listdir_output(None)
        choice = output_list[0]
    choice = "20" + choice
    page = 0
    _page = choice.split("/")
    if len(_page) > 1:
        choice = _page[0]
        page = int(_page[1])
    path_gallery = os.path.join(path_outputs, choice)
    images_gallery0 = sorted([f for f in util.get_files_from_folder(path_gallery, ['.png'], None)], reverse=True)
    nums = len(images_gallery0)
    if page > 0:
        if page*max_per_page < nums:
            images_gallery0 = images_gallery0[(page-1)*max_per_page:page*max_per_page-1]
        else:
            images_gallery0 = images_gallery0[nums-max_per_page:]
    images_gallery = [os.path.join(path_gallery,f) for f in images_gallery0]
    print(f'path_gallery={path_gallery}, nums={nums}, page={page}')
    return images_gallery

def change_gallery_index(choice):
    global output_list
    return gr.update(visible=True, preview=True, value=get_images_from_gallery_index(choice))

def hidden_gallery_index():
    return gr.update(visible=False, open=False)

output_list = get_listdir_output(None)
print(f'output_list={output_list}')
