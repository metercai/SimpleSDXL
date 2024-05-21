"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 Searge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import shutil
import sys

from inspect import currentframe, getframeinfo
from pathlib import Path
from urllib import request
from urllib.error import ContentTooShortError

SEARGE_SDXL_GIT_URL = "https://github.com/SeargeDP/SeargeSDXL"

COMFY_UI_FOLDER = "ComfyUI"
EMBEDDED_PYTHON_DIR_NAME = "python_embeded"
MODELS_DIR_NAME = os.path.join(COMFY_UI_FOLDER, "models")
EXTENSION_DIR_NAME = os.path.join(COMFY_UI_FOLDER, "custom_nodes")

current_directory = Path(getframeinfo(currentframe()).filename).resolve().parent
embedded_python_directory = current_directory.joinpath(EMBEDDED_PYTHON_DIR_NAME)

alternative_directory = current_directory.parent.parent.parent.parent
embedded_python_directory2 = alternative_directory.joinpath(EMBEDDED_PYTHON_DIR_NAME)

running_in_comfy_portable = Path.exists(embedded_python_directory)
running_in_extension_directory = Path.exists(embedded_python_directory2)

installer_path = str(current_directory)
comfy_path = installer_path if running_in_comfy_portable else str(alternative_directory)

os.chdir(comfy_path)

comfyui_folder = Path(comfy_path).joinpath(COMFY_UI_FOLDER)
found_comfyui_folder = Path.exists(comfyui_folder)

models_folder = Path(comfy_path).joinpath(MODELS_DIR_NAME)
found_models_folder = Path.exists(models_folder)

extensions_folder = Path(comfy_path).joinpath(EXTENSION_DIR_NAME)
found_extensions_folder = Path.exists(extensions_folder)

MODELS = [
    {
        "filename": "sd_xl_base_1.0_0.9vae.safetensors",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/",
        "folder": "checkpoints",
        "importance": "required",
    },
    {
        "filename": "sd_xl_refiner_1.0_0.9vae.safetensors",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/",
        "folder": "checkpoints",
        "importance": "recommended",
    },
    {
        "filename": "sdxl_vae.safetensors",
        "url": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/",
        "folder": "vae",
        "importance": "optional",
    },
    {
        "filename": "sd_xl_offset_example-lora_1.0.safetensors",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/",
        "folder": "loras",
        "importance": "optional",
    },
    {
        "filename": "4x-UltraSharp.pth",
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/",
        "folder": "upscale_models",
        "importance": "recommended",
    },
    {
        "filename": "4x_NMKD-Siax_200k.pth",
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/",
        "folder": "upscale_models",
        "importance": "recommended",
    },
    {
        "filename": "4x_Nickelback_70000G.pth",
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/",
        "folder": "upscale_models",
        "importance": "recommended",
    },
    {
        "filename": "1x-ITF-SkinDiffDetail-Lite-v1.pth",
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/",
        "folder": "upscale_models",
        "importance": "optional",
    },
    {
        "filename": "ControlNetHED.pth",
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/",
        "folder": "annotators",
        "importance": "required",
    },
    {
        "filename": "res101.pth",
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/",
        "folder": "annotators",
        "importance": "required",
    },
    {
        "filename": "clip_vision_g.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/revision/",
        "folder": "clip_vision",
        "importance": "recommended",
    },
    {
        "filename": "control-lora-canny-rank256.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/",
        "folder": "controlnet",
        "importance": "recommended",
    },
    {
        "filename": "control-lora-depth-rank256.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/",
        "folder": "controlnet",
        "importance": "recommended",
    },
    {
        "filename": "control-lora-recolor-rank256.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/",
        "folder": "controlnet",
        "importance": "recommended",
    },
    {
        "filename": "control-lora-sketch-rank256.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/",
        "folder": "controlnet",
        "importance": "recommended",
    },
]


# -----==========-----

def get_input(prompt, default=None, options=None):
    valid = False

    info = ""
    if options is not None:
        info += f" [{'|'.join(options)}]"

    if default is not None:
        info += f" (default = [{default}])"

    inp = ""
    while not valid:
        inp = input(f"{prompt}\n{info} ")

        if default is not None and inp == "":
            return default

        if options is None:
            valid = True

        if inp in options:
            valid = True

    return inp


# -----==========-----

def check_test_version():
    print("Checking if old test version SeargeSDXL-Test is installed")

    test_path = extensions_folder.joinpath("SeargeSDXL-Test")
    test_path_exists = Path.exists(test_path)

    if not test_path_exists:
        return True

    print(f"Old test version of SeargeSDXL found in {test_path}")
    inp = get_input("Should the test version be deleted?", "n", ["y", "n"])

    if inp == "y":
        shutil.rmtree(test_path, ignore_errors=True)
    else:
        print("Please delete the old test version manually to avoid potential conflicts")

    return True


# -----==========-----

def check_non_git_version():
    print("Checking if SeargeSDXL is already installed")

    ext_path = extensions_folder.joinpath("SeargeSDXL")
    ext_git_path = ext_path.joinpath(".git")
    if not Path.exists(ext_path) or Path.exists(ext_git_path):
        return True

    print(f"Found a version of SeargeSDXL in {ext_path}")
    print(f"But it has not been installed via git, missing folder: {ext_git_path}")

    inp = get_input("Should the old version be deleted and replaced?", "y", ["y", "n"])

    if inp == "y":
        shutil.rmtree(ext_path, ignore_errors=True)
    else:
        print("Please delete the old version manually and restart the install script afterwards")
        return False

    return True


# -----==========-----

def check_git():
    print("Checking git version")
    ec = os.system("git --version")
    if ec != 0:
        print("Please make sure git is installed on your system")
        return False

    return True


# -----==========-----

def install_searge_sdxl():
    print("Installing or updating SeargeSDXL")

    ext_path = extensions_folder.joinpath("SeargeSDXL")
    ext_path_exists = Path.exists(ext_path)
    ext_git_path = ext_path.joinpath(".git")

    if ext_path_exists and not Path.exists(ext_git_path):
        print("The SeargeSDXL extension exists, but is incorrectly installed. Please remove the "
              f"directory {ext_path} and restart the install script")
        return False

    old_cwd = os.getcwd()
    if ext_path_exists:
        os.chdir(ext_path)

        print("Fetching latest commits from the git repository")
        ec = os.system("git fetch --all")
        if ec != 0:
            print("The command 'git fetch' failed, please restart the install script to try again")
            return False

        os.chdir(old_cwd)

    else:
        os.chdir(extensions_folder)

        print("Cloning SeargeSDXL from the git repository")
        ec = os.system(f"git clone {SEARGE_SDXL_GIT_URL}")
        if ec != 0:
            print(f"The command 'git clone' failed, please delete the directory {ext_path} and "
                  "restart the install script to try again")
            return False

        os.chdir(old_cwd)

    print("\nUsually the release branch has the latest stable version and the test branch has the latest "
          "test version.")

    inp = get_input("Do you want to switch the latest [r]elease branch, latest [t]est branch, or stay on the "
                    "[c]urrently installed version?", "r", ["r", "t", "c"])

    if inp == "t":
        os.chdir(ext_path)

        ec = os.system("git stash")
        if ec == 0:
            ec = os.system("git switch -C dev origin/dev")

        os.chdir(old_cwd)

    elif inp == "r":
        os.chdir(ext_path)

        ec = os.system("git stash")
        if ec == 0:
            ec = os.system("git switch -C main origin/main")

        os.chdir(old_cwd)

    else:
        ec = 0

    if ec != 0:
        print("Could not switch to the selected branch, please restart the install script to try again")
        return False

    return True


# -----==========-----

def install_models():
    print("Installing required models and checkpoints")

    done = False
    while not done:
        required, recommended, optional, ids = im_show_model_status()

        if len(ids) == 0:
            print("\nAll models and checkpoints are installed, nothing to do here")
            return True

        default = "d"

        if len(required) > 0:
            info1 = "re[q]uired, "
            ids += ["q"]
            default = "q"
        else:
            info1 = ""

        if len(recommended) > 0:
            if default == "q":
                info2 = "[r]ecommended+required, "
            else:
                info2 = "[r]ecommended, "
            ids += ["r"]
            default = "r"
        else:
            info2 = ""

        print("")
        inp = get_input(f"Select a number for a model or: {info1}{info2}[a]ll, [d]one",
                        default, ids + ["a", "d"])
        models = None
        model = None
        if inp == "d":
            done = True
        elif inp == "q":
            models = required
        elif inp == "r":
            models = required + recommended
        elif inp == "a":
            models = required + recommended + optional
        elif inp.isnumeric() and inp in ids:
            model = MODELS[int(inp)]
        else:
            print("\nUnhandled selection from model list")

        print("")
        if models is not None:
            for model in models:
                if not install_model(model):
                    return False
        elif model is not None:
            if not install_model(model):
                return False

    return True


# -----==========-----

def im_show_model_status():
    required = []
    recommended = []
    optional = []
    ids = []

    for i in range(len(MODELS)):
        model = MODELS[i]

        folder = model["folder"]
        filename = model["filename"]
        importance = model["importance"]

        full_path = models_folder.joinpath(folder).joinpath(filename)
        if Path.exists(full_path):
            print(f"[  ] - already installed: {folder}/{filename}")
        else:
            ids.append(str(i))
            if importance == "required":
                print(f"[{i:2}] - *** required *** : {folder}/{filename}")
                required.append(model)

            elif importance == "recommended":
                print(f"[{i:2}] -  + recommended + : {folder}/{filename}")
                recommended.append(model)

            elif importance == "optional":
                print(f"[{i:2}] -   ( optional )  : {folder}/{filename}")
                optional.append(model)

    return required, recommended, optional, ids


# -----==========-----

def install_model(model):
    folder = model["folder"]
    filename = model["filename"]
    url = model["url"]

    dl_url = f"{url}{filename}"
    full_folder = models_folder.joinpath(folder)
    full_path = full_folder.joinpath(filename)

    if not Path.exists(full_folder):
        os.makedirs(full_folder, exist_ok=True)

    print(f"Downloading {folder}/{filename}\n from {dl_url}")

    counter = {"last": 0}

    def progress(blocks, blksize, total):
        transferred = blocks * blksize

        mb = int(transferred / (1024 * 1024))
        tmb = int(total / (1024 * 1024))

        last = counter["last"]
        if mb != last:
            counter["last"] = mb
            print(f"Transferred {mb} MB / {tmb} MB")

    try:
        request.urlretrieve(dl_url, full_path, progress)

    except ContentTooShortError:
        print("Download incomplete")
        return False

    return True


# -----==========-----

def template():
    print("")

    return True


# -----==========-----

def check_opencv():
    print("Checking if opencv-python is installed")
    try:
        import cv2
        print("The library opencv-python is correctly installed")
    except ImportError:
        print("The library opencv-python is not installed properly, please run the SeargeSDXL-Installer.bat file!")
        return False

    return True


# -----==========-----

def do_install():
    if running_in_extension_directory:
        print(f"Switched to directory: {os.getcwd()}")

    correct_directory = os.getcwd() == comfy_path
    if not correct_directory:
        print("The script is NOT running in the correct directory")
        return False

    if found_comfyui_folder:
        print(f"Found ComfyUI folder: {comfyui_folder}")
    else:
        print(f"Could NOT find ComfyUI folder: {comfyui_folder}")
        return False

    if found_models_folder:
        print(f"Found models folder: {models_folder}")
    else:
        print(f"Could NOT find models folder: {models_folder}")
        return False

    if found_extensions_folder:
        print(f"Found extensions folder: {extensions_folder}")
    else:
        print(f"Could NOT find extensions folder: {extensions_folder}")
        return False

    print("")
    if not check_git():
        return False

    print("")
    if not check_opencv():
        return False

    print("")
    if not check_test_version():
        return False

    print("")
    if not check_non_git_version():
        return False

    print("")
    if not install_searge_sdxl():
        return False

    print("")
    if not install_models():
        return False

    return True


# -----==========-----

if len(sys.argv) < 2 or sys.argv[1] != "from_batch":
    print("Please run this script from the batch file")
    _ = input("Press enter to end this script now")

elif running_in_comfy_portable or running_in_extension_directory:
    print(f"\nInstaller python script running in {installer_path}")
    print(f"This is the {'recommended' if running_in_comfy_portable else 'unsupported'} directory to install from\n")

    _ = input("\nPlease make sure ComfyUI is not running, then press enter\n")

    if do_install():
        print("\nInstaller script finished\n")

        if running_in_extension_directory:
            print("If any updates were received during 'git fetch', it's recommended to run this script once more.\n")
    else:
        print("\nInstaller FAILED to finish properly\n")

else:
    print(f"Not running in the correct directory or {EMBEDDED_PYTHON_DIR_NAME} could not be found, "
          "we are in {os.getcwd()}\n")
