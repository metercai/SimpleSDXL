import os
import json
import gradio as gr
import modules.config as config
import base64
import hashlib
import requests
import time
import enhanced.token_did as token_did
from enhanced.models_hub_host import models_hub_host


models_info = {}
models_info_muid = {}
models_info_rsync = {}
models_info_file = ['models_info', 0]
models_info_path = os.path.abspath(f'./models/{models_info_file[0]}.json')

default_models_info = {
    "checkpoints/juggernautXL_version6Rundiffusion.safetensors": {
        "size": 7105348560,
        "hash": "H+bH7FTHhgQM2rx7TolyAGnZcJaSLiDQHxPndkQStH8=" },
    "checkpoints/sd_xl_base_1.0_0.9vae.safetensors": {
        "size": 6938078334,
        "hash": "5rueqFu/e/ZHinxtGLcSRvIuldQbzdgO1AqiEsM8/v8=" },
    "checkpoints/bluePencilXL_v050.safetensors": {
        "size": 6938040682,
        "hash": "jEYqgPS1KRsk/rYk1nJIvQn3H5B01wFNiteQGxIL3Mc=" },
    "checkpoints/sd_xl_refiner_1.0_0.9vae.safetensors": {
        "size": 6075981930,
        "hash": "jQzmwBYATL2s1Q+Tfa04HYw5ZijWIaf5cZFHBTJ4AWQ=" },
    "checkpoints/realisticStockPhoto_v10.safetensors": {
        "size": 6938053346,
        "hash": "LUTON43cySJ2nfTv/nTa4zcKiH6piXCT4fX73FC2egI=" },
    "checkpoints/DreamShaper_8_pruned.safetensors": {
        "size": 2132625894,
        "hash": "h521I8MNO5AXFD1WcFAV4VostWKHYsEdCG/tlTir1/0=" },
    "checkpoints/juggernautXL_v7Rundiffusion.safetensors": {
        "size": 7105352832,
        "hash": "ByRRjGtahV4XYEMvvwPIBblb0voHJy8PNq33rtlaAeM=" },
    "checkpoints/bluePencilXLV200.cXoH.safetensors": {
        "size": 6938040682,
        "hash": "1ZYnt0lbBYHe0rBIismPneuGtS1D6dkCkU8Pnc/9sWw=" },
    "checkpoints/dreamshaperXL_turboDpmppSDE.safetensors": {
        "size": 6939220250,
        "hash": "Z28NYMjoYBRtXooNgCWZyt0E58rfhcKD8Yn0HwHJ41k=" }
     }
    

def init_models_info():
    global models_info, models_info_file, models_info_path

    if os.path.exists(models_info_path):
        file_mtime = time.localtime(os.path.getmtime(models_info_path)) 
        if (models_info is None or file_mtime != models_info_file[1]):
            try:
                with open(models_info_path, "r", encoding="utf-8") as json_file:
                    models_info.update(json.load(json_file))
                models_info_file[1] = file_mtime
            except Exception as e:
                print(f'[ModelInfo] Load model info file [{models_info_path}] failed!')
                print(e)
    refresh_models_info_from_path()
    return
    
def refresh_models_info_from_path():
    global models_info, models_info_file, models_info_path

    model_filenames = config.get_model_filenames(config.paths_checkpoints)
    lora_filenames = config.get_model_filenames(config.paths_loras)
    embedding_filenames = config.get_model_filenames([config.path_embeddings])
    models_info_muid = {}
    new_filenames = []
    new_models_info = {}
    for k in model_filenames:
        filename = 'checkpoints/'+k
        if filename not in models_info.keys():
            new_filenames.append(filename)
        else:
            new_models_info.update({filename: models_info[filename]})
            if models_info[filename]['muid']:
                models_info_muid.update({models_info[filename]['muid']: filename})
    for k in lora_filenames:
        filename = 'loras/'+k
        if filename not in models_info.keys():
            new_filenames.append(filename)
        else:
            new_models_info.update({filename: models_info[filename]})
            if models_info[filename]['muid']:
                models_info_muid.update({models_info[filename]['muid']: filename})
    for k in embedding_filenames:
        filename = 'embeddings/'+k
        if filename not in models_info.keys():
            new_filenames.append(filename)
        else:
            new_models_info.update({filename: models_info[filename]})
            if models_info[filename]['muid']:
                models_info_muid.update({models_info[filename]['muid']: filename})
    models_info = new_models_info

    if len(new_filenames)>0:
        try:
            for f in new_filenames:
                if f.startswith("checkpoints/"):
                    file_path = os.path.join(config.paths_checkpoints[0], f[12:])
                elif f.startswith("loras/"):
                    file_path = os.path.join(config.paths_loras[0], f[6:])
                elif f.startswith("embeddings/"):
                    file_path = os.path.join(config.path_embeddings, f[11:])
                else:
                    file_path = os.path.abspath(f'./models/{f}')
                size = os.path.getsize(file_path)
                if f in default_models_info.keys() and size == default_models_info[f]["size"]:
                    hash = default_models_info[f]["hash"]
                else:
                    hash = ''
                models_info.update({f:{'size': size, 'hash': hash, 'url': None, 'muid': None}})
            with open(models_info_path, "w", encoding="utf-8") as json_file:
                json.dump(models_info, json_file, indent=4)
            models_info_file[1] = time.localtime(os.path.getmtime(models_info_path))
        except Exception as e:
            print(f'[ModelInfo] Update model info file [{models_info_path}] failed!')
            print(e)
    
    return

                
def get_file_sha256(file_path):
    sha256obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            b = f.read(4*1024*1024)
            if not b:
                break
            sha256obj.update(b)
    return base64.b64encode(sha256obj.digest()).decode('utf-8')
                

def sync_model_info_click(*args):
    global models_info

    downurls = list(args)
    #print(f'downurls:{downurls} \nargs:{args}, len={len(downurls)}')
    keylist = sync_model_info(downurls)
    results = []
    nums = 0
    for k in keylist:
        muid = ' ' if models_info[k]['muid'] is None else models_info[k]['muid']
        durl = None if models_info[k]['url'] is None else models_info[k]['url']
        nums += 1 if models_info[k]['muid'] is None else 0
        results += [gr.update(info=f'MUID={muid}', value=durl)]
    if nums:
        print(f'[ModelInfo] There are {nums} model files missing MUIDs, which need to be added with download URLs before synchronizing.')
    return results


def sync_model_info(downurls):
    global models_info, models_info_rsync, models_info_file, models_info_path

    keys = sorted(models_info.keys())
    # file hash completion
    for f in keys:
        if not models_info[f]['hash']:
            print(f'[ModelInfo] Computing file hash for {f}')
            if f.startswith("checkpoints/"):
                file_path = os.path.join(config.paths_checkpoints[0], f[12:])
            elif f.startswith("loras/"):
                file_path = os.path.join(config.paths_loras[0], f[6:])
            elif f.startswith("embeddings/"):
                file_path = os.path.join(config.path_embeddings, f[11:])
            else:
                file_path = os.path.abspath(f'./models/{f}')
            models_info[f].update({'hash':get_file_sha256(file_path)})
    keylist = []
    for i in range(len(keys)):
        if keys[i].startswith('checkpoints'):
            keylist.append(keys[i])
        if keys[i].startswith('loras'):
            keylist.append(keys[i])
    for i in range(len(keys)):
        if not keys[i].startswith('checkpoints') and not keys[i].startswith('loras'):
            keylist.append(keys[i])

    models_info_rsync = {}
    models_info_update_flag = False
    for i in range(len(keylist)):
        #print(f'downurls: i={i}, k={keylist[i]}, {downurls[i]}')
        durl = '' if i >= len(downurls) else downurls[i]
        if durl and models_info[keylist[i]]['url'] != durl:
            models_info_rsync.update({keylist[i]: {"hash": models_info[keylist[i]]['hash'], "url": durl}})
            models_info[keylist[i]]['url'] = durl
            models_info_update_flag = True

    file_mtime = time.localtime(os.path.getmtime(models_info_path))
    for k in models_info.keys():
        if not models_info[k]['muid'] and k not in models_info_rsync.keys():
            models_info_rsync.update({k: {"hash": models_info[k]['hash'], "url": ""}})
    try:
        response = requests.post(f'{models_hub_host}/register_claim/', data = token_did.get_register_claim('SimpleSDXLHub'))
        rsync_muid_msg = { "files": token_did.encrypt_default(json.dumps(models_info_rsync)) }
        headers = { "DID": token_did.DID}
        response = requests.post(f'{models_hub_host}/rsync_muid/', data = json.dumps(rsync_muid_msg), headers = headers)
        results = json.loads(response.text)
        if (results["message"] == "it's ok!" and results["results"]):
            for k in results["results"].keys():
                models_info[k]['muid'] = results["results"][k]['muid']
                models_info_muid[results["results"][k]['muid']] = k
                models_info[k]['url'] = results["results"][k]['url']
            print(f'[ModelInfo] Rsync {len(results["results"].keys())} MUIDs info from model hub.')
            with open(models_info_path, "w", encoding="utf-8") as json_file:
                json.dump(models_info, json_file, indent=4)
            models_info_file[1] = time.localtime(os.path.getmtime(models_info_path))
    except Exception as e:
            print(f'[ModelInfo] Connect the models hub site failed!')
            print(e)

    file_mtime2 = time.localtime(os.path.getmtime(models_info_path))
    if (models_info_update_flag and file_mtime == file_mtime2):
        with open(models_info_path, "w", encoding="utf-8") as json_file:
            json.dump(models_info, json_file, indent=4)
        models_info_file[1] = time.localtime(os.path.getmtime(models_info_path))
    return keylist

init_models_info()

