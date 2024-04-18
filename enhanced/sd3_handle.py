import os
import time
import json
import random
import requests
import modules.config as config 

def sd3_generate_api(prompt, aspect_ratio='16:9', mode='text-to-image', negative_prompt='', model='sd3', seed=0, output_format='jpeg'):
    path_filename = ''
    path_out_temp = 'temp_out'
    try:
        api_tokens = {}
        api_token_file = os.path.abspath(f'./enhanced/api_token.json')
        if os.path.exists(api_token_file):
            with open(api_token_file, "r", encoding="utf-8") as json_file:
                api_tokens.update(json.load(json_file))
        if 'sai_sd3_api' in api_tokens:
            print(f'[SD3Generator] Generating image with api of SAI-SD3') 
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={
                    "authorization": api_tokens['sai_sd3_api'],
                    "accept": "image/*"
                },
                files={"none": ''},
                data={
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "mode": mode,
                    "model": model,
                    "seed": scale_u64_to_u32(seed),
                    "output_format": output_format,
                    "negative_prompt": negative_prompt,
                },
            )

            if response.status_code == 200:
                path = os.path.join(config.path_outputs, path_out_temp)
                if not os.path.exists(path):
                    os.makedirs(path)
                path_filename = os.path.join(path, f'{time.time()}_{str(random.randint(0, 9999)).zfill(4)}.{output_format}')
                with open(path_filename, 'wb') as file:
                    file.write(response.content)
            else:
                raise Exception(str(response.json()))
    except Exception as e:
        print(f"[SD3Generator] An error occurred: {e}")
    return path_filename

def scale_u64_to_u32(value):
    uint64_max = 2**64 - 1
    uint32_max = 2**32 - 1

    scale_factor = uint64_max // uint32_max

    scaled_value = (value // scale_factor) & uint32_max

    return scaled_value

