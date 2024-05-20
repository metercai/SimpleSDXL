from hydit.constants import SAMPLER_FACTORY
from hydit.config import get_args
from hydit.inference import End2End
from modules.config import path_t2i
import tarfile
from pathlib import Path
import numpy as np
from io import BytesIO
from PIL import Image
import os
from modules.model_loader import load_file_from_url


SAMPLERS = list(SAMPLER_FACTORY.keys())
default_sampler=SAMPLERS[0]
new_args = ["--use_fp16", "--lang", "zh", "--load-key", "module", "--infer-mode", "fa"]
hydit_args = get_args(new_args)
gen = None

def init_load_model():
    global hydit_args, gen

    hydit_models_root = Path(os.path.join(path_t2i, "t2i"))
    if not hydit_models_root.exists():
        print(f"`hydit_models_root` not exists: {hydit_models_root}")
        hydit_models_root.mkdir(parents=True, exist_ok=True)
        downloading_hydit_model(hydit_models_root)
    if gen is None:
        gen = End2End(hydit_args, path_t2i)

def unload_free_model():
    global gen
    if gen is not None:
        gen = None

def get_scheduler_name(sampler):
    params = SAMPLER_FACTORY[sampler]
    return params["scheduler"], params["name"]


def inferencer(
    prompt,
    negative_prompt,
    seed,
    cfg_scale,
    infer_steps,
    width, height,
    sampler,
):
    seed = seed & 0xFFFFFFFF
    enhanced_prompt = None
    params = SAMPLER_FACTORY[sampler]
    print(f'[HyDiT] Ready to HyDiT Task:\n    prompt={prompt}\n    negative_prompt={negative_prompt}\n    seed={seed}\n    cfg_scale={cfg_scale}\n    steps={infer_steps}\n    width,height={width},{height}\n    scheduler={params["scheduler"]}\n    sampler={params["name"]}')
    results = gen.predict(prompt,
                          height=height,
                          width=width,
                          seed=seed,
                          enhanced_prompt=enhanced_prompt,
                          negative_prompt=negative_prompt,
                          infer_steps=infer_steps,
                          guidance_scale=cfg_scale,
                          batch_size=1,
                          src_size_cond=(width, height),
                          sampler=sampler,
                          )
    image = results['images'][0]
    return [np.array(image)]

def downloading_hydit_model(path_root):
    load_file_from_url(
        url='https://huggingface.co/metercai/backup/resolve/main/hydit_models.tar.gz',
        model_dir=path_root,
        file_name='hydit_models.tar.gz'
    )
    downfile = os.path.join(path_root, 'hydit_models.tar.gz')
    with tarfile.open(downfile, 'r:gz') as tarf:
        tarf.extractall(path_root)
    os.remove(downfile)
    pass
