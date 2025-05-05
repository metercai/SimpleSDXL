import importlib.metadata
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.31.0'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")

def print_memory(device):
    memory = torch.cuda.memory_allocated(device) / 1024**3
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    log.info(f"-------------------------------")
    log.info(f"Allocated memory: {memory=:.3f} GB")
    log.info(f"Max allocated memory: {max_memory=:.3f} GB")
    log.info(f"Max reserved memory: {max_reserved=:.3f} GB")
    log.info(f"-------------------------------")
    #memory_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    #log.info(f"Memory Summary:\n{memory_summary}")

def convert_to_diffusers(prefix, weights_sd):
    # convert from default LoRA to diffusers
    # https://github.com/kohya-ss/musubi-tuner/blob/main/convert_lora.py

    # get alphas
    lora_alphas = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]  # before first dot
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = weight

    new_weights_sd = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            if "alpha" in key:
                continue

            lora_name = key.split(".", 1)[0]  # before first dot

            module_name = lora_name[len(prefix) :]  # remove "lora_unet_"
            module_name = module_name.replace("_", ".")  # replace "_" with "."
            
            # HunyuanVideo lora name to module name: ugly but works
            #module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
            module_name = module_name.replace("single.transformer.blocks.", "single_transformer_blocks.")  # fix single blocks
            module_name = module_name.replace("transformer.blocks.", "transformer_blocks.")  # fix double blocks
            
            module_name = module_name.replace("img.", "img_")  # fix img
            module_name = module_name.replace("txt.", "txt_")  # fix txt
            module_name = module_name.replace("to.q", "to_q")  # fix attn
            module_name = module_name.replace("to.k", "to_k")
            module_name = module_name.replace("to.v", "to_v")
            module_name = module_name.replace("to.add.out", "to_add_out")
            module_name = module_name.replace("add.k.proj", "add_k_proj")
            module_name = module_name.replace("add.q.proj", "add_q_proj")
            module_name = module_name.replace("add.v.proj", "add_v_proj")
            module_name = module_name.replace("add.out.proj", "add_out_proj")
            module_name = module_name.replace("proj.", "proj_")  # fix proj
            module_name = module_name.replace("to.out", "to_out")  # fix to_out
            module_name = module_name.replace("ff.context", "ff_context")  # fix ff context
    
            diffusers_prefix = "transformer"
            if "lora_down" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                dim = weight.shape[0]
            elif "lora_up" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                dim = weight.shape[1]
            else:
                log.warning(f"unexpected key: {key} in default LoRA format")
                continue

            # scale weight by alpha
            if lora_name in lora_alphas:
                # we scale both down and up, so scale is sqrt
                scale = lora_alphas[lora_name] / dim
                scale = scale.sqrt()
                weight = weight * scale
            else:
                log.warning(f"missing alpha for {lora_name}")

            new_weights_sd[new_key] = weight

    return new_weights_sd

