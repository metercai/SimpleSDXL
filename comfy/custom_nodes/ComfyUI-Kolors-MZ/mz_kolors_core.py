

import gc
import json
import os
import random
import re

import torch
import folder_paths
import comfy.model_management as mm


def KolorsTextEncode(chatglm3_model, prompt):
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    mm.unload_all_models()
    mm.soft_empty_cache()
    # Function to randomly select an option from the brackets

    def choose_random_option(match):
        options = match.group(1).split('|')
        return random.choice(options)

    prompt = re.sub(r'\{([^{}]*)\}', choose_random_option, prompt)

    if "|" in prompt:
        prompt = prompt.split("|")

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    # Define tokenizers and text encoders
    tokenizer = chatglm3_model['tokenizer']
    text_encoder = chatglm3_model['text_encoder']
    text_encoder.to(device)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    output = text_encoder(
        input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask'],
        position_ids=text_inputs['position_ids'],
        output_hidden_states=True)

    # [batch_size, 77, 4096]
    prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
    text_proj = output.hidden_states[-1][-1,
                                         :, :].clone()  # [batch_size, 4096]
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(
        bs_embed, seq_len, -1)

    bs_embed = text_proj.shape[0]
    text_proj = text_proj.repeat(1, 1).view(
        bs_embed, -1
    )
    text_encoder.to(offload_device)
    mm.soft_empty_cache()
    gc.collect()
    return prompt_embeds, text_proj


def MZ_ChatGLM3Loader_call(args):
    # from .mz_kolors_utils import Utils
    # llm_dir = os.path.join(Utils.get_models_path(), "LLM")
    chatglm3_checkpoint = args.get("chatglm3_checkpoint")

    chatglm3_checkpoint_path = folder_paths.get_full_path('llms', chatglm3_checkpoint)

    if not os.path.exists(chatglm3_checkpoint_path):
        raise RuntimeError(
            f"ERROR: Could not find chatglm3 checkpoint: {chatglm3_checkpoint_path}")

    from .chatglm3.configuration_chatglm import ChatGLMConfig
    from .chatglm3.modeling_chatglm import ChatGLMModel
    from .chatglm3.tokenization_chatglm import ChatGLMTokenizer

    from .mz_kolors_utils import Utils

    offload_device = mm.unet_offload_device()

    text_encoder_config = os.path.join(
        os.path.dirname(__file__), 'configs', 'text_encoder_config.json')
    with open(text_encoder_config, 'r') as file:
        config = json.load(file)

    text_encoder_config = ChatGLMConfig(**config)

    from comfy.utils import load_torch_file
    from contextlib import nullcontext
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        is_accelerate_available = True
    except:
        pass

    with (init_empty_weights() if is_accelerate_available else nullcontext()):
        with torch.no_grad():
            # 打印版本号
            print("torch version:", torch.__version__)
            text_encoder = ChatGLMModel(text_encoder_config).eval()
            if '4bit' in chatglm3_checkpoint:
                text_encoder.quantize(4)
            elif '8bit' in chatglm3_checkpoint:
                text_encoder.quantize(8)
    text_encoder_sd = load_torch_file(chatglm3_checkpoint_path)
    if is_accelerate_available:
        for key in text_encoder_sd:
            set_module_tensor_to_device(
                text_encoder, key, device=offload_device, value=text_encoder_sd[key])
    else:
        text_encoder.load_state_dict()

    tokenizer_path = os.path.join(
        os.path.dirname(__file__), 'configs', "tokenizer")
    tokenizer = ChatGLMTokenizer.from_pretrained(tokenizer_path)

    return ({"text_encoder": text_encoder, "tokenizer": tokenizer},)


def MZ_ChatGLM3TextEncode_call(args):

    text = args.get("text")
    chatglm3_model = args.get("chatglm3_model")

    prompt_embeds, pooled_output = KolorsTextEncode(
        chatglm3_model,
        text,
    )

    from torch import nn
    hid_proj: nn.Linear = args.get("hid_proj")

    if hid_proj.weight.dtype != prompt_embeds.dtype:
        with torch.cuda.amp.autocast(dtype=hid_proj.weight.dtype):
            prompt_embeds = hid_proj(prompt_embeds)
    else:
        prompt_embeds = hid_proj(prompt_embeds)

    return ([[
        prompt_embeds,
        {"pooled_output": pooled_output},
    ]], )


def MZ_ChatGLM3TextEncodeV2_call(args):

    text = args.get("text")
    chatglm3_model = args.get("chatglm3_model")

    prompt_embeds, pooled_output = KolorsTextEncode(
        chatglm3_model,
        text,
    )

    return ([[
        prompt_embeds,
        {"pooled_output": pooled_output},
    ]], )


import comfy

import comfy.samplers as samplers
if "original_CFGGuider_inner_set_conds" not in globals():
    original_CFGGuider_inner_set_conds = samplers.CFGGuider.set_conds


def patched_set_conds(self, positive, negative):
    if "kolors_hid_proj" in self.model_options:
        import copy
        hid_proj = self.model_options["kolors_hid_proj"]
        positive = copy.deepcopy(positive)
        negative = copy.deepcopy(negative)

        if hid_proj is not None:
            positive[0][0] = hid_proj(positive[0][0])
            negative[0][0] = hid_proj(negative[0][0])

            # comfy.mz_log("positive", positive)
            # comfy.mz_log("negative", negative)

            if "control" in positive[0][1]:
                if hasattr(positive[0][1]["control"], "control_model"):
                    positive[0][1]["control"].control_model.label_emb = self.model_patcher.model.diffusion_model.label_emb

            if "control" in negative[0][1]:
                if hasattr(negative[0][1]["control"], "control_model"):
                    negative[0][1]["control"].control_model.label_emb = self.model_patcher.model.diffusion_model.label_emb

    return original_CFGGuider_inner_set_conds(self, positive, negative)


samplers.CFGGuider.set_conds = patched_set_conds


def MZ_KolorsUNETLoaderV2_call(kwargs):
    # samplers.CFGGuider.set_conds = patched_set_conds

    from torch import nn
    from . import hook_comfyui
    import comfy.sd

    load_device = mm.get_torch_device()
    with hook_comfyui.apply_kolors():
        unet_name = kwargs.get("unet_name")
        unet_path = folder_paths.get_full_path("unet", unet_name)
        import comfy.utils
        sd = comfy.utils.load_torch_file(unet_path)

        encoder_hid_proj_weight = sd.pop("encoder_hid_proj.weight")
        encoder_hid_proj_bias = sd.pop("encoder_hid_proj.bias")
        hid_proj = nn.Linear(
            encoder_hid_proj_weight.shape[1], encoder_hid_proj_weight.shape[0])
        hid_proj.weight.data = encoder_hid_proj_weight
        hid_proj.bias.data = encoder_hid_proj_bias
        hid_proj = hid_proj.to(load_device)

        model = comfy.sd.load_unet_state_dict(sd)
        if model is None:
            raise RuntimeError(
                "ERROR: Could not detect model type of: {}".format(unet_path))

        model.model_options["kolors_hid_proj"] = hid_proj
        # comfy.mz_log("model1", model)

        model_function_wrapper = model.model_options.get(
            "model_function_wrapper", None)

        # Callable[[UnetApplyFunction, UnetParams], torch.Tensor]
        # model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}
        # def kolors_unet_forward_wrapper(apply_model, unet_params):
        #     input_x = unet_params["input"]
        #     timestep_ = unet_params["timestep"]
        #     c = unet_params["c"]
        #     cond_or_uncond = unet_params["cond_or_uncond"]

        #     comfy.mz_log("unet_params", unet_params)
        #     comfy.mz_log("input_x", input_x)
        #     comfy.mz_log("timestep_", timestep_)
        #     comfy.mz_log("c", c)
        #     comfy.mz_log("c_crossattn", c["c_crossattn"])
        #     comfy.mz_log("cond_or_uncond", cond_or_uncond)

        #     unet_params["c"]["c_crossattn"] = hid_proj(
        #         unet_params["c"]["c_crossattn"])

        #     if model_function_wrapper is not None:
        #         return model_function_wrapper(apply_model, unet_params)
        #     else:
        #         return apply_model(input_x, timestep_, **unet_params["c"])

        # model.set_model_unet_function_wrapper(kolors_unet_forward_wrapper)

        return (model, )


def MZ_FakeCond_call(kwargs):
    import torch
    # cond: torch.Size([1, 77, ])
    cond = torch.zeros(2, 256, 4096)
    # torch.Size([1, 1280])
    pool = torch.zeros(2, 4096)

    return ([[
        cond,
        {"pooled_output": pool},
    ]],)


def load_unet_state_dict(sd):  # load unet in diffusers or regular format
    from comfy import model_management, model_detection
    import comfy.utils

    # Allow loading unets from checkpoint files
    checkpoint = False
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd
        checkpoint = True

    parameters = comfy.utils.calculate_parameters(sd)
    unet_dtype = model_management.unet_dtype(model_params=parameters)
    load_device = model_management.get_torch_device()

    from torch import nn
    hid_proj: nn.Linear = None
    if True:
        model_config = model_detection.model_config_from_diffusers_unet(sd)
        if model_config is None:
            return None

        diffusers_keys = comfy.utils.unet_to_diffusers(
            model_config.unet_config)

        new_sd = {}
        for k in diffusers_keys:
            if k in sd:
                new_sd[diffusers_keys[k]] = sd.pop(k)
            else:
                print("{} {}".format(diffusers_keys[k], k))

        encoder_hid_proj_weight = sd.pop("encoder_hid_proj.weight")
        encoder_hid_proj_bias = sd.pop("encoder_hid_proj.bias")
        hid_proj = nn.Linear(
            encoder_hid_proj_weight.shape[1], encoder_hid_proj_weight.shape[0])
        hid_proj.weight.data = encoder_hid_proj_weight
        hid_proj.bias.data = encoder_hid_proj_bias
        hid_proj = hid_proj.to(load_device)

    offload_device = model_management.unet_offload_device()
    unet_dtype = model_management.unet_dtype(
        model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys in unet: {}".format(left_over))
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device), hid_proj


def MZ_KolorsUNETLoader_call(kwargs):

    from . import hook_comfyui
    with hook_comfyui.apply_kolors():
        unet_name = kwargs.get("unet_name")
        unet_path = folder_paths.get_full_path("unet", unet_name)
        import comfy.utils
        sd = comfy.utils.load_torch_file(unet_path)
        model, hid_proj = load_unet_state_dict(sd)
        if model is None:
            raise RuntimeError(
                "ERROR: Could not detect model type of: {}".format(unet_path))
        return (model, hid_proj)
