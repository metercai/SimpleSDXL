

import gc
import json
import os
import random
import re
import subprocess
import sys
from types import MethodType

import torch
import folder_paths
import comfy.model_management as mm


def chatglm3_text_encode(chatglm3_model, prompt):
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    mm.unload_all_models()
    mm.soft_empty_cache()
    # Function to randomly select an option from the brackets

    def choose_random_option(match):
        options = match.group(1).split('|')
        return random.choice(options)

    prompt = re.sub(r'\{([^{}]*)\}', choose_random_option, prompt)

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

    chatglm3_checkpoint_path = folder_paths.get_full_path(
        'LLM', chatglm3_checkpoint)

    if not os.path.exists(chatglm3_checkpoint_path):
        raise RuntimeError(
            f"ERROR: Could not find chatglm3 checkpoint: {chatglm3_checkpoint_path}")

    from .chatglm3.configuration_chatglm import ChatGLMConfig
    from .chatglm3.modeling_chatglm import ChatGLMModel
    from .chatglm3.tokenization_chatglm import ChatGLMTokenizer

    offload_device = mm.unet_offload_device()

    text_encoder_config = os.path.join(
        os.path.dirname(__file__), 'configs', 'text_encoder_config.json')
    with open(text_encoder_config, 'r') as file:
        config = json.load(file)

    text_encoder_config = ChatGLMConfig(**config)

    from comfy.utils import load_torch_file
    from contextlib import nullcontext
    is_accelerate_available = False
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
                try:
                    import cpm_kernels
                except ImportError:
                    print("Installing cpm_kernels...")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "cpm_kernels"], check=True)
                    pass
                text_encoder.quantize(4)
            elif '8bit' in chatglm3_checkpoint:
                text_encoder.quantize(8)
    text_encoder_sd = load_torch_file(chatglm3_checkpoint_path)
    if is_accelerate_available:
        for key in text_encoder_sd:
            set_module_tensor_to_device(
                text_encoder, key, device=offload_device, value=text_encoder_sd[key])
    else:
        print("WARNING: Accelerate not available, use load_state_dict load model")
        text_encoder.load_state_dict(text_encoder_sd)

    tokenizer_path = os.path.join(
        os.path.dirname(__file__), 'configs', "tokenizer")
    tokenizer = ChatGLMTokenizer.from_pretrained(tokenizer_path)

    return ({"text_encoder": text_encoder, "tokenizer": tokenizer},)


def MZ_ChatGLM3TextEncodeV2_call(args):
    text = args.get("text")
    chatglm3_model = args.get("chatglm3_model")
    prompt_embeds, pooled_output = chatglm3_text_encode(
        chatglm3_model,
        text,
    )
    extra_kwargs = {
        "pooled_output": pooled_output,
    }
    extra_cond_keys = [
        "width",
        "height",
        "crop_w",
        "crop_h",
        "target_width",
        "target_height"
    ]
    for key, value in args.items():
        if key in extra_cond_keys:
            extra_kwargs[key] = value
    return ([[
        prompt_embeds,
        # {"pooled_output": pooled_output},
        extra_kwargs
    ]], )


def MZ_ChatGLM3Embeds2Conditioning_call(args):
    kolors_embeds = args.get("kolors_embeds")

    # kolors_embeds = {
    #     'prompt_embeds': prompt_embeds,
    #     'negative_prompt_embeds': negative_prompt_embeds,
    #     'pooled_prompt_embeds': text_proj,
    #     'negative_pooled_prompt_embeds': negative_text_proj
    # }

    positive = [[
        kolors_embeds['prompt_embeds'],
        {
            "pooled_output": kolors_embeds['pooled_prompt_embeds'],
            "width": args.get("width"),
            "height": args.get("height"),
            "crop_w": args.get("crop_w"),
            "crop_h": args.get("crop_h"),
            "target_width": args.get("target_width"),
            "target_height": args.get("target_height")
        }
    ]]

    negative = [[
        kolors_embeds['negative_prompt_embeds'],
        {
            "pooled_output": kolors_embeds['negative_pooled_prompt_embeds'],
        }
    ]]

    return (positive, negative)


def MZ_KolorsUNETLoaderV2_call(kwargs):

    from . import hook_comfyui_kolors_v2
    import comfy.sd

    with hook_comfyui_kolors_v2.apply_kolors():
        unet_name = kwargs.get("unet_name")
        unet_path = folder_paths.get_full_path("unet", unet_name)
        import comfy.utils
        sd = comfy.utils.load_torch_file(unet_path)
        model = comfy.sd.load_unet_state_dict(sd)
        if model is None:
            raise RuntimeError(
                "ERROR: Could not detect model type of: {}".format(unet_path))

        return (model, )


def MZ_KolorsCheckpointLoaderSimple_call(kwargs):
    checkpoint_name = kwargs.get("ckpt_name")

    ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)

    from . import hook_comfyui_kolors_v2
    import comfy.sd

    with hook_comfyui_kolors_v2.apply_kolors():
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=False, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        unet, _, vae = out[:3]
        return (unet, vae)


from comfy.cldm.cldm import ControlNet
from comfy.controlnet import ControlLora


def MZ_KolorsControlNetLoader_call(kwargs):
    control_net_name = kwargs.get("control_net_name")
    controlnet_path = folder_paths.get_full_path(
        "controlnet", control_net_name)

    from torch import nn
    from . import hook_comfyui_kolors_v2
    import comfy.controlnet

    with hook_comfyui_kolors_v2.apply_kolors():
        control_net = comfy.controlnet.load_controlnet(controlnet_path)
        return (control_net, )


def MZ_KolorsControlNetPatch_call(kwargs):
    import copy
    from . import hook_comfyui_kolors_v2
    import comfy.model_management
    import comfy.model_patcher

    model = kwargs.get("model")
    control_net = kwargs.get("control_net")

    if hasattr(control_net, "control_model") and hasattr(control_net.control_model, "encoder_hid_proj"):
        return (control_net,)

    control_net = copy.deepcopy(control_net)

    import comfy.controlnet
    if isinstance(control_net, ControlLora):
        del_keys = []
        for k in control_net.control_weights:
            if k.startswith("label_emb.0.0."):
                del_keys.append(k)

        for k in del_keys:
            control_net.control_weights.pop(k)

        super_pre_run = ControlLora.pre_run
        super_forward = ControlNet.forward

        def KolorsControlNet_forward(self, x, hint, timesteps, context, **kwargs):
            with torch.cuda.amp.autocast(enabled=True):
                context = self.encoder_hid_proj(context)
                return super_forward(self, x, hint, timesteps, context, **kwargs)

        def KolorsControlLora_pre_run(self, *args, **kwargs):
            result = super_pre_run(self, *args, **kwargs)

            if hasattr(self, "control_model"):
                if hasattr(self.control_model, "encoder_hid_proj"):
                    return result

                setattr(self.control_model, "encoder_hid_proj",
                        model.model.diffusion_model.encoder_hid_proj)

                self.control_model.forward = MethodType(
                    KolorsControlNet_forward, self.control_model)

            return result

        control_net.pre_run = MethodType(
            KolorsControlLora_pre_run, control_net)

        super_copy = ControlLora.copy

        def KolorsControlLora_copy(self, *args, **kwargs):
            c = super_copy(self, *args, **kwargs)
            c.pre_run = MethodType(
                KolorsControlLora_pre_run, c)
            return c

        control_net.copy = MethodType(
            KolorsControlLora_copy, control_net)

        control_net = copy.deepcopy(control_net)

    elif isinstance(control_net, comfy.controlnet.ControlNet):
        model_label_emb = model.model.diffusion_model.label_emb

        control_net.control_model.label_emb = model_label_emb
        setattr(control_net.control_model, "encoder_hid_proj",
                model.model.diffusion_model.encoder_hid_proj)

        control_net.control_model_wrapped = comfy.model_patcher.ModelPatcher(
            control_net.control_model, load_device=control_net.load_device, offload_device=comfy.model_management.unet_offload_device())

        super_forward = ControlNet.forward

        def KolorsControlNet_forward(self, x, hint, timesteps, context, **kwargs):
            with torch.cuda.amp.autocast(enabled=True):
                context = self.encoder_hid_proj(context)
                return super_forward(self, x, hint, timesteps, context, **kwargs)

        control_net.control_model.forward = MethodType(
            KolorsControlNet_forward, control_net.control_model)

    else:
        raise NotImplementedError(
            f"Type {control_net} not supported for KolorsControlNetPatch")

    return (control_net,)


def MZ_KolorsCLIPVisionLoader_call(kwargs):
    import comfy.clip_vision
    from . import hook_comfyui_kolors_v2
    clip_name = kwargs.get("clip_name")
    clip_path = folder_paths.get_full_path("clip_vision", clip_name)
    with hook_comfyui_kolors_v2.apply_kolors():
        clip_vision = comfy.clip_vision.load(clip_path)
        return (clip_vision,)


def MZ_ApplySDXLSamplingSettings_call(kwargs):
    model = kwargs.get("model").clone()

    import comfy.model_sampling
    sampling_base = comfy.model_sampling.ModelSamplingDiscrete
    sampling_type = comfy.model_sampling.EPS

    class SDXLSampling(sampling_base, sampling_type):
        pass

    model.model.model_config.sampling_settings["beta_schedule"] = "linear"
    model.model.model_config.sampling_settings["linear_start"] = 0.00085
    model.model.model_config.sampling_settings["linear_end"] = 0.012
    model.model.model_config.sampling_settings["timesteps"] = 1000

    model_sampling = SDXLSampling(model.model.model_config)

    model.add_object_patch("model_sampling", model_sampling)

    return (model,)


def MZ_ApplyCUDAGenerator_call(kwargs):
    model = kwargs.get("model")

    def prepare_noise(latent_image, seed, noise_inds=None):
        """
        creates random noise given a latent image and a seed.
        optional arg skip can be used to skip and discard x number of noise generations for a given seed
        """
        generator = torch.Generator(device="cuda").manual_seed(seed)
        if noise_inds is None:
            return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cuda")

        unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
        noises = []
        for i in range(unique_inds[-1] + 1):
            noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype,
                                layout=latent_image.layout, generator=generator, device="cuda")
            if i in unique_inds:
                noises.append(noise)
        noises = [noises[i] for i in inverse]
        noises = torch.cat(noises, axis=0)
        return noises

    import comfy.sample
    comfy.sample.prepare_noise = prepare_noise
    return (model,)
