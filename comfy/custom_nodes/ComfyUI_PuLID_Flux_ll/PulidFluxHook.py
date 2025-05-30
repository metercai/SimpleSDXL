import torch
from einops import rearrange
from torch import Tensor
from comfy.ldm.flux.layers import timestep_embedding
import comfy
from .patch_util import PatchKeys

def set_model_dit_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"]

    if "dit" not in to["patches_replace"]:
        to["patches_replace"]["dit"] = {}
    else:
        to["patches_replace"]["dit"] = to["patches_replace"]["dit"]

    if key not in to["patches_replace"]["dit"]:
        if "double_block" in key:
            if key == ("double_block", 18):
                to["patches_replace"]["dit"][key] = LastDitDoubleBlockReplace(pulid_patch, **patch_kwargs)
            else:
                to["patches_replace"]["dit"][key] = DitDoubleBlockReplace(pulid_patch, **patch_kwargs)
        else:
            to["patches_replace"]["dit"][key] = DitSingleBlockReplace(pulid_patch, **patch_kwargs)
        # model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["dit"][key].add(pulid_patch, **patch_kwargs)

def pulid_patch(img, pulid_model=None, ca_idx=None, weight=1.0, embedding=None, mask=None, transformer_options={}):
    pulid_img = weight * pulid_model.model.pulid_ca[ca_idx].to(img.device)(embedding, img)
    if mask is not None:
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        latent_image_shape = pulid_temp_attrs.get("latent_image_shape", None)
        if latent_image_shape is not None:
            bs, c, h, w = latent_image_shape
            mask = comfy.sampler_helpers.prepare_mask(mask, (bs, c, h, w), img.device)
            patch_size = transformer_options[PatchKeys.running_net_model].patch_size
            mask = comfy.ldm.common_dit.pad_to_patch_size(mask, (patch_size, patch_size))
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            # (b, seq_len, _) =>(b, seq_len, pulid.dim)
            mask = mask[..., 0].unsqueeze(-1).repeat(1, 1, pulid_img.shape[-1]).to(dtype=pulid_img.dtype)
            del patch_size, latent_image_shape

        pulid_img = pulid_img * mask

        del mask, pulid_temp_attrs

    return pulid_img

class DitDoubleBlockReplace:
    def __init__(self, callback, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_args, extra_options):
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        sigma = pulid_temp_attrs["timesteps"][0].detach().cpu().item()
        out = extra_options["original_block"](input_args)
        img = out['img']
        temp_img = img
        for i, callback in enumerate(self.callback):
            if self.kwargs[i]["sigma_start"] >= sigma >= self.kwargs[i]["sigma_end"]:
                img = img + callback(temp_img,
                                     pulid_model=self.kwargs[i]['pulid_model'],
                                     ca_idx=self.kwargs[i]['ca_idx'],
                                     weight=self.kwargs[i]['weight'],
                                     embedding=self.kwargs[i]['embedding'],
                                     mask = self.kwargs[i]['mask'],
                                     transformer_options=transformer_options
                                     )
        out['img'] = img

        del temp_img, pulid_temp_attrs, sigma, transformer_options, img

        return out


class LastDitDoubleBlockReplace(DitDoubleBlockReplace):
    def __call__(self, input_args, extra_options):
        out = super().__call__(input_args, extra_options)
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        pulid_temp_attrs["double_blocks_txt"] = out['txt']
        return out

class DitSingleBlockReplace:
    def __init__(self, callback, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_args, extra_options):
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})

        out = extra_options["original_block"](input_args)

        sigma = pulid_temp_attrs["timesteps"][0].detach().cpu().item()
        img = out['img']
        txt = pulid_temp_attrs['double_blocks_txt']
        real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
        temp_img = real_img
        for i, callback in enumerate(self.callback):
            if self.kwargs[i]["sigma_start"] >= sigma >= self.kwargs[i]["sigma_end"]:
                real_img = real_img + callback(temp_img,
                                               pulid_model=self.kwargs[i]['pulid_model'],
                                               ca_idx=self.kwargs[i]['ca_idx'],
                                               weight=self.kwargs[i]['weight'],
                                               embedding=self.kwargs[i]['embedding'],
                                               mask=self.kwargs[i]['mask'],
                                               transformer_options = transformer_options,
                                               )

        img = torch.cat((txt, real_img), 1)

        out['img'] = img

        del temp_img, pulid_temp_attrs, sigma, transformer_options, real_img, img

        return out

def pulid_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control = None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    transformer_options[PatchKeys.running_net_model] = self
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})

    for i, block in enumerate(self.double_blocks):
        # 0 -> 18
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask
                                                       },
                                                      {
                                                          "original_block": block_wrap,
                                                          "transformer_options": transformer_options
                                                      })
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe,
                             attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        # 0 -> 37
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask
                                                       },
                                                      {
                                                          "original_block": block_wrap,
                                                          "transformer_options": transformer_options
                                                      })
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add

    img = img[:, txt.shape[1]:, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    del transformer_options[PatchKeys.running_net_model]

    return img


def pulid_enter(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    pulid_temp_attrs['timesteps'] = timesteps
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask


def pulid_patch_double_blocks_after(img, txt, transformer_options):
    pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    pulid_temp_attrs['double_blocks_txt'] = txt
    return img, txt
