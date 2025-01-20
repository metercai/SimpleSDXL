import numpy as np
import torch
import comfy
import folder_paths
import nodes
import os
import math
import re
import safetensors
import glob
from collections import namedtuple

@torch.no_grad()
def automerge(tensor, threshold):
    (batchsize, slices, dim) = tensor.shape
    newTensor=[]
    for batch in range(batchsize):
        tokens = []
        lastEmbed = tensor[batch,0,:]
        merge=[lastEmbed]
        tokens.append(lastEmbed)
        for i in range(1,slices):
            tok = tensor[batch,i,:]
            cosine = torch.dot(tok,lastEmbed)/torch.sqrt(torch.dot(tok,tok)*torch.dot(lastEmbed,lastEmbed))
            if cosine >= threshold:
                merge.append(tok)
                lastEmbed = torch.stack(merge).mean(dim=0)
            else:
                tokens.append(lastEmbed)
                merge=[]
                lastEmbed=tok
        newTensor.append(torch.stack(tokens))
    return torch.stack(newTensor)

STRENGTHS = ["highest", "high", "medium", "low", "lowest"]
STRENGTHS_VALUES = [1,2, 3,4,5]

class StyleModelApplySimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "image_strength": (STRENGTHS, {"default": "medium"})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision_output, style_model, conditioning, image_strength):
        stren = STRENGTHS.index(image_strength)
        downsampling_factor = STRENGTHS_VALUES[stren]
        mode="area" if downsampling_factor==3 else "bicubic"
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if downsampling_factor>1:
            (b,t,h)=cond.shape
            m = int(np.sqrt(t))
            cond=torch.nn.functional.interpolate(cond.view(b, m, m, h).transpose(1,-1), size=(m//downsampling_factor, m//downsampling_factor), mode=mode)#
            cond=cond.transpose(1,-1).reshape(b,-1,h)
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )

def standardizeMask(mask):
    if mask is None:
        return None
    if len(mask.shape) == 2:
        (h,w)=mask.shape
        mask=mask.view(1,1,h,w)
    elif len(mask.shape)==3:
        (b,h,w)=mask.shape
        mask=mask.view(b,1,h,w)
    return mask

def crop(img, mask, box, desiredSize):
    (ox,oy,w,h) = box
    if mask is not None:
        mask=torch.nn.functional.interpolate(mask, size=(h,w), mode="bicubic").view(-1,h,w,1)
    img = torch.nn.functional.interpolate(img.transpose(-1,1), size=(w,h), mode="bicubic", antialias=True)
    return (img[:, :, ox:(desiredSize+ox), oy:(desiredSize+oy)].transpose(1,-1), None if mask == None else mask[:, oy:(desiredSize+oy), ox:(desiredSize+ox),:])

def letterbox(img, mask, w, h, desiredSize):
    (b,oh,ow,c) = img.shape
    img = torch.nn.functional.interpolate(img.transpose(-1,1), size=(w,h), mode="bicubic", antialias=True).transpose(1,-1)
    letterbox = torch.zeros(size=(b,desiredSize,desiredSize, c))
    offsetx = (desiredSize-w)//2
    offsety = (desiredSize-h)//2
    letterbox[:, offsety:(offsety+h), offsetx:(offsetx+w), :] += img
    img = letterbox
    if mask is not None:
        mask=torch.nn.functional.interpolate(mask, size=(h,w), mode="bicubic")
        letterbox = torch.zeros(size=(b,1,desiredSize,desiredSize))
        letterbox[:, :, offsety:(offsety+h), offsetx:(offsetx+w)] += mask
        mask = letterbox.view(b,1,desiredSize,desiredSize)
    return (img, mask)

def getBoundingBox(mask, w, h, relativeMargin, desiredSize):
    mask=mask.view(h,w)
    marginW = math.ceil(relativeMargin * w)
    marginH = math.ceil(relativeMargin * h)
    indices = torch.nonzero(mask, as_tuple=False)
    y_min, x_min = indices.min(dim=0).values
    y_max, x_max = indices.max(dim=0).values    
    x_min = max(0, x_min.item() - marginW)
    y_min = max(0, y_min.item() - marginH)
    x_max = min(w, x_max.item() + marginW)
    y_max = min(h, y_max.item() + marginH)
    
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    larger_edge = max(box_width, box_height, desiredSize)
    if box_width < larger_edge:
        delta = larger_edge - box_width
        left_space = x_min
        right_space = w - x_max
        expand_left = min(delta // 2, left_space)
        expand_right = min(delta - expand_left, right_space)
        expand_left += min(delta - (expand_left+expand_right), left_space-expand_left)
        x_min -= expand_left
        x_max += expand_right

    if box_height < larger_edge:
        delta = larger_edge - box_height
        top_space = y_min
        bottom_space = h - y_max
        expand_top = min(delta // 2, top_space)
        expand_bottom = min(delta - expand_top, bottom_space)
        expand_top += min(delta - (expand_top+expand_bottom), top_space-expand_top)
        y_min -= expand_top
        y_max += expand_bottom

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    return x_min, y_min, x_max, y_max


def patchifyMask(mask, patchSize=14):
    if mask is None:
        return mask
    (b, imgSize, imgSize,_) = mask.shape
    toks = imgSize//patchSize
    return torch.nn.MaxPool2d(kernel_size=(patchSize,patchSize),stride=patchSize)(mask.view(b,imgSize,imgSize)).view(b,toks,toks,1)

def prepareImageAndMask(visionEncoder, image, mask, mode, autocrop_margin, desiredSize=384):
    mode = IMAGE_MODES.index(mode)
    (B,H,W,C) = image.shape
    if mode==0: # center crop square
        imgsize = min(H,W)
        ratio = desiredSize/imgsize
        (w,h) = (round(W*ratio), round(H*ratio))
        image, mask = crop(image, standardizeMask(mask), ((w - desiredSize)//2, (h - desiredSize)//2, w, h), desiredSize)
    elif mode==1:
        if mask is None:
            mask = torch.ones(size=(B,H,W))
        imgsize = max(H,W)
        ratio = desiredSize/imgsize
        (w,h) = (round(W*ratio), round(H*ratio))
        image, mask = letterbox(image, standardizeMask(mask), w, h, desiredSize)
    elif mode==2:
        (bx,by,bx2,by2) = getBoundingBox(mask,W,H,autocrop_margin, desiredSize)
        image = image[:,by:by2,bx:bx2,:]
        mask = mask[:,by:by2,bx:bx2]
        imgsize = max(bx2-bx,by2-by)
        ratio = desiredSize/imgsize
        (w,h) = (round((bx2-bx)*ratio), round((by2-by)*ratio))
        image, mask = letterbox(image, standardizeMask(mask), w, h, desiredSize)
    return (image,mask)

def processMask(mask,imgSize=384, patchSize=14):
    if len(mask.shape) == 2:
        (h,w)=mask.shape
        mask=mask.view(1,1,h,w)
    elif len(mask.shape)==3:
        (b,h,w)=mask.shape
        mask=mask.view(b,1,h,w)
    scalingFactor = imgSize/min(h,w)
    # scale
    mask=torch.nn.functional.interpolate(mask, size=(round(h*scalingFactor),round(w*scalingFactor)), mode="bicubic")
    # crop
    horizontalBorder = (imgSize-mask.shape[3])//2
    verticalBorder = (imgSize-mask.shape[2])//2
    mask=mask[:, :, verticalBorder:(verticalBorder+imgSize),horizontalBorder:(horizontalBorder+imgSize)].view(b,imgSize,imgSize)
    toks = imgSize//patchSize
    return torch.nn.MaxPool2d(kernel_size=(patchSize,patchSize),stride=patchSize)(mask).view(b,toks,toks,1)

IMAGE_MODES = [
    "center crop (square)",
    "keep aspect ratio",
    "autocrop with mask"
]

class ReduxAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision": ("CLIP_VISION", ),
                             "image": ("IMAGE",),
                             "downsampling_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 9.0}),
                             "downsampling_function": (["nearest", "bilinear", "bicubic","area","nearest-exact"], {"default": "area"}),
                             "mode": (IMAGE_MODES, {"default": "center crop (square)"}),
                             "weight": ("FLOAT", {"default": 1.0, "min":0.0, "max":1.0, "step":0.01})
                            },
                "optional": {
                            "mask": ("MASK", ),
                            "autocrop_margin": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01})
                }}
    RETURN_TYPES = ("CONDITIONING","IMAGE", "MASK")
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision, image, style_model, conditioning, downsampling_factor, downsampling_function,mode,weight, mask=None, autocrop_margin=0.0):
        image, masko = prepareImageAndMask(clip_vision, image, mask, mode, autocrop_margin)
        clip_vision_output,mask=(clip_vision.encode_image(image), patchifyMask(masko))
        mode="area"
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        (b,t,h)=cond.shape
        m = int(np.sqrt(t))
        if downsampling_factor>1:
            cond = cond.view(b, m, m, h)
            if mask is not None:
                cond = cond*mask
            downsampled_size = (round(m/downsampling_factor), round(m/downsampling_factor))
            cond=torch.nn.functional.interpolate(cond.transpose(1,-1), size=downsampled_size, mode=downsampling_function)
            cond=cond.transpose(1,-1).reshape(b,-1,h)
            mask = None if mask is None else torch.nn.functional.interpolate(mask.view(b, m, m, 1).transpose(1,-1), size=downsampled_size, mode=mode).transpose(-1,1)
        cond = cond*(weight*weight)
        c = []
        if mask is not None:
            mask = (mask>0).reshape(b,-1)
            max_len = mask.sum(dim=1).max().item()
            padded_embeddings = torch.zeros((b, max_len, h), dtype=cond.dtype, device=cond.device)
            for i in range(b):
                filtered = cond[i][mask[i]]
                padded_embeddings[i, :filtered.size(0)] = filtered
            cond = padded_embeddings

        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, image, masko)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "StyleModelApplySimple": StyleModelApplySimple,
    "ReduxAdvanced": ReduxAdvanced
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelApplySimple": "Apply style model (simple)",
    "ReduxAdvanced": "Apply Redux model (advanced)"
}