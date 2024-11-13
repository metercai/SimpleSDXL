class Model_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "model_1": ("MODEL",),
                "model_2": ("MODEL",),
            },
            "optional": {
                "model_3": ("MODEL",),  
                "model_4": ("MODEL",), 
                "model_5": ("MODEL",), 
                "model_6": ("MODEL",), 
                "model_7": ("MODEL",), 
                "model_8": ("MODEL",), 
                "model_9": ("MODEL",), 
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "model_selector"
    CATEGORY = "Zho模块组/Selector"

    def model_selector(self, model_1, model_2, select=1, 
                       model_3=None, model_4=None,
                       model_5=None, model_6=None,
                       model_7=None, model_8=None,
                       model_9=None):
        if select == 1:
            return (model_1, )
        elif select == 2:
            return (model_2, )
        elif select == 3:
            return (model_3, ) if model_3 is not None else ()
        elif select == 4:
            return (model_4, ) if model_4 is not None else ()
        elif select == 5:
            return (model_5, ) if model_5 is not None else ()
        elif select == 6:
            return (model_6, ) if model_6 is not None else ()
        elif select == 7:
            return (model_7, ) if model_7 is not None else ()
        elif select == 8:
            return (model_8, ) if model_8 is not None else ()
        elif select == 9:
            return (model_9, ) if model_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
class LoRA_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "loar_1": ("MODEL",),
                "loar_2": ("MODEL",),
            },
            "optional": {
                "loar_3": ("MODEL",),  
                "loar_4": ("MODEL",), 
                "loar_5": ("MODEL",), 
                "loar_6": ("MODEL",), 
                "loar_7": ("MODEL",), 
                "loar_8": ("MODEL",), 
                "loar_9": ("MODEL",), 
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "lora_selector" 
    CATEGORY = "Zho模块组/Selector"

    def lora_selector(self, loar_1, loar_2, select=1, 
                       loar_3=None, loar_4=None,
                       loar_5=None, loar_6=None,
                       loar_7=None, loar_8=None,
                       loar_9=None):
        if select == 1:
            return (loar_1, )
        elif select == 2:
            return (loar_2, )
        elif select == 3:
            return (loar_3, ) if loar_3 is not None else ()
        elif select == 4:
            return (loar_4, ) if loar_4 is not None else ()
        elif select == 5:
            return (loar_5, ) if loar_5 is not None else ()
        elif select == 6:
            return (loar_6, ) if loar_6 is not None else ()
        elif select == 7:
            return (loar_7, ) if loar_7 is not None else ()
        elif select == 8:
            return (loar_8, ) if loar_8 is not None else ()
        elif select == 9:
            return (loar_9, ) if loar_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
class CLIP_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "clip_1": ("CLIP",),  # 更改为 CLIP
                "clip_2": ("CLIP",),  # 更改为 CLIP
            },
            "optional": {
                "clip_3": ("CLIP",),  # 更改为 CLIP
                "clip_4": ("CLIP",),  # 更改为 CLIP
                "clip_5": ("CLIP",),  # 更改为 CLIP
                "clip_6": ("CLIP",),  # 更改为 CLIP
                "clip_7": ("CLIP",),  # 更改为 CLIP
                "clip_8": ("CLIP",),  # 更改为 CLIP
                "clip_9": ("CLIP",),  # 更改为 CLIP
            }
        }

    RETURN_TYPES = ("CLIP",)  # 更改为 CLIP
    FUNCTION = "clip_selector" 
    CATEGORY = "Zho模块组/Selector"

    def clip_selector(self, clip_1, clip_2, select=1, 
                       clip_3=None, clip_4=None,
                       clip_5=None, clip_6=None,
                       clip_7=None, clip_8=None,
                       clip_9=None):
        if select == 1:
            return (clip_1, )
        elif select == 2:
            return (clip_2, )
        elif select == 3:
            return (clip_3, ) if clip_3 is not None else ()
        elif select == 4:
            return (clip_4, ) if clip_4 is not None else ()
        elif select == 5:
            return (clip_5, ) if clip_5 is not None else ()
        elif select == 6:
            return (clip_6, ) if clip_6 is not None else ()
        elif select == 7:
            return (clip_7, ) if clip_7 is not None else ()
        elif select == 8:
            return (clip_8, ) if clip_8 is not None else ()
        elif select == 9:
            return (clip_9, ) if clip_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
class VAE_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "vae_1": ("VAE",),  # 更改为 VAE
                "vae_2": ("VAE",),  # 更改为 VAE
            },
            "optional": {
                "vae_3": ("VAE",),  # 更改为 VAE
                "vae_4": ("VAE",),  # 更改为 VAE
                "vae_5": ("VAE",),  # 更改为 VAE
                "vae_6": ("VAE",),  # 更改为 VAE
                "vae_7": ("VAE",),  # 更改为 VAE
                "vae_8": ("VAE",),  # 更改为 VAE
                "vae_9": ("VAE",),  # 更改为 VAE
            }
        }

    RETURN_TYPES = ("VAE",)  # 更改为 VAE
    FUNCTION = "vae_selector" 
    CATEGORY = "Zho模块组/Selector"

    def vae_selector(self, vae_1, vae_2, select=1, 
                       vae_3=None, vae_4=None,
                       vae_5=None, vae_6=None,
                       vae_7=None, vae_8=None,
                       vae_9=None):
        if select == 1:
            return (vae_1, )
        elif select == 2:
            return (vae_2, )
        elif select == 3:
            return (vae_3, ) if vae_3 is not None else ()
        elif select == 4:
            return (vae_4, ) if vae_4 is not None else ()
        elif select == 5:
            return (vae_5, ) if vae_5 is not None else ()
        elif select == 6:
            return (vae_6, ) if vae_6 is not None else ()
        elif select == 7:
            return (vae_7, ) if vae_7 is not None else ()
        elif select == 8:
            return (vae_8, ) if vae_8 is not None else ()
        elif select == 9:
            return (vae_9, ) if vae_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
class Text_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "text_1": ("STRING", {"multiline": True}),
                "text_2": ("STRING", {"multiline": True}),
            },
            "optional": {
                "text_3": ("STRING", {"multiline": True}),  
                "text_4": ("STRING", {"multiline": True}),
                "text_5": ("STRING", {"multiline": True}),
                "text_6": ("STRING", {"multiline": True}),
                "text_7": ("STRING", {"multiline": True}),
                "text_8": ("STRING", {"multiline": True}),
                "text_9": ("STRING", {"multiline": True}), 
            }
        }

    RETURN_TYPES = ("STRING",)  # Change this to "STRING"
    FUNCTION = "txt_selector"  # Change this to "txt_selector"
    CATEGORY = "Zho模块组/Selector"

    def txt_selector(self, text_1, text_2, select=1, 
                     text_3=None, text_4=None,
                     text_5=None, text_6=None,
                     text_7=None, text_8=None,
                     text_9=None):
        if select == 1:
            return (text_1, )
        elif select == 2:
            return (text_2, )
        elif select == 3:
            return (text_3, ) if text_3 is not None else ()
        elif select == 4:
            return (text_4, ) if text_4 is not None else ()
        elif select == 5:
            return (text_5, ) if text_5 is not None else ()
        elif select == 6:
            return (text_6, ) if text_6 is not None else ()
        elif select == 7:
            return (text_7, ) if text_7 is not None else ()
        elif select == 8:
            return (text_8, ) if text_8 is not None else ()
        elif select == 9:
            return (text_9, ) if text_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
class Image_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
            "optional": {
                "image_3": ("IMAGE",),  
                "image_4": ("IMAGE",), 
                "image_5": ("IMAGE",), 
                "image_6": ("IMAGE",), 
                "image_7": ("IMAGE",), 
                "image_8": ("IMAGE",), 
                "image_9": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_selector"  # Change this to "image_selector"
    CATEGORY = "Zho模块组/Selector"

    def image_selector(self, image_1, image_2, select=1, 
                       image_3=None, image_4=None,
                       image_5=None, image_6=None,
                       image_7=None, image_8=None,
                       image_9=None):
        if select == 1:
            return (image_1, )
        elif select == 2:
            return (image_2, )
        elif select == 3:
            return (image_3, ) if image_3 is not None else ()
        elif select == 4:
            return (image_4, ) if image_4 is not None else ()
        elif select == 5:
            return (image_5, ) if image_5 is not None else ()
        elif select == 6:
            return (image_6, ) if image_6 is not None else ()
        elif select == 7:
            return (image_7, ) if image_7 is not None else ()
        elif select == 8:
            return (image_8, ) if image_8 is not None else ()
        elif select == 9:
            return (image_9, ) if image_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
class Latent_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "latent_1": ("LATENT",),
                "latent_2": ("LATENT",),
            },
            "optional": {
                "latent_3": ("LATENT",),  
                "latent_4": ("LATENT",), 
                "latent_5": ("LATENT",), 
                "latent_6": ("LATENT",), 
                "latent_7": ("LATENT",), 
                "latent_8": ("LATENT",), 
                "latent_9": ("LATENT",), 
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "latent_selector" 
    CATEGORY = "Zho模块组/Selector"

    def latent_selector(self, latent_1, latent_2, select=1, 
                       latent_3=None, latent_4=None,
                       latent_5=None, latent_6=None,
                       latent_7=None, latent_8=None,
                       latent_9=None):
        if select == 1:
            return (latent_1, )
        elif select == 2:
            return (latent_2, )
        elif select == 3:
            return (latent_3, ) if latent_3 is not None else ()
        elif select == 4:
            return (latent_4, ) if latent_4 is not None else ()
        elif select == 5:
            return (latent_5, ) if latent_5 is not None else ()
        elif select == 6:
            return (latent_6, ) if latent_6 is not None else ()
        elif select == 7:
            return (latent_7, ) if latent_7 is not None else ()
        elif select == 8:
            return (latent_8, ) if latent_8 is not None else ()
        elif select == 9:
            return (latent_9, ) if latent_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
class Conditioning_Input_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
            },
            "optional": {
                "conditioning_3": ("CONDITIONING",),  
                "conditioning_4": ("CONDITIONING",), 
                "conditioning_5": ("CONDITIONING",), 
                "conditioning_6": ("CONDITIONING",), 
                "conditioning_7": ("CONDITIONING",), 
                "conditioning_8": ("CONDITIONING",), 
                "conditioning_9": ("CONDITIONING",), 
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "conditioning_input_selector"
    CATEGORY = "Zho模块组/Selector"

    def conditioning_input_selector(self, conditioning_1, conditioning_2, select=1, 
                                    conditioning_3=None, conditioning_4=None,
                                    conditioning_5=None, conditioning_6=None,
                                    conditioning_7=None, conditioning_8=None,
                                    conditioning_9=None):
        if select == 1:
            return (conditioning_1, )
        elif select == 2:
            return (conditioning_2, )
        elif select == 3:
            return (conditioning_3, ) if conditioning_3 is not None else ()
        elif select == 4:
            return (conditioning_4, ) if conditioning_4 is not None else ()
        elif select == 5:
            return (conditioning_5, ) if conditioning_5 is not None else ()
        elif select == 6:
            return (conditioning_6, ) if conditioning_6 is not None else ()
        elif select == 7:
            return (conditioning_7, ) if conditioning_7 is not None else ()
        elif select == 8:
            return (conditioning_8, ) if conditioning_8 is not None else ()
        elif select == 9:
            return (conditioning_9, ) if conditioning_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
class Ctrlnet_Selector_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 9}),
                "ctrlnet_1": ("CONDITIONING",),
                "ctrlnet_2": ("CONDITIONING",),
            },
            "optional": {
                "ctrlnet_3": ("CONDITIONING",),  
                "ctrlnet_4": ("CONDITIONING",), 
                "ctrlnet_5": ("CONDITIONING",), 
                "ctrlnet_6": ("CONDITIONING",), 
                "ctrlnet_7": ("CONDITIONING",), 
                "ctrlnet_8": ("CONDITIONING",), 
                "ctrlnet_9": ("CONDITIONING",), 
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "ctrlnet_selector"
    CATEGORY = "Zho模块组/Selector"

    def ctrlnet_selector(self, ctrlnet_1, ctrlnet_2, select=1, 
                       ctrlnet_3=None, ctrlnet_4=None,
                       ctrlnet_5=None, ctrlnet_6=None,
                       ctrlnet_7=None, ctrlnet_8=None,
                       ctrlnet_9=None):
        if select == 1:
            return (ctrlnet_1, )
        elif select == 2:
            return (ctrlnet_2, )
        elif select == 3:
            return (ctrlnet_3, ) if ctrlnet_3 is not None else ()
        elif select == 4:
            return (ctrlnet_4, ) if ctrlnet_4 is not None else ()
        elif select == 5:
            return (ctrlnet_5, ) if ctrlnet_5 is not None else ()
        elif select == 6:
            return (ctrlnet_6, ) if ctrlnet_6 is not None else ()
        elif select == 7:
            return (ctrlnet_7, ) if ctrlnet_7 is not None else ()
        elif select == 8:
            return (ctrlnet_8, ) if ctrlnet_8 is not None else ()
        elif select == 9:
            return (ctrlnet_9, ) if ctrlnet_9 is not None else ()
        else:
            raise ValueError("Invalid select value")

#------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Model_Selector_Zho": Model_Selector_Zho,
    "LoRA_Selector_Zho": LoRA_Selector_Zho,
    "CLIP_Selector_Zho": CLIP_Selector_Zho,
    "VAE_Selector_Zho": VAE_Selector_Zho,
    "Text_Selector_Zho": Text_Selector_Zho,
    "Image_Selector_Zho": Image_Selector_Zho,
    "Latent_Selector_Zho": Latent_Selector_Zho,
    "Conditioning_Input_Selector_Zho": Conditioning_Input_Selector_Zho,
    "Ctrlnet_Selector_Zho": Ctrlnet_Selector_Zho,
}
