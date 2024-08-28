from .SeamlessTile import (CircularVAEDecode, MakeCircularVAE, OffsetImage,
                           TiledKSampler, SeamlessTile)

NODE_CLASS_MAPPINGS = {
    "TiledKSampler": TiledKSampler,
    "SeamlessTile": SeamlessTile,
    "CircularVAEDecode": CircularVAEDecode,
    "MakeCircularVAE": MakeCircularVAE,
    "OffsetImage": OffsetImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledKSampler": "KSampler(Tiled)",
    "SeamlessTile": "Seamless Tile",
    "CircularVAEDecode": "Circular VAE Decode (Tiled)",
    "MakeCircularVAE": "Make Circular VAE",
    "OffsetImage": "Offset Image",
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
