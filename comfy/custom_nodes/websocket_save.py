from PIL import Image
import numpy as np
import comfy.utils
import time

#You can use this node to save full size images through the websocket, the
#images will be sent in exactly the same format as the image previews: as
#binary images on the websocket with a 8 byte header indicating the type
#of binary message (first 4 bytes) and the image format (next 4 bytes).

#Note that no metadata will be put in the images saved with this node.

class SaveImageWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                    # "format": (["PNG", "JPEG", "WEBP"], {"default": "PNG"})
                    }
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "api/image"

    def save_images(self, images): #, format):
        format = 'png'
        pbar = comfy.utils.ProgressBar(images.shape[0])
        step = 0
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pbar.update_absolute(step, images.shape[0], (format, img, None))
            step += 1

        return {}

    @classmethod
    def IS_CHANGED(s, images): #, format):
        return time.time()

class SaveImageWebsocketLazy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "format": (["PNG", "JPEG", "WEBP"], {"default": "PNG"})
                    }
                }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )

    FUNCTION = "save_images"

    OUTPUT_NODE = False

    CATEGORY = "api/image"

    def save_images(self, images, format):
        format = format.lower()
        pbar = comfy.utils.ProgressBar(images.shape[0])
        step = 0
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pbar.update_absolute(step, images.shape[0], (format, img, None))
            step += 1

        return (images,)

class SaveVideoWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "format": (["MP4", "WEBM"], {"default": "WEBM"}),
                     "codec": (["h264", "h265", "vp9", "av1"], {"default": "vp9"}),
                     "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 60.0, "step": 0.01}),
                     "crf": ("INT", {"default": 32, "min": 0, "max": 63, "step": 1}),
                    }
                }

    RETURN_TYPES = ()
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "api/video"

    def save_video(self, images, format, codec, fps, crf):
        import av
        import io
        import torch
        from tqdm import tqdm
        from fractions import Fraction

        codec_map = {
            "h264": "libx264",
            "h265": "libx265",
            "vp9": "libvpx-vp9",
            "av1": "libaom-av1"
        }

        if format == "MP4" and codec in ["vp9", "av1"]:
            codec = "h264"
        elif format == "WEBM" and codec in ["h264", "h265"]:
            codec = "vp9"
        if crf > 51 and format == "MP4":
            crf = 51

        buffer = io.BytesIO()

        container = av.open(buffer, mode='w', format=format.lower())

        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))

        height, width = images[0].shape[0], images[0].shape[1]
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'

        stream.options = {'crf': str(crf)}

        pbar = comfy.utils.ProgressBar(len(images))

        for i, img in tqdm(enumerate(images), desc="Encoding Frame", unit="frame", total=len(images)):
            frame_data = torch.clamp(img[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy()
            frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()

        video_data = buffer.getvalue()

        pbar.update_absolute(0, 1, (format, video_data, None))

        return {}

    @classmethod
    def IS_CHANGED(s, images, format, codec, fps, crf):
        return time.time()


NODE_CLASS_MAPPINGS = {
    "SaveImageWebsocket": SaveImageWebsocket,
    "SaveImageWebsocketLazy": SaveImageWebsocketLazy,
    "SaveVideoWebsocket": SaveVideoWebsocket,
}
