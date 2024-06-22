from pathlib import Path
import json
import threading
from .civit import Civit
import time
import os
from custom.OneButtonPrompt.utils import path_fixed, root_path_fixed

class PathManager:
    DEFAULT_PATHS = {
        "path_checkpoints": root_path_fixed("../models/checkpoints/"),
        "path_loras": root_path_fixed("../models/loras/"),
        "path_controlnet": root_path_fixed("../models/controlnet/"),
        "path_vae_approx": root_path_fixed("../models/vae_approx/"),
        "path_preview": root_path_fixed("../outputs/preview.jpg"),
        "path_faceswap": root_path_fixed("../models/faceswap/"),
        "path_upscalers": root_path_fixed("../models/upscale_models"),
        "path_outputs": root_path_fixed("../outputs/"),
        "path_clip": root_path_fixed("../models/clip/"),
        "path_cache": root_path_fixed("../cache/"),
    }

    EXTENSIONS = [".pth", ".ckpt", ".bin", ".safetensors"]

    civit_worker_folders = []
    loras_keywords_path = path_fixed("models/loras/")

    def __init__(self):
        self.paths = self.load_paths()
        self.model_paths = self.get_model_paths()
        self.default_model_names = self.get_default_model_names()
        self.update_all_model_names()

    def load_paths(self):
        paths = self.DEFAULT_PATHS.copy()
        settings_path = Path(path_fixed("settings/paths.json"))
        if settings_path.exists():
            with settings_path.open() as f:
                paths.update(json.load(f))
        for key in self.DEFAULT_PATHS:
            if key not in paths:
                paths[key] = self.DEFAULT_PATHS[key]
        with settings_path.open("w") as f:
            json.dump(paths, f, indent=2)
        return paths

    def get_model_paths(self):
        return {
            "modelfile_path": self.get_abspath_folder(self.paths["path_checkpoints"]),
            "lorafile_path": self.get_abspath_folder(self.paths["path_loras"]),
            "controlnet_path": self.get_abspath_folder(self.paths["path_controlnet"]),
            "vae_approx_path": self.get_abspath_folder(self.paths["path_vae_approx"]),
            "temp_outputs_path": self.get_abspath_folder(self.paths["path_outputs"]),
            "temp_preview_path": self.get_abspath(self.paths["path_preview"]),
            "faceswap_path": self.get_abspath_folder(self.paths["path_faceswap"]),
            "upscaler_path": self.get_abspath_folder(self.paths["path_upscalers"]),
            "clip_path": self.get_abspath_folder(self.paths["path_clip"]),
            "cache_path": self.get_abspath_folder(self.paths["path_cache"]),
        }

    def get_default_model_names(self):
        return {
            "default_base_model_name": "sd_xl_base_1.0_0.9vae.safetensors",
            "default_lora_name": "sd_xl_offset_example-lora_1.0.safetensors",
            "default_lora_weight": 0.5,
        }

    def get_abspath_folder(self, path):
        folder = self.get_abspath(path)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_abspath(self, path):
        return Path(path) if Path(path).is_absolute() else Path(__file__).parent / path

    def civit_update_worker(self, folder_path, cache, isLora):
        if folder_path in self.civit_worker_folders:
            # Already working on this folder
            return
        else:
            return
        if cache:
            cache_path = Path(self.model_paths["cache_path"] / cache)
        self.civit_worker_folders.append(folder_path)
        civit = Civit()
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                txtcheck = Path(os.path.join(self.loras_keywords_path, path.with_suffix(".txt").name))
                # get file name, add cache path change suffix
                cache_file = Path(cache_path / path.name)

                has_preview = False
                suffixes = [".jpeg", ".jpg", ".png", ".gif"]
                for suffix in suffixes:
                    thumbcheck = cache_file.with_suffix(suffix)
                    if Path(thumbcheck).is_file():
                        has_preview = True
                        break

                if not has_preview:
                    hash = civit.model_hash(str(path))
                    print(f"Downloading model thumbnail for {path}")
                    models = civit.get_models_by_hash(hash)
                    civit.get_image(models, thumbcheck)
                    time.sleep(1)

                txtcheck = cache_file.with_suffix(".txt")
                if isLora and not txtcheck.exists():
                    hash = civit.model_hash(str(path))
                    print(f"[OneButtonPrompt] Downloading LoRA keywords for {path}")
                    models = civit.get_models_by_hash(hash)
                    keywords = civit.get_keywords(models)
                    txtcheck.parent.mkdir(parents=True, exist_ok=True)
                    with open(txtcheck, "w") as f:
                        f.write(", ".join(keywords))
                    time.sleep(1)

        self.civit_worker_folders.remove(folder_path)

    def get_model_filenames(self, folder_path, cache=None, isLora=False):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError("Folder path is not a valid directory.")
        threading.Thread(
            target=self.civit_update_worker,
            args=(
                folder_path,
                cache,
                isLora,
            ),
            daemon=True,
        ).start()
        filenames = []
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                if isLora:
                    txtcheck = path.with_suffix(".txt")
                    if txtcheck.exists():
                        fstats = txtcheck.stat()
                        if fstats.st_size > 0:
                            path = path.with_suffix(f"{path.suffix}")
                filenames.append(str(path.relative_to(folder_path)))
        # Return a sorted list, prepend names with 0 if they are in a folder or 1
        # if it is a plain file. This will sort folders above files in the dropdown
        return sorted(
            filenames,
            key=lambda x: f"0{x.casefold()}"
            if not str(Path(x).parent) == "."
            else f"1{x.casefold()}",
        )

    def update_all_model_names(self):
        self.model_filenames = self.get_model_filenames(
            self.model_paths["modelfile_path"],
            cache="checkpoints"
        )
        self.lora_filenames = self.get_model_filenames(
            self.model_paths["lorafile_path"], 
            cache="loras",
            isLora=True
        )
        self.upscaler_filenames = self.get_model_filenames(
            self.model_paths["upscaler_path"]
        )

    def find_lcm_lora(self):
        path = Path(self.model_paths["lorafile_path"])
        filename = "lcm-lora-sdxl.safetensors"
        for child in path.rglob(filename):
            if child.name == filename:
                return child.relative_to(path)
