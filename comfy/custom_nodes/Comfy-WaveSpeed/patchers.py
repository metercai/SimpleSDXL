import contextlib
import copy
import unittest

import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils
import torch


class QuantizedModelPatcher(comfy.model_patcher.ModelPatcher):
    _object_to_patch_default = None
    _quantize_fn_default = None
    _lowvram_default = True
    _full_load_default = True
    _is_quantized_default = False

    _load_device = None
    _offload_device = None
    _disable_load = False

    @classmethod
    @contextlib.contextmanager
    def _override_defaults(cls, **kwargs):
        old_defaults = {}
        for k in ("object_to_patch", "quantize_fn", "lowvram", "full_load"):
            if k in kwargs:
                old_defaults[k] = getattr(cls, f"_{k}_default")
                setattr(cls, f"_{k}_default", kwargs[k])
        try:
            yield
        finally:
            for k in old_defaults:
                setattr(cls, f"_{k}_default", old_defaults[k])

    @classmethod
    @contextlib.contextmanager
    def _set_disable_load(cls, disable_load=True):
        old_disable_load = cls._disable_load
        cls._disable_load = disable_load
        try:
            yield
        finally:
            cls._disable_load = old_disable_load

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_to_patch = QuantizedModelPatcher._object_to_patch_default
        self._quantize_fn = QuantizedModelPatcher._quantize_fn_default
        self._lowvram = QuantizedModelPatcher._lowvram_default
        self._full_load = QuantizedModelPatcher._full_load_default
        self._is_quantized = QuantizedModelPatcher._is_quantized_default

    def load(
        self, device_to=None, force_patch_weights=False, full_load=False, **kwargs
    ):
        if self._disable_load:
            return

        if self._is_quantized:
            super().load(
                device_to=device_to,
                force_patch_weights=force_patch_weights,
                full_load=full_load,
                **kwargs,
            )
            return

        with unittest.mock.patch.object(
            QuantizedModelPatcher, "_load_device", self.load_device
        ), unittest.mock.patch.object(
            QuantizedModelPatcher, "_offload_device", self.offload_device
        ):
            # always call `patch_weight_to_device` even for lowvram
            super().load(
                torch.device("cpu") if self._lowvram else device_to,
                force_patch_weights=True,
                full_load=self._full_load or full_load,
                **kwargs,
            )

            if self._quantize_fn is not None:
                if self._object_to_patch is None:
                    target_model = self.model
                else:
                    target_model = comfy.utils.get_attr(
                        self.model, self._object_to_patch
                    )
                target_model = self._quantize_fn(target_model)
                if self._object_to_patch is None:
                    self.model = target_model
                else:
                    comfy.utils.set_attr(
                        self.model, self._object_to_patch, target_model
                    )

            if self._lowvram:
                if device_to.type == "cuda":
                    torch.cuda.empty_cache()
                self.model.to(device_to)

        self._is_quantized = True

    # def model_size(self):
    #     return super().model_size() // 2

    def clone(self, *args, **kwargs):
        n = QuantizedModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup

        n._object_to_patch = getattr(
            self, "_object_to_patch", QuantizedModelPatcher._object_to_patch_default
        )
        n._quantize_fn = getattr(
            self, "_quantize_fn", QuantizedModelPatcher._quantize_fn_default
        )
        n._lowvram = getattr(self, "_lowvram", QuantizedModelPatcher._lowvram_default)
        n._full_load = getattr(
            self, "_full_load", QuantizedModelPatcher._full_load_default
        )
        n._is_quantized = getattr(
            self, "_is_quantized", QuantizedModelPatcher._is_quantized_default
        )
        return n
