import contextlib
import unittest

import torch


# wildcard trick is taken from pythongossss's
class AnyType(str):

    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


def get_weight_dtype_inputs():
    return {
        "weight_dtype": (
            [
                "default",
                "float32",
                "float64",
                "bfloat16",
                "float16",
                "fp8_e4m3fn",
                "fp8_e4m3fn_fast",
                "fp8_e5m2",
            ],
        ),
    }


def parse_weight_dtype(model_options, weight_dtype):
    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e4m3fn_fast": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }.get(weight_dtype, None)
    if dtype is not None:
        model_options["dtype"] = dtype
    if weight_dtype == "fp8_e4m3fn_fast":
        model_options["fp8_optimizations"] = True
    return model_options


@contextlib.contextmanager
def disable_load_models_gpu():
    def foo(*args, **kwargs):
        pass

    from comfy import model_management

    with unittest.mock.patch.object(model_management, "load_models_gpu", foo):
        yield


def patch_optimized_module():
    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:
        return

    if getattr(OptimizedModule, "_patched", False):
        return

    def __getattribute__(self, name):
        if name == "_orig_mod":
            return object.__getattribute__(self, "_modules")[name]
        if name in (
            "__class__",
            "_modules",
            "state_dict",
            "load_state_dict",
            "parameters",
            "named_parameters",
            "buffers",
            "named_buffers",
            "children",
            "named_children",
            "modules",
            "named_modules",
        ):
            return getattr(object.__getattribute__(self, "_orig_mod"), name)
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        # unload_lora_weights() wants to del peft_config
        return delattr(self._orig_mod, name)

    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, OptimizedModule) or issubclass(
            object.__getattribute__(instance, "__class__"), cls
        )

    OptimizedModule.__getattribute__ = __getattribute__
    OptimizedModule.__delattr__ = __delattr__
    OptimizedModule.__instancecheck__ = __instancecheck__
    OptimizedModule._patched = True
