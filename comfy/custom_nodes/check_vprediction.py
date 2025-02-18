import comfy
from comfy.model_base import ModelType

class CheckVpredictionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "check_mode": (["any_vpred", "specific_type"], {"default": "any_vpred"}),
                "specific_type": (["V_PREDICTION", "V_PREDICTION_EDM", "V_PREDICTION_CONTINUOUS"], 
                                {"default": "V_PREDICTION"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_match", "model_type")
    FUNCTION = "check_vpred"
    CATEGORY = "custom/model_check"

    def check_vpred(self, model, check_mode, specific_type):
        model_type = model.model.model_type
        
        if callable(model_type):
            try:
                model_type = model_type()
            except TypeError:
                state_dict = model.model.state_dict()
                model_type = model_type(state_dict)

        if isinstance(model_type, ModelType):
            type_name = model_type.name
        elif hasattr(model_type, "name"):
            type_name = model_type.name
        else:
            type_name = str(model_type)
        
        v_pred_types = {
            ModelType.V_PREDICTION,
            ModelType.V_PREDICTION_EDM,
            ModelType.V_PREDICTION_CONTINUOUS
        }
        
        if check_mode == "any_vpred":
            is_match = model_type in v_pred_types
        else:
            try:
                target_type = ModelType[specific_type]
                is_match = (model_type == target_type)
            except KeyError:
                is_match = False
                type_name = f"Unknown type: {type_name}"
            
        return (is_match, type_name)

NODE_CLASS_MAPPINGS = {
    "CheckVpredictionNode": CheckVpredictionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckVpredictionNode": "Advanced V-Prediction Checker"
}