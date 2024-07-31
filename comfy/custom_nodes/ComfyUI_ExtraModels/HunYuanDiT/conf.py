"""
List of all HYDiT model types / settings
"""
from argparse import Namespace
hydit_args = Namespace(**{ # normally from argparse
	"infer_mode": "torch",
	"norm": "layer",
	"learn_sigma": True,
	"text_states_dim": 1024,
	"text_states_dim_t5": 2048,
	"text_len": 77,
	"text_len_t5": 256,
})

hydit_conf = {
	"G/2": { # Seems to be the main one
		"unet_config": {
			"depth"       :   40,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1408,
			"mlp_ratio" : 4.3637,
			"input_size": (1024//8, 1024//8),
			"args": hydit_args,
		},
		"sampling_settings" : {
			"beta_schedule" : "linear",
			"linear_start"  : 0.00085,
			"linear_end"    : 0.03,
			"timesteps"     : 1000,
		},
	},
	"G/2-1.2": {
		"unet_config": {
			"depth"       :   40,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1408,
			"mlp_ratio" : 4.3637,
			"input_size": (1024//8, 1024//8),
			"cond_style": False,
            "cond_res"  : False,
			"args": hydit_args,
		},
		"sampling_settings" : {
			"beta_schedule" : "linear",
			"linear_start"  : 0.00085,
			"linear_end"    : 0.018,
			"timesteps"     : 1000,
		},
	}
}

# these are the same as regular DiT, I think
from ..DiT.conf import dit_conf
for name in ["XL/2", "L/2", "B/2"]:
	hydit_conf[name] = {
		"unet_config": dit_conf[name]["unet_config"].copy(),
		"sampling_settings": hydit_conf["G/2"]["sampling_settings"],
	}
	hydit_conf[name]["unet_config"]["args"] = hydit_args
