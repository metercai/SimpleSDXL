
import gc
import json
from types import MethodType
import safetensors.torch
import torch
import torch.nn as nn
import safetensors


from torch import Tensor, nn
import copy


def Flux1PartialLoad_Patch(args={}):
    model = args.get("model")

    double_blocks_cuda_size = args.get("double_blocks_cuda_size")
    single_blocks_cuda_size = args.get("single_blocks_cuda_size")

    def other_to_cpu():
        model.model.diffusion_model.img_in.to("cpu")
        model.model.diffusion_model.time_in.to("cpu")
        model.model.diffusion_model.guidance_in.to("cpu")
        model.model.diffusion_model.vector_in.to("cpu")
        model.model.diffusion_model.txt_in.to("cpu")
        model.model.diffusion_model.pe_embedder.to("cpu")

        torch.cuda.empty_cache()

    def other_to_cuda():
        model.model.diffusion_model.img_in.to("cuda")
        model.model.diffusion_model.time_in.to("cuda")
        model.model.diffusion_model.guidance_in.to("cuda")
        model.model.diffusion_model.vector_in.to("cuda")
        model.model.diffusion_model.txt_in.to("cuda")
        model.model.diffusion_model.pe_embedder.to("cuda")

    def double_blocks_to_cpu(layer_start=0, layer_size=-1):
        if layer_size == -1:
            model.model.diffusion_model.double_blocks.to("cpu")
        else:
            model.model.diffusion_model.double_blocks[layer_start:layer_start +
                                                      layer_size].to("cpu")
        torch.cuda.empty_cache()
        # gc.collect()

    def double_blocks_to_cuda(layer_start=0, layer_size=-1):
        if layer_size == -1:
            model.model.diffusion_model.double_blocks.to("cuda")
        else:
            model.model.diffusion_model.double_blocks[layer_start:layer_start +
                                                      layer_size].to("cuda")

    def single_blocks_to_cpu(layer_start=0, layer_size=-1):
        if layer_size == -1:
            model.model.diffusion_model.single_blocks.to("cpu")
        else:
            model.model.diffusion_model.single_blocks[layer_start:layer_start +
                                                      layer_size].to("cpu")
        torch.cuda.empty_cache()
        # gc.collect()

    def single_blocks_to_cuda(layer_start=0, layer_size=-1):
        if layer_size == -1:
            model.model.diffusion_model.single_blocks.to("cuda")
        else:
            model.model.diffusion_model.single_blocks[layer_start:layer_start +
                                                      layer_size].to("cuda")

    def generate_double_blocks_forward_hook(layer_start, layer_size):
        def pre_only_double_blocks_forward_hook(module, inp):

            other_to_cpu()

            if layer_start > 0:
                double_blocks_to_cpu(layer_start=0, layer_size=layer_start)

            double_blocks_to_cuda(layer_start=layer_start,
                                  layer_size=layer_size)
            # print("pre_only_double_blocks_forward_hook: ",
            #       layer_start, layer_size)
            # input("Press Enter to continue...")
            return inp
        return pre_only_double_blocks_forward_hook

    def generate_single_blocks_forward_hook(layer_start, layer_size):
        def pre_only_single_blocks_forward_hook(module, inp):
            double_blocks_to_cpu()
            if layer_start > 0:
                single_blocks_to_cpu(layer_start=0, layer_size=layer_start)

            single_blocks_to_cuda(layer_start=layer_start,
                                  layer_size=layer_size)
            # print("pre_only_single_blocks_forward_hook: ",
            #       layer_start, layer_size)
            # input("Press Enter to continue...")
            return inp
        return pre_only_single_blocks_forward_hook

    def pre_only_model_forward_hook(module, inp):
        # print("double_blocks to cpu")
        double_blocks_to_cpu()
        # print("single_blocks to cpu")
        single_blocks_to_cpu()
        # print("other to cuda")
        other_to_cuda()
        return inp

    model.model.diffusion_model.register_forward_pre_hook(
        pre_only_model_forward_hook)

    double_blocks_depth = len(model.model.diffusion_model.double_blocks)
    steps = double_blocks_cuda_size
    for i in range(0, double_blocks_depth, steps):
        s = steps
        if i + s > double_blocks_depth:
            s = double_blocks_depth - i
        model.model.diffusion_model.double_blocks[i].register_forward_pre_hook(
            generate_double_blocks_forward_hook(i, s))

    single_blocks_depth = len(model.model.diffusion_model.single_blocks)
    steps = single_blocks_cuda_size
    for i in range(0, single_blocks_depth, steps):
        s = steps
        if i + s > single_blocks_depth:
            s = single_blocks_depth - i
        model.model.diffusion_model.single_blocks[i].register_forward_pre_hook(
            generate_single_blocks_forward_hook(i, s))

    return (model,)
