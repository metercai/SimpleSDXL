# ComfyUI Noise

This repo contains 6 nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows for more control and flexibility over the noise. This allows e.g. for workflows with small variations to generations or finding the accompanying noise to some input image and prompt.

## Nodes

### Noisy Latent Image:
This node lets you generate noise, you can find this node under `latent>noise` and it the following settings:
- **source**: where to generate the noise, currently supports GPU and CPU.
- **seed**: the noise seed.
- **width**: image width.
- **height**: image height.
- **batch_size**: batch size.

### Duplicate Batch Index:
The functionality of this node has been moved to core, please use: `Latent>Batch>Repeat Latent Batch` and `Latent>Batch>Latent From Batch` instead. 

This node lets you duplicate a certain sample in the batch, this can be used to duplicate e.g. encoded images but also noise generated from the node listed above. You can find this node under `latent` and it has the following settings:
- **latents**: the latents.
- **batch_index**: which sample in the latents to duplicate.
- **batch_size**: the new batch size, (i.e. how many times to duplicate the sample).

### Slerp Latents:
This node lets you mix two latents together. Both of the input latents must share the same dimensions or the node will ignore the mix factor and instead output the top slot. When it comes to other things attached to the latents such as e.g. masks, only those of the top slot are passed on. You can find this node under `latent` and it comes with the following inputs:
- **latents1**: first batch of latents.
- **latents2**: second batch of latents. This input is optional.
- **mask**: determines where in the latents to slerp. This input is optional
- **factor**: how much of the second batch of latents should be slerped into the first.

### Get Sigma:
This node can be used to calculate the amount of noise a sampler expects when it starts denoising. You can find this node under `latent>noise` and it comes with the following inputs and settings:
- **model**: The model for which to calculate the sigma.
- **sampler_name**: the name of the sampler for which to calculate the sigma.
- **scheduler**: the type of schedule used in the sampler
- **steps**: the total number of steps in the schedule
- **start_at_step**: the start step of the sampler, i.e. how much noise it expects in the input image
- **end_at_step**: the current end step of the previous sampler, i.e. how much noise already is in the image.

Most of the time you'd simply want to keep `start_at_step` at zero, and `end_at_step` at `steps`, but if you'd want to re-inject some noise in between two samplers, e.g. one sampler that denoises from 0 to 15, and a second that denoises from 10 to 20, you'd want to use a `start_at_step` 10 and an `end_at_step` of 15. So that the image we get, which is at step 15, can be noised back down to step 10, so the second sampler can bring it to 20. Take note that the Advanced Ksampler has a settings for `add_noise` and `return_with_leftover_noise` which when working with these nodes we both want to have disabled.

### Inject Noise:
This node lets you actually inject the noise into an image latent, you can find this node under `latent>noise` and it comes with the following inputs:
- **latents**: The latents to inject the noise into.
- **noise**: The noise. This input is optional
- **mask**: determines where to inject noise. This input is optional
- **strength**: The strength of the noise. Note that we can use the node above to calculate for us an appropriate strength value.

### Unsampler:
This node does the reverse of a sampler. It calculates the noise that would generate the image given the model and the prompt. You can find this node under `sampling` and it takes the following inputs and settings:
- **model**: The model to target.
- **steps**: number of steps to noise.
- **end_step**: to what step to travel back to.
- **cfg**: classifier free guidance scale.
- **sampler_name**: The name of the sampling technique to use.
- **scheduler**: The type of schedule to use.
- **normalize**: whether to normalize the noise before output. Useful when passing it on to an Inject Noise node which expects normalizes noise.
- **positive**: Positive prompt.
- **negative**: Negative prompt.
- **latent_image**: The image to renoise.

When trying to reconstruct the target image as faithful as possible this works best if both the unsampler and sampler use a cfg scale close to 1.0 and similar number of steps. But it is fun and worth it to play around with these settings to get a better intuition of the results. This node let's you do similar things the A1111 [img2img alternative](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#img2img-alternative-test) script does

## Examples

here are some examples that show how to use the nodes above. Workflows to these examples can be found in the `example_workflow` folder.

<details>
<summary>
generating variations
</summary>

![screenshot of a workflow that demos generating small variations to a given seed](https://github.com/BlenderNeko/ComfyUI_noise/blob/master/examples/example_variation.png)

To create small variations to a given generation we can do the following: We generate the noise of the seed that we're interested using a `Noisy Latent Image` node, we then create an entire batch of these with a `Duplicate Batch Index` node. Note that if we were doing this for img2img we can use this same node to duplicate the image latents. Next we generate some more noise, but this time we generate a batch of noise rather than a single sample. We then Slerp this newly created noise into the other one with a `Slerp Latents` node. To figure out the required strength for injecting this noise we use a `Get Sigma` node. And finally we inject the slerped noise into a batch of empty latents with a `Inject Noise` node. Take note that we use an advanced Ksampler with the `add_noise` setting disabled

</details>

<details>
<summary>
"unsampling"
</summary>

![screenshot of a workflow that demos generating small variations to a given seed](https://github.com/BlenderNeko/ComfyUI_noise/blob/master/examples/example_unsample.png)

To get the noise that recreates a certain image, we first load an image. Then we use the `Unsampler` node with a low cfg value. To check if this is working we then take the resulting noise and feed it back into an advanced ksampler with the `add_noise` setting disabled, and a cfg of 1.0.

</details>

