# ComfyUI_AdvancedReduxControl

As many of you might noticed, the recently released Redux model is rather a model for generating multiple variants of an image, but it does not allow for changing an image based on a prompt.

If you use Redux with an Image and add a prompt, your prompt is just ignored. In general, there is no strength slider or anything in Redux to control how much the coniditioning image should determine the final outcome of your image.

For this purpose I wrote this little custom node that allows you change the strength of the Redux effect.

## Changelog

- **v2.0** This version adds the option to mask the conditioning image and to submit non-square images. The simple node hasn't changed and is backcompatible, the advanced mode is completely redesigned, so use v1 if you need backward compatibility.

## Examples

I used the following pexel image as an example conditioning image: [[https://www.pexels.com/de-de/foto/29455324/]]

![original](https://github.com/user-attachments/assets/16c8bce5-8eb3-4acf-93e9-847a81e969e0)

Lets say we want to have a similar image, but as comic/cartoon. The prompt I use is "comic, cartoon, vintage comic"

Using Redux on Flux1-dev I obtain the following image.

**original Redux setting**
![ComfyUI_00106_](https://github.com/user-attachments/assets/0c5506ef-5131-4b57-962c-ab3703881363)

As you can see, the prompt is vastly ignored. Using the custom node and "medium" setting I obtain

**Redux medium strength**
![image](https://github.com/user-attachments/assets/eb81a55a-6bdd-43ef-a8da-8d27f210c116)

Lets do the same with anime. The prompt is "anime drawing in anime style. Studio Ghibli, Makoto Shinkai."

As the anime keyword has a strong effect in Flux, we see a better prompt following on default than with comics.

**original Redux setting**
![image](https://github.com/user-attachments/assets/e5795369-2b8e-477a-974f-e0250d8689b6)

Still, its far from perfect. With "medium" setting we get an image that is much closer to anime or studio Ghibli.

**Redux medium strength**
![image](https://github.com/user-attachments/assets/b632457a-3a7e-4d99-981e-6c2682d16e2e)


You can also mix more than one images together. Here is an example with adding a second image: [[https://www.pexels.com/de-de/foto/komplizierte-bogen-der-mogul-architektur-in-jaipur-29406307/]]

Mixing both together and using the anime prompt above gives me

![image](https://github.com/user-attachments/assets/1385b22f-4497-4fdf-8255-3a15bda74a1d)

Finally, we try a very challenging prompt: "Marble statues, sculptures, stone statues. stone and marble texture. Two sculptures made out of marble stone.". As you can see, I repeated the prompt multiple times to increase its strength.
But despite the repeats, the default Redux workflow will just give us the input image Reduxed - our prompt is totally ignored.

**original Redux setting**
![ComfyUI_00108_](https://github.com/user-attachments/assets/24ad66e9-4f21-497d-8d0e-cb4778f0d1e9)

With medium we get an image back that looks more like porcelain instead of marble, but at least the two women are sculptures now.

**Redux medium strength**
![image](https://github.com/user-attachments/assets/dce4aa6f-52ab-4ef0-b027-193318895969)

Further decreasing the Redux strength will transform the woman into statues finally, but it will also further decrease their likeness to the conditioning image. In almost all my experiments, it was better to repeat multiple seeds with the "medium" setting instead of further decreasing the strength.

# Masked Conditioning Images

With **v.2** you can now also add a mask to the conditioning image.

![image](https://github.com/user-attachments/assets/71644833-1169-47b9-a843-83fd739b17c8)

In this example I just masked the flower pattern on the clothing of the right women. I then prompt for "Man walking in New York, smiling, holding a smart phone in his hand.". As you can see, his shirt adapts to the flower pattern, while nothing outside the mask has any impact on the outcoming image.

![image](https://github.com/user-attachments/assets/2163c140-4980-4738-88db-77c8286742e6)

When the masking area is very small, you have to increase the strength of the conditioning image as "less of the image" is used to condition your generation.

## Non-Square Images

Redux (or better CLIP) cannot deal with non-square images by default. It will just center-crop your conditioning image such that it has a square resolution. Now that the node supports masking, we can simply support non-square images, too. We just make the image square by adding black borders to the shorter edge. Of course, we do not want to have these borders in the generated image, so we add a mask that cover the original image but not the black padding border.

You do not have to do this yourself. There is a "keep aspect ratio" option that automatically generates the padding and adjust the mask for you.

Here is an example: the input image (again from Pexel: ) is this one: https://www.pexels.com/photo/man-wearing-blue-gray-and-black-crew-neck-shirt-1103832/

To make a point, I cropped the images to make it maximal non-square.

![image](https://github.com/user-attachments/assets/8add3187-77b3-49b2-bd9e-c82297925a8d)

With the normal workflow and the prompt "comic, vintage comic, cartoon" we would get this image back:

![image](https://github.com/user-attachments/assets/169d33e0-27db-4e4c-b5cc-ec0b6729032a)

With the "keep aspect ratio" option enabled, we get this instead:

![image](https://github.com/user-attachments/assets/10e3a8b8-9752-4060-b6b5-ed35f9764320)

Similar to masks, the conditioning effect will be weaker when we use only a small mask (or here: when the aspect ratio is extremely unbalanced). Thus, I would still recommend to avoid images with too extreme aspect ratios as this example image above.

## Usage

I was told that the images above for some reason do not contain the workflow. So I just uploaded the workflow files into the github. The **simple_workflow.json** is the workflow containing a single setting, the **advanced_workflow.json** has several customization options as well as masking and aspect ratio.

### StyleModelApplySimple

This workflow is a replacement for the ComfyUI StyleModelApply node. It has a single option that controls the influence of the conditioning image on the generation. The example images are all generated with the "medium" strength option. However, when using masking, you might have to use "strongest" or "strong" instead.


### ReduxAdvanced

This node allows for more customization. As input it gets the conditioning (prompt), the Redux style model, the CLIP vision model and optionally(!) the mask. Its parameters are:

- **downsampling_factor**: This is the most important parameter and it determines how strongly the conditioning image influences the generated image. In fact, the strength value in the StyleModelApplySimple node is just changing this single value from 1 (strongest), to 5 (weakest). The "medium" strength option is a downsampling_factor of 3. You can also choose other values up to 9.
- **downsampling_function**: These functions are the same as in any graphics program when you resize an image. The default is "area", but "bicubic" and "nearest_exact" are also interesting choices. The chosen function can very much change the outcome, so its worth experimenting a bit.
- **mode**: How the image should be cropped:
- - center crop (square) is what Redux is doing by default: cropping your image to a square image and then resizing it to 384x384 pixel
- - keep aspect ratio will add a padding to your image such that it gets square. It will adjust the mask (or generate one if no one is given) such that the padding is not part of the mask. Note that the final image is still resized to 384x384, so your image will be compressed more if you add more padding.
- - autocrop with mask will crop your image in a way that the masked area is in the center. It will also crop your image such that only the masked area and a margin remains. The margin is specified as the autocrop_margin parameter and is relative to the total size of the image. So autocropping with a margin of 0.1 means the crop is done on the masked area + 10% of the image on each side.
- **weight**: This option downscales the Redux tokens by the given value squared. This is very similar to the "conditioning average" approach many people use to reduce the effect of Redux on the generated image. This is an alternative way of reducing Redux' impact, but downsampling works better in most cases. However, feel free to also experiment with this value, or even with a combination of both: downsampling AND weight.
- **autocrop_margin** this parameter is only used when "autocrop with mask" is selected as mode

The node outputs the conditioning, as well as the cropped and resized image and its mask. You neither need the image nor the mask, they are just for debugging. Play around with the cropping option and use the "Image preview" node to see how it effects the cropped image and mask.

## Short background on Redux

Redux works in two steps. First there is a Clip Vision model that crops your input image into square aspect ratio and reduce its size to 384x384 pixels. It splits this image into 27x27 small patches and each patch is projected into CLIP space.

Redux itself is just a very small linear function that projects these clip image patches into the T5 latent space. The resulting tokens are then added to your T5 prompt.

Intuitively, Redux is translating your conditioning input image into "a prompt" that is added at the end of your own prompt.

So why is Redux dominating the final prompt? It's because the user prompt is usually very short (255 or 512 tokens). Redux, in contrast, adds 729 new tokens to your prompt. This might be 3 times as much as your original prompt. Also, the Redux prompt might contain much more information than a user written prompt that just contains the word "anime". 

So there are two solutions here: Either we shrink the strength of the Redux prompt, or we shorten the Redux prompt.

The next sections are a bit chaotic: I changed the method several times and many stuff I tried is outdated already. The only and best technique I found so far is described in **Interpolation methods**.

## Controling Redux with Token downsampling
To shrink the Redux prompt and increase the influence of the user prompt, we can use a simple trick: We take the 27x27 image patches and split them into 9x9 blocks, each containing 3x3 patches. We then merge all 3x3 tokens into one by averaging their latent embeddings. So instead of having a very long prompt with 27x27=729 tokens we now only have 9x9=81 tokens. So our newly added prompt is much smaller than the user provided prompt and, thus, have less influence on the image generation.

Downsampling is what happens when you use the "medium" setting. Of all three techniques I tried to decrease the Redux effect, downsampling worked best. ~~However, there are no further customization options. You can only downsample to 81 tokens (downsampling more is too much)~~.

## Interpolation methods

Instead of averaging over small blocks of tokens, we can use a convolution function to shrink our 27x27 images patches to an arbitrary size. There are different functions available which most of you probably know from image resizing (its the same procedure). The averaging method above is "area", but there are also other methods available such as "bicubic".  

## Controling Redux with Token merging

The idea here is to shrink the Redux prompt length by merging similar tokens together. Just think about large part of your input image contain more or less the same stuff anyways, so why having always 729 tokens? My implementation here is extremely simple and stupid and not very efficient, but anyways: I just go over all Redux tokens and merge two tokens if their cosine similarity is above a user defined threshold.

Even a threshold like 0.9 is already removing half of the Redux tokens. A threshold of 0.8 is often reducing the Redux tokens so much that they are in similar length as the user prompt.

I would start with a threshold of 0.8. If the image is blurry, increase the value a bit. If there is no effect of your prompt, decrease the threshold slightly.

## Controling Redux with Token downscaling

We can also just multiply the tokens by a certain strength value. As lower the strength, as closer the values are to zero. This is similar to prompt weighting which was quite popular for earlier stable diffusion versions, but never really worked that well for T5. Nevertheless, this technique seem to work well enough for flux.

If you use downscaling, you have to use a very low weight. You can directly start with 0.3 and go down to 0.1 if you want to improve the effect. High weights like 0.6 usually have no impact.

## Doing both or all three?

My feeling currently is that downsampling by far works best. So I would first try downsampling with 1:3 and only use the other options if the effect is too weak or too strong.
