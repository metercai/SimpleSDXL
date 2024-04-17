---
language:
- en

tags:
- text2text-generation

widget:
- text: 'Expand the following prompt to add more detail:  A man in a suit walking down the street'
  example_title: Prompt Upsampling
license: mit
---

# SuperPrompt

A T5 model fine-tuned on the SuperPrompt dataset for upsampling text prompts to more detailed descriptions. Meant to be used as a pre-generation step for text-to-image models that benefit from more detailed prompts.

**Usage**

```bash
pip install transformers
```

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("roborovski/superprompt-v1", device_map="auto")

input_text = "Expand the following prompt to add more detail: A storefront with 'Text to Image' written on it."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids, max_new_tokens=77)
print(tokenizer.decode(outputs[0]))

# The neon sign above the storefront reads "NeurIPS" in bold, white letters. The storefront is surrounded by a bustling cityscape, with skyscrapers and neon signs lining the walls. The sign is surrounded by a variety of colorful goods, including a variety of fruits, vegetables, and fruits, all arranged in a neat and organized manner. The storefront is surrounded by a bustling crowd of people, all chatting and laughing as they go about their daily routines.

```

It's important that you use the exact task prefix: `Expand the following prompt to add more detail:` and limit the max tokens to 77 - which is the max prompt length for Stable Diffusion models.


## Examples

Sample prompts taken from the [DrawBench](https://imagen.research.google/) dataset.

| Prompt | Upsampled |
| --- | --- |
| A storefront with 'Hello World' written on it. | The storefront of "Hello World" is painted in vibrant shades of orange and yellow, with the words "Hello World" written in bold, cursive letters across the front. The storefront is surrounded by a bustling cityscape, with skyscrapers and neon signs lining the walls. The storefront is surrounded by a sea of people, all gazing up at the sky with a sense of excitement and excitement.
| An old photograph of a 1920s airship shaped like a pig, floating over a wheat field. | In a surreal and whimsical scene, a vintage airplane shaped like a pig glides effortlessly over a vast wheat field. The pig's eyes are closed, and its tail swishes back and forth as it floats effortlessly over the landscape. The sun casts a warm glow on the scene, illuminating the intricate details of the pig's fur and the surrounding landscape. The airship's engines hum softly, a testament to the power of imagination and imagination.
| A painting by Grant Wood of an astronaut couple, american gothic style. | In this surreal masterpiece, Grant Wood depicts a lone astronaut, dressed in a sleek silver spacesuit, gazing out at the vast expanse of space. The astronaut's eyes are closed, and his hands are clasped tightly around his body. The scene is set against a backdrop of swirling, distorted clouds, reminiscent of the American gothic style. The painting is a true masterpiece, capturing the essence of the human spirit and the beauty of the universe.
| A sheep to the right of a wine glass. | A majestic white sheep with a wagging tail stands to the right of a sparkling wine glass, her long ears twitching as she gazes intently at the glass. The sun is setting in the background, casting a warm orange glow on the scene. The scene is set in a cozy living room, with a fireplace and a wooden table in the background.