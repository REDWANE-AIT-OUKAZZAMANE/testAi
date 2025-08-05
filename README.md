---
license: apache-2.0
language:
- en
- zh
library_name: diffusers
pipeline_tag: text-to-image
---
<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p>
<p align="center">
          💜 <a href="https://chat.qwen.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/qwen-image">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1600"/>
<p>

## Introduction
We are thrilled to release **Qwen-Image**, an image generation foundation model in the Qwen series that achieves significant advances in **complex text rendering** and **precise image editing**. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/bench.png#center)

## News
- 2025.08.04: We released the [Technical Report](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf) of Qwen-Image!
- 2025.08.04: We released Qwen-Image weights! Check at [huggingface](https://huggingface.co/Qwen/Qwen-Image) and [Modelscope](https://modelscope.cn/models/Qwen/Qwen-Image)!
- 2025.08.04: We released Qwen-Image! Check our [blog](https://qwenlm.github.io/blog/qwen-image) for more details!


## Quick Start

Install the latest version of diffusers
```
pip install git+https://github.com/huggingface/diffusers
```

The following contains a code snippet illustrating how to use the model to generate images based on text prompts:

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = [
    "en": "Ultra HD, 4K, cinematic composition." # for english prompt,
    "zh": "超清，4K，电影级构图" # for chinese prompt,
]

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

negative_prompt = " "


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")
```

## Running with quantization (24GB and 16GB GPUs)

If you're working with a resource-constained environment, consider applying quantization. Below, we provide two snippets of code using different quantization schemes for 24GB and 16GB GPUs.

### 24GB VRAM GPU and `torchao`

```py
# make sure torchao is installed: `pip install -U torchao
import torch

from diffusers import AutoModel, DiffusionPipeline, TorchAoConfig


model_id = "Qwen/Qwen-Image"
torch_dtype = torch.bfloat16
device = "cuda"

quantization_config = TorchAoConfig("int8wo")

transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
pipe = DiffusionPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch_dtype)
pipe.enable_model_cpu_offload()

prompt = "a woman and a man sitting at a cafe, the woman has red hair and she's wearing purple sweater with a black scarf and a white hat, the man is sitting on the other side of the table and he's wearing a white shirt with a purple scarf and red hat, both of them are sipping their coffee while in the table there's some cake slices on their respective plates, each with forks and knives at each side."
negative_prompt = ""

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1664,
    height=928,
    num_inference_steps=25,
    true_cfg_scale=4.0,
    generator=generator,
).images[0]

image.save("qwen_torchao.png")
```

### 16GB VRAM GPU and `bitsandbytes`

```py
# make sure bitsandbytes is installed: `pip install -U bitsandbytes
import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel


model_id = "Qwen/Qwen-Image"
torch_dtype = torch.bfloat16
device = "cuda"

quantization_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)

transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
transformer = transformer.to("cpu")

quantization_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
text_encoder = text_encoder.to("cpu")

pipe = QwenImagePipeline.from_pretrained(
    model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload()

prompt = "a woman and a man sitting at a cafe, the woman has red hair and she's wearing purple sweater with a black scarf and a white hat, the man is sitting on the other side of the table and he's wearing a white shirt with a purple scarf and red hat, both of them are sipping their coffee while in the table there's some cake slices on their respective plates, each with forks and knives at each side."
negative_prompt = ""

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1664,
    height=928,
    num_inference_steps=25,
    true_cfg_scale=4.0,
    generator=generator,
).images[0]

image.save("qwen_bnb.png")
```

## Show Cases

One of its standout capabilities is high-fidelity text rendering across diverse images. Whether it’s alphabetic languages like English or logographic scripts like Chinese, Qwen-Image preserves typographic details, layout coherence, and contextual harmony with stunning accuracy. Text isn’t just overlaid—it’s seamlessly integrated into the visual fabric.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center)

Beyond text, Qwen-Image excels at general image generation with support for a wide range of artistic styles. From photorealistic scenes to impressionist paintings, from anime aesthetics to minimalist design, the model adapts fluidly to creative prompts, making it a versatile tool for artists, designers, and storytellers.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center)

When it comes to image editing, Qwen-Image goes far beyond simple adjustments. It enables advanced operations such as style transfer, object insertion or removal, detail enhancement, text editing within images, and even human pose manipulation—all with intuitive input and coherent output. This level of control brings professional-grade editing within reach of everyday users.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center)

But Qwen-Image doesn’t just create or edit—it understands. It supports a suite of image understanding tasks, including object detection, semantic segmentation, depth and edge (Canny) estimation, novel view synthesis, and super-resolution. These capabilities, while technically distinct, can all be seen as specialized forms of intelligent image editing, powered by deep visual comprehension.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center)

Together, these features make Qwen-Image not just a tool for generating pretty pictures, but a comprehensive foundation model for intelligent visual creation and manipulation—where language, layout, and imagery converge.


## License Agreement

Qwen-Image is licensed under Apache 2.0. 

## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@article{qwen-image,
    title={Qwen-Image Technical Report}, 
    author={Qwen Team},
    journal={arXiv preprint},
    year={2025}
}
```