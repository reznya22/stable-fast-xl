# Stable-fast-xl
<div>
    <a href='https://huggingface.co/artemtumch/stable-fast-xl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFaceModel-StableFastXL-blue'></a>&ensp;
    <a href='https://github.com/chengzeyi/stable-fast'><img src='https://img.shields.io/badge/StableFast-262826?logo=github'></a>&ensp;
    <a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFaceModel-SDXL-blue'></a>&ensp;
</div>

Stable-fast is an ultra lightweight inference optimization framework for HuggingFace Diffusers on NVIDIA GPUs. stable-fast provides super fast inference optimization by utilizing some key techniques.
this repository contains a compact installation of the stable-fast compiler https://github.com/chengzeyi/stable-fast and its inference with the stable-diffusion-xl-base-1.0
Inference with [stable-diffusion-xl-base-1.0)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)

![image.png](https://cdn-uploads.huggingface.co/production/uploads/670503434c094132b2282e63/Xib4SHo9PX7-oSWP3Or3Y.png)

![image.png](https://cdn-uploads.huggingface.co/production/uploads/670503434c094132b2282e63/-a7V70NkS09TeMSZAKgVB.png)

# iinference sdxl model 30%+ faster!!!

## Differences With Other Acceleration Libraries
#### Fast:
stable-fast is specialy optimized for HuggingFace Diffusers. It achieves a high performance across many libraries. And it provides a very fast compilation speed within only a few seconds. It is significantly faster than **torch.compile**, **TensorRT** and **AITemplate** in compilation time.
#### Minimal:
stable-fast works as a plugin framework for **PyTorch**. It utilizes existing PyTorch functionality and infrastructures and is compatible with other acceleration techniques, as well as popular fine-tuning techniques and deployment solutions.


# How to use

### Install dependencies
```bash
pip install diffusers, transformers, safetensors, accelerate, sentencepiece
```

### Download repository and run script for stable-fast installation
```bash
git clone https://huggingface.co/artemtumch/stable-fast-xl
cd stable-fast-xl
```
open **install_stable-fast.sh** file and change cp311 for your python version in this line

pip install -q https://github.com/chengzeyi/stable-fast/releases/download/v0.0.15/stable_fast-0.0.15+torch210cu118-cp311-cp311-manylinux2014_x86_64.whl

where **cp311** -> for **python 3.11**  **|** **cp38** -> for **python3.8**

then run script
```bash
sh install_stable-fast.sh
```

## Generate image
```py
from diffusers import DiffusionPipeline
import torch

from sfast.compilers.stable_diffusion_pipeline_compiler import (
compile, CompilationConfig
)

import xformers
import triton

pipe = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16,
use_safetensors=True,
variant="fp16"
)

# enable to reduce GPU VRAM usage (~30%)
# pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)

pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

config = CompilationConfig.Default()

config.enable_xformers = True
config.enable_triton = True
config.enable_cuda_graph = True

pipe = compile(pipe, config)

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]
```

## Inpainting
```py
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch

from sfast.compilers.stable_diffusion_pipeline_compiler import (
compile, CompilationConfig
)

import xformers
import triton

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
"diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
torch_dtype=torch.float16,
variant="fp16"
)

# enable to reduce GPU VRAM usage (~30%)
# pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)

pipe.to("cuda")

config = CompilationConfig.Default()

config.enable_xformers = True
config.enable_triton = True
config.enable_cuda_graph = True

pipe = compile(pipe, config)

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a tiger sitting on a park bench"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
prompt=prompt,
image=image,
mask_image=mask_image,
guidance_scale=8.0,
num_inference_steps=20, # steps between 15 and 30 work well
strength=0.99, # make sure to use `strength` below 1.0
generator=generator,
).images[0]

```
