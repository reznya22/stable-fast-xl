#!/bin/bash

pip install  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 torchdata==0.7.0 torchtext==0.16.0 triton==2.1.0 xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118

pip install fastai==2.7.13

pip install -q https://github.com/chengzeyi/stable-fast/releases/download/v0.0.15/stable_fast-0.0.15+torch210cu118-cp311-cp311-manylinux2014_x86_64.whl


#change cp311 for your version where cp311 -> for python 3.11 version  | cp38 -> for python3.8 version


wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O libtcmalloc_minimal.so.4

export LD_PRELOAD=libtcmalloc_minimal.so.4

env LD_PRELOAD=libtcmalloc_minimal.so.4

ldconfig /usr/lib64-nvidia
