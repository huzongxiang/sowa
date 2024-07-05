<div align="center">

# Soldier-Offier Window self-Attention (SOWA)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

We introduce a window self-attention mechanism based on the CLIP model, combined with learnable prompts to process multi-level features within a Soldier-Offier Window self-Attention (SOWA) framework.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/huzongxiang/sowa
cd sowa

# [OPTIONAL] create conda environment
conda create -n sowa python=3.9
conda activate sowa

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/huzongxiang/sowa
cd sowa

# create conda environment and install dependencies
conda env create -f environment.yaml -n sowa

# activate conda environment
conda activate sowa
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu data=sowa_visa model=sowa_hfwa

# train on GPU
python src/train.py trainer=gpu data=sowa_visa model=sowa_hfwa
```