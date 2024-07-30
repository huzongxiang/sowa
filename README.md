<div align="center">

# Soldier-Offier Window self-Attention (SOWA)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2407.03634-B31B1B.svg)](https://arxiv.org/abs/2407.03634)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

<div align="center">
  <img src="https://github.com/huzongxiang/sowa/blob/resources/fig1.png" alt="concept" style="width: 50%;">
</div>

Visual anomaly detection is critical in industrial manufacturing, but traditional methods often rely on extensive
normal datasets and custom models, limiting scalability.
Recent advancements in large-scale visual-language models have significantly improved zero/few-shot anomaly detection. However, these approaches may not fully utilize hierarchical features, potentially missing nuanced details. We
introduce a window self-attention mechanism based on the
CLIP model, combined with learnable prompts to process
multi-level features within a Soldier-Offier Window selfAttention (SOWA) framework. Our method has been tested
on five benchmark datasets, demonstrating superior performance by leading in 18 out of 20 metrics compared to existing state-of-the-art techniques.

![architecture](https://github.com/huzongxiang/sowa/blob/resources/fig2.png)

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

## Results

Comparisons with few-shot (K=4) anomaly detection methods on datasets of MVTec-AD, Visa, BTAD, DAGM and DTD Synthetic. 
| Metric    | Dataset        | WinCLIP     | VAND        | Ours        |
|-----------|----------------|-------------|-------------|-------------|
| AC AUROC  | MVTec-AD       | 95.2±1.3    | 92.8±0.2    | 96.8±0.3    |
|           | Visa           | 87.3±1.8    | 92.6±0.4    | 92.9±0.2    |
|           | BTAD           | 87.0±0.2    | 92.1±0.2    | 94.8±0.2    |
|           | DAGM           | 93.8±0.2    | 96.2±1.1    | 98.9±0.3    |
|           | DTD-Synthetic  | 98.1±0.2    | 98.5±0.1    | 99.1±0.0    |
| AC AP     | MVTec-AD       | 97.3±0.6    | 96.3±0.1    | 98.3±0.3    |
|           | Visa           | 88.8±1.8    | 94.5±0.3    | 94.5±0.2    |
|           | BTAD           | 86.8±0.0    | 95.2±0.5    | 95.5±0.7    |
|           | DAGM           | 83.8±1.1    | 86.7±4.5    | 95.2±1.7    |
|           | DTD-Synthetic  | 99.1±0.1    | 99.4±0.0    | 99.6±0.0    |
| AS AUROC  | MVTec-AD       | 96.2±0.3    | 95.9±0.0    | 95.7±0.1    |
|           | Visa           | 97.2±0.2    | 96.2±0.0    | 97.1±0.0    |
|           | BTAD           | 95.8±0.0    | 94.4±0.1    | 97.1±0.0    |
|           | DAGM           | 93.8±0.1    | 88.9±0.4    | 96.9±0.0    |
|           | DTD-Synthetic  | 96.8±0.2    | 96.7±0.0    | 98.7±0.0    |
| AS AUPRO  | MVTec-AD       | 89.0±0.8    | 91.8±0.1    | 92.4±0.2    |
|           | Visa           | 87.6±0.9    | 90.2±0.1    | 91.4±0.0    |
|           | BTAD           | 66.6±0.2    | 78.2±0.1    | 81.2±0.2    |
|           | DAGM           | 82.4±0.3    | 77.8±0.9    | 94.4±0.1    |
|           | DTD-Synthetic  | 90.1±0.5    | 92.2±0.0    | 96.6±0.1    |



Performance Comparison on MVTec-AD and Visa Datasets. 
| Method        | Source                  | MVTec-AD AC AUROC | MVTec-AD AS AUROC | MVTec-AD AS PRO | Visa AC AUROC | Visa AS AUROC | Visa AS PRO |
|---------------|-------------------------|-------------------|-------------------|-----------------|---------------|---------------|-------------|
| SPADE         | arXiv 2020              | 84.8±2.5          | 92.7±0.3          | 87.0±0.5        | 81.7±3.4      | 96.6±0.3      | 87.3±0.8    |
| PaDiM         | ICPR 2021               | 80.4±2.4          | 92.6±0.7          | 81.3±1.9        | 72.8±2.9      | 93.2±0.5      | 72.6±1.9    |
| PatchCore     | CVPR 2022               | 88.8±2.6          | 94.3±0.5          | 84.3±1.6        | 85.3±2.1      | 96.8±0.3      | 84.9±1.4    |
| WinCLIP       | CVPR 2023               | 95.2±1.3          | 96.2±0.3          | 89.0±0.8        | 87.3±1.8      | 97.2±0.2      | 87.6±0.9    |
| April-GAN     | CVPR 2023 VAND workshop | 92.8±0.2          | 95.9±0.0          | 91.8±0.1        | 92.6±0.4      | 96.2±0.0      | 90.2±0.1    |
| PromptAD      | CVPR 2024               | 96.6±0.9          | 96.5±0.2          | -               | 89.1±1.7      | 97.4±0.3      | -           |
| InCTRL        | CVPR 2024               | 94.5±1.8          | -                 | -               | 87.7±1.9      | -             | -           |
| HFWA          | Ours                    | 96.8±0.3          | 95.7±0.1          | 92.4±0.2        | 92.9±0.2      | 97.1±0.0      | 91.4±0.0    |


Comparisons with few-shot anomaly detection methods on datasets of MVTec-AD, Visa, BTAD, DAGM and DTD Synthetic.
![few-shot](https://github.com/huzongxiang/sowa/blob/resources/fig5.png)

## Visualization
Visualization results under the few-shot setting (K=4).
![visualize](https://github.com/huzongxiang/sowa/blob/resources/fig6.png)

## Mechanism
Hierarchical Results on MVTec-AD Dataset. A set of images showing the real outputs of the model, illustrating how different layers (H1 to H4) process various feature modes. Each row represents a different sample, with columns showing the original image, segmentation mask, heatmap, and feature outputs from H1 to H4, and fusion.
![mechanism](https://github.com/huzongxiang/sowa/blob/resources/fig7.png)

## Inference Speed
Inference performance comparison of different methods on a single NVIDIA RTX3070 8GB GPU.
![mechanism](https://github.com/huzongxiang/sowa/blob/resources/fig9.png)