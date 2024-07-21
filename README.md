# peerRTF: Robust MVDR Beamforming Using Graph Convolutional Network

<div align="center">

[Paper](https://arxiv.org/abs/2407.01779) |
[Project Page](https://peerrtf.github.io/) |
[Introduction](#introduction) |
[Training](#training) |
[Evaluation](#evaluation) |
[Citation](#citation)

</div>

![](https://github.com/levidaniel96/peerRTF/blob/main/Block_diagram.png)
## Introduction

This repository contains the implementation of the method introduced in our paper, which presents a novel approach for accurately and robustly estimating Relative Transfer Functions (RTFs). Accurate RTF estimation is crucial for designing effective microphone array beamformers, particularly in challenging noisy and reverberant environments.


## Overview

The proposed method leverages prior knowledge of the acoustic environment to enhance the robustness of RTF estimation by learning the RTF manifold. The key innovation in this work is the use of a Graph Convolutional Network (GCN) to learn and infer a robust representation of the RTFs within a confined area. This approach significantly improves the performance of beamformers by providing more reliable RTF estimation in complex acoustic settings.
 
## Installation
 to create the environment, run the following command:
```bash

conda create --name your_environment_name --file requirements.txt
```

## Training

for training, run the following command:

```bash
python main.py
```
Note that you need to collect the data and put it in the data folder. You can estimate the RTFs using the code provided in this repository.

the data should be in the following format:
```bash
data
├── train
│   ── noisy graphs
│      ├── graph_data_1.pt
│      ├── graph_data_2.pt
│      └── ...
│── val
│   ── noisy graphs
│      ├── graph_data_1.pt
│      ├── graph_data_2.pt
│      └── ...
└── test
    ── noisy graphs
       ├── graph_data_1.pt
       ├── graph_data_2.pt
       └── ...
```
each graph_data should contain the following:

```bash
{
    'graph': graph, # the graph
    'edge_index': edge_index, # the edge index of the graph
    'RTF': RTF, # the RTF of the noisy signal are the nodes of the graph
    'clean': clean, # the target RTF(Oracle)
    'y': noisy data, # the noisy signal(M channels)
    'x': clean data, # the clean signal(M channels)
    'n': noise data, # the noise signal(M channels)
    'index': index, # the index of the node in the graph

}

```
The code will create a model, train it, and save it in the models folder.

## Evaluation

For evaluation, run the following command:


```bash
cd evaluation 
python evaluation.py
```
During the evaluation, the code creates new noisy examples and estimates the RTFs using the GEVD. These estimated RTFs are then connected to the graphs using the KNN algorithm. The trained model is used to estimate the robust RTFs. Finally, the estimated RTFs are used to estimate the speech signal using the MVDR beamformer. The SNR, STOI, ESTOI, and DNSMOS scores for the estimated speech signal are then calculated.

example of an output:
```bash
SNR in: -6.00
SNR out GEVD: 17.48
SNR out peerRTF: 19.30
STOI in: 31.82
STOI out GEVD: 59.85
STOI out peerRTF: 59.70
ESTOI in: 20.52
ESTOI out GEVD: 43.14
ESTOI out peerRTF: 43.68

DNSMOS results:
referance signal 3.08
noisy signal 2.25
peerRTF 2.55
GEVD 2.44
```
## Citation
If you use this code in your research, please cite our paper:

```bash

@article{peerRTF,
  title={peerRTF: Robust MVDR Beamforming Using Graph Convolutional Network },
  author={Amit Sofer, Daniel Levi, Sharon Gannot},
  journal=arXiv preprint arXiv:2407.01779},
  year={2024},
}
```
