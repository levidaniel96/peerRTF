# peerRTF: Robust MVDR Beamforming Using Graph Convolutional Network

![](https://github.com/levidaniel96/peerRTF/blob/main/Block_diagram.png)
## Description

This repository contains the code for the paper "peerRTF: Robust MVDR Beamforming Using Graph Convolutional Network"

arxiv link:(https://arxiv.org/abs/2407.01779)


## Installation
 to create the environment, run the following command:
```bash

conda create --name your_environment_name --file requirements.txt
```

## Usage

for training, run the following command:

```bash
python main.py
```
note that you need to collect the data and put it in the data folder. 
you can estimate the RTFs using the code that provided in: ### add sharon lab git link
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
    'y': noisy data, # the noisy signal(M channel)
    'x': clean data, # the clean signal(M channel)
    'n': noise data, # the noise signal(M channel)
    'index': index, # the index of the node in the graph

}

```

the code will create a model, train him and save it in the models folder.

for evaluation, run the following command:

```bash
cd evaluation 
python evaluation.py
```
in the evaluation the code create new noisy examples and estimate the RTFs using the GEVD. then we connect the estimated RTFs to the graphs by KNN algorithm. then we use the trained model to estimate the robust RTFs.
after that, we use the estimated RTFs to estimate the speech signal using the MVDR beamformer.
and finally, we calculate the SNR, STOI,ESTOI and DNSMOS scores for the estimated speech signal.

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
