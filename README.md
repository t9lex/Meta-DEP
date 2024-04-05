# Meta_DES

## Introduction
Meta_DES is an interpretable deep learning-based path reasoning framework that builds a deep learning model that can be used for drug efficacy prediction based on drug-protein-disease heterogeneous networks.

## Requirements
see environment.yaml

## Installation
To install the required packages for running Meta_DES, please use the following command first. If you meet any problems when installing pytorch, please refer to [pytorch official website](https://pytorch.org/)
```bash
conda env create -f environment.yml
```

## Train
```bash
python -m Train.Train.py
```
