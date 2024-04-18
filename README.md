# Meta_DEP

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

## Use
### Example Command:
```bash
python tool.py txtname.txt
```
### input
`txtname.txt :`  File of disease protein list. We recommend that the number of processed proteins in step one must be greater than 20 (this is a necessary requirement to make the calculation meaningful), and preferably less than 100 (this is a recommendation to reduce calculation time).


