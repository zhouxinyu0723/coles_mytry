##  Contrastive Laplacian Eigenmaps (COLES) 

### this is cloned from https://github.com/allenhaozhu/COLES.git which is the code for the paper written by Hao Zhu, Ke Sun, Piotr Koniusz
I ,as a reader, tried to implement the coles_gcn training in the file Untitled.ipynb and Untitled1.ipynb. The model in Untitled.ipynb was created guided by the paper. The model in Untitled1.ipynb is from orginial code, with a loss function which I do not quite understand.

### Overview
This repo contains an example implementation of the Neurips 2021 paper: Contrastive Laplacian Eigenmaps.
This code is based on SSGC (Simple Spectral Graph Convolution).
In this code, we provide codes for Table 2 (contrastive classification) and node clustering experiments (Table 7). To prevent unnecessary issues, we submit the data along with the code.
For reddit and ogb-arxiv, we do not provide the code because the corresponding datasets are too big.
We also provide the log file for comparisons to help if the code cannot run correctly in reviewers' environment (any unknown issues with packages etc.)

This home repo contains the implementation for citation networks (Cora, Citeseer, and Pubmed, Cora Full).


### Dependencies
Our implementation works with PyTorch>=1.0.0

### Data
We provide the citation network datasets under `data/`.

### Usage

```
$ python train_ssgc_(dataset_name).py
$ python train_ssgc_(dataset_name)_clustering.py
```
