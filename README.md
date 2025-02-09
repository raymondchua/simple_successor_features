# Learning Successor Features the Simple Way 
This is the official code for our paper, "Learning Successor Features the Simple Way," accepted at NeurIPS 2024. 

The authors are Raymond Chua, Arna Ghosh, Christos Kaplanis, Blake Richards and Doina Precup. 

## This repository is a work in progress. More details on the repo will be added soon. (Last updated: 4 Nov 2024)

## TLDR: A simple and elegant approach to learning Successor Features 🌟

This repository contains the code for the experiments in the paper. The code is written in PyTorch and is adapted from
the <a href='https://github.com/rll-research/url_benchmark'>Unsupervised Reinforcement Learning Benchmark (URLB) repository.</a> 

## Introduction
In the paper, we presented the architecture for the discrete action setting. 
Here, we provide the code for the continuous action setting, which requires some modifications to the architecture. 
The figure below shows the architecture for the continuous action setting.



## Structure
***
The repository is structured as follows:

| Folder           |          Description          |
|:-----------------|:-----------------------------:|
| agent            | Implementations of the agents | 
| custom_dmc_tasks |             Tasks             |



## Citations
***
If you find this repository useful in your research, please consider citing our paper:

```bibtex
@inproceedings{NEURIPS2024_597254dc,
 author = {Chua, Raymond and Ghosh, Arna and Kaplanis, Christos and Richards, Blake and Precup, Doina},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {49957--50030},
 publisher = {Curran Associates, Inc.},
 title = {Learning Successor Features the Simple Way},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/597254dc45be8c166d3ccf0ba2d56325-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```




