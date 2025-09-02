# Spiking Neural Networks and Adversarial Attack Project

***

## Overview

This repository contains Python code for the paper titled "Neuromorphic Computing Paradigms Enhance Robustness through Spiking Neural Networks."

The project investigates how SNN can improve robustness using temporal processing capabilities.


This repository includes codes for:

- Development of an SNN architecture.
- Evaluation of SNN under adversarial attack.
- Analysis of SNN robustness.

## Installation

### Prerequisites

Ensure you have Python 3.8 or later installed. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Recommend Libraries

- torch >= 2.0.0+cu118
- torchvision >= 0.15.1+cu118
- numpy >= 1.24.2
- matplotlib >= 3.7.1
- tqdm >= 4.61.2
- umap-learn == 0.5.7
- gdown == 5.2.0

### You can also run with the following Libraries

- pytorch == 2.4.0 # on cpu
- torchvision == 0.19.0
- numpy >= 1.24.2
- matplotlib >= 3.7.1
- tqdm >= 4.61.2
- gdown == 5.2.0
<!-- - umap-learn == 0.5.7 -->

To install PyTorch, refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Typical install time on a "normal" desktop computer depends on the network condition!

## Project Structure

*part1* corresponds to the results in *Prioritizing task-critical temporal information during encoding boosts SNN robustness*

*part2* corresponds to the results in *Training algorithms for SNNs enabling flexible encoding achieve robust generalization*

*part3* corresponds to the results in *Early exiting SNN decoding protects against subsequent perturbations improving robustness*

*part4* corresponds to the results in *SNN employs fused input encoding to enhance performance in complex attacks*

```
|-- global_tools/           # Tools for adversarial attack and for training SNNs
|-- global_configs.py            # Script for setting dataset, and encoding methods, as well as url for model checkpoints
|-- data/                   # place to save clean datasets

|-- part1/                  # Codes evaluating the robustness of SNN with regard to encoding methods
|-- part2/                  # Codes evaluating the robustness of SNN with regard to training methods
|-- part3/                  # Codes evaluating the robustness of SNN with regard to decoding methods
|-- part4                   # Codes evaluating the scability of the proposed methods
|-- part4_attack/           # place to save attacked CIFAR-10 dataset

|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
```

## Usage

### Before you run

Go to *global_tools/global_configs* and set

```bash
dataset_path = {
    'mnist': your_data_path/mnist,
    'fashionmnist': your_data_path/fashionmnist,
    'cifar10': your_data_path/cifar10
}
```

### 1. Download checkpoints

To download pretrained checkpoints for evaluation, run

```bash
python download_all.py
```

### 2. Evaluate SNNs/ANNs in *part1*, *part2*, *part3*, *part4*

To get all data to plot the main figure corresponding to the main results, go to the corresponding directories and run:
```bash
python run_all_eval.py
```
Expected run time on a "normal" desktop computer: within 10 seconds.

### 2. Plots in supplementary materials

To get all data to plot the figures in the supplementary materials, go to the corresponding directories and run:
```bash
python run_supp_fig.py
```

## In case one want to train the model:
### 1. Obtaining the SNN/ANN Model in *part1*

To train the ANN model in *part1*, go to *part1_tools* and run

```bash
python run_ANN_train.py -dataset=MNIST -hidden=100 -bb=0 -seed=A
```

Then, to get SNN, run
```bash
python run_SNN_convert.py -dataset=MNIST -hidden=100 -pth_file=model_MNIST_seedA.pth
```

### 2. Obtaining the SNN/ANN Model in *part2*, *part3*, *part4*

To train the SNN/ANN model in *partx*, go to *partx_tools* and run

```bash
python run_train.py -config=../net_params/config_name
```

*config_name* is a placeholder for YAML files in *net_params*.

Then, to get converted SNN for *part2*, open `run_SNN_convert.py` in *part2_tools*, change seed, and then run
```bash
python run_SNN_convert.py
```

### 3. Obtaining the SNN Model with fused encoding in *part4*

Find configs that have string 'mix' in the name, such as `CIFAR10_ResNet18_ACTBP_mix_entropy-CRT.yaml`.

Then, to get SNN, go to *part4_tools* and run
```bash
python run_train-mix.py -config=../net_params/config_name
```




## License

This project is licensed under the MIT License. See `LICENSE` for details.
