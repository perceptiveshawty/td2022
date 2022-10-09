# Image Jigsaw Puzzle Solver - TAMU Datathon 2022

See [E-SSL](https://github.com/rdangovs/essl/tree/main/imagenet/simclr) for all training scripts. The backbone model (resnet50) was pre-trained for 20 epochs with a lambda coefficient of 0.7 (weight of the rotation prediction loss). The backbone weights were then frozen, and the penultimate ffn (classifier) was fine-tuned for another 80 epochs on the jigsaw puzzle dataset.

## Usage

[![Python](https://img.shields.io/badge/python-3.7.4-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-374/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.12.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)

### Requirements
* Python 3.7.4

### Packages and model weights
```
pip install -r requirements.txt
```
Tested with venv. Download the pretrained weights [here](https://drive.google.com/file/d/1FkwXMzi5SZE5Rz_M1MWzMgrxcQlnHTPt/view?usp=sharing).

### Inference
```
python submission.py 
```
