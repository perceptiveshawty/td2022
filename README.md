# Image Jigsaw Puzzle Solver - TAMU Datathon 2022

See [E-SSL](https://github.com/rdangovs/essl/tree/main/imagenet/simclr) for all training scripts. The backbone model (resnet50) was pre-trained for 20 epochs with a lambda coefficient of 0.7 (weight of the rotation prediction loss). The backbone weights were then frozen, and the penultimate ffn was fine-tuned for another 80 epochs on the jigsaw puzzle dataset.

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
## Training Stats

### Pre-training task
{"epoch": 18, "step": 14670, "learning_rate": 0.15161152976772346, "loss": 2.669699192047119, "acc": 0.734375, "time": 1289}

### Jigsaw solving task (fine-tuning linear classifier)
{"epoch": 19, "acc1": 65.85144927536231, "acc5": 95.51127214170693, "best_acc1": 65.85144927536231, "best_acc5": 95.51127214170693}

Note that epochs were not recorded correctly for the jigsaw solving task, so '19' is not correct. For the pre-training task, accuracy refers to single-fold rotation prediction; Jigsaw solver, classification on a held out test set.

## Credit

If this work was interesting, please check out and/or cite [Equivariant Contrastive Learning](http://super-ms.mit.edu/essl.html) (Dangovski et al. 2022). Many thanks to the authors for open-sourcing their code and methods.
