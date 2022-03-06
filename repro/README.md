# PENCIL
A Modular Pytorch Re-Implementation of Probabilistic End-to-end Noise Correction for Learning with Noisy Labels (CVPR 2019)

# Introduction

[PENCIL](https://arxiv.org/abs/1903.07788) is a recent method for training vision models under label noise. This implementation makes PENCIL modular in order to make possible the use of PENCIL with other loss functions and training regimes. Big thanks to the authors and their official implementation (https://github.com/yikun2019/PENCIL) which was used as a starting point (cifar and resnet code was taken directly).

# Requirements

# Usage

* *main.ipynb* contains the core script needed to train and validate on CIFAR
* *pencil.py* contains the loss function and learning rate controls proposed in the PENCIL paper which may be integrated into other training settings.
