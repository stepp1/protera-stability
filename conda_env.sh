#!/bin/bash

conda create -n protera-stability \
    pytorch torchvision torchaudio cudatoolkit=11.1 \
    pytorch-lightning  pandas numpy matplotlib jupyterlab\
    scikit-learn=0.23.2 scikit-optimize skorch \
    biopython seaborn cloudpickle \
    -c pytorch-nightly -c nvidia