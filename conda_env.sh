#!/bin/bash

# create conda env
conda create -n protera-stability \
    pytorch torchvision torchaudio cudatoolkit=11.1 \
    pytorch-lightning scikit-learn=0.23.2 scikit-optimize skorch \
    pandas numpy matplotlib jupyterlab \
    biopython seaborn cloudpickle dill \
    -c pytorch-nightly -c nvidia

# install fair-esm
$CONDA_PREFIX/envs/protera-stability/bin/python -m pip install fair-esm 