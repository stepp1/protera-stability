from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from protera_stability.config.lazy import LazyCall as L
from protera_stability.data.dataset import ProteinStabilityDataset
from protera_stability.data.sampler import SubsetDiversitySampler
from protera_stability.config.common.utils import (
    get_diversity_sequences,
    get_dataset_indices,
    get_max_size,
    get_random_indices,
    get_max_size,
)

# this assumes that we are running a structure like:
# project_name/
# |
# |--data/          # ----> where you have the data related to your project
# |   |
# |   |---stability_train.h5
# |   |---stability_test.h5
# |
# |--experiments/   # ----> where you are currently running the experiment

base_dataset = OmegaConf.create()
base_dataset.data = L(ProteinStabilityDataset)(
    proteins_path="../data/stability_train.h5",
    ret_dict=False,
)
base_dataset.random_split = 0.8


def get_train_val_indices(dataset, random_split: float):
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))

    np.random.shuffle(dataset_indices)

    val_split_index = int(np.floor((1 - random_split) * dataset_size))

    train_idx, val_idx = (
        dataset_indices[val_split_index:],
        dataset_indices[:val_split_index],
    )
    return train_idx, val_idx


base_dataloader = OmegaConf.create()

base_dataloader.train = L(torch.utils.data.DataLoader)(
    dataset=base_dataset.data,
    batch_size=256,
    num_workers=12,
    pin_memory=True,
)

base_dataloader.valid = L(torch.utils.data.DataLoader)(
    dataset=base_dataset.data,
    batch_size=256 * 2,
    num_workers=12,
    pin_memory=True,
)

base_dataloader.test = L(torch.utils.data.DataLoader)(
    dataset=L(ProteinStabilityDataset)(
        proteins_path="../data/stability_test.h5",
        ret_dict=False,
    ),
    batch_size=256 * 2,
    num_workers=12,
    pin_memory=True,
)

base_sampler = OmegaConf.create()

base_sampler.name = ""

base_sampler.diversity = L(SubsetDiversitySampler)(
    set_sequences=L(get_diversity_sequences)(
        dataset=base_dataset.data,
        set_indices=L(get_dataset_indices)(dataset=base_dataset.data),
    ),
    diversity_path="../data/stability_train_diversity.csv",
    diversity_cutoff=0.8,
    max_size=L(get_max_size)(dataset=base_dataset.data, max_percent=0.8),
)

base_sampler.random = L(torch.utils.data.SubsetRandomSampler)(
    indices=L(get_random_indices)(
        dataset=base_dataset.data,
        set_indices=L(get_dataset_indices)(dataset=base_dataset.data),
        random_percent=0.3,
    )
)
