from typing import List
import numpy as np


def get_diversity_sequences(dataset, set_indices: List[bytes]):
    bytes_to_str = lambda x: x.decode("utf8")
    v_bytes_to_str = lambda x: list(np.vectorize(bytes_to_str)(x))

    return v_bytes_to_str(dataset.sequences[set_indices])


def get_dataset_indices(dataset):
    return dataset.indices


def get_max_size(dataset, max_percent: float = 0.8):
    return int(max_percent * len(dataset))


def get_random_indices(dataset, set_indices: List[int], random_percent: int = 0.3):
    np.random.shuffle(set_indices)

    random_split_idx = int(np.floor(random_percent * len(dataset)))
    return set_indices[:random_split_idx]


def get_max_size(dataset_size: int, max_percent: float = 0.8):
    return max_percent * dataset_size
