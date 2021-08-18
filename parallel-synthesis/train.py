from typing import List, Sized, Iterator
import sys

from torch.utils.data import Sampler, DataLoader, SubsetRandomSampler, Dataset
import torch
import pandas as pd
import numpy as np
import h5py
import dill

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class ProteinStabilityDataset(Dataset):
    """Protein1D Stability Dataset."""

    def __init__(self, proteins_path, ret_dict = True):
        """
        Args:
            proteins_path (string): Path to the H5Py file that contains sequences and embeddings.
            ret_dict (bool): If True, it will return a dictionary as batch. Otherwise, X and y tensors will be returned.
        """
        self.stability_path = proteins_path
        self.ret_dict = ret_dict
        
        with h5py.File(str(self.stability_path), "r") as dset:
            self.sequences = dset["sequences"][:]
            self.X = dset["embeddings"][:]
            self.y = dset["labels"][:]
            
        self.indices = list(range(len(self.X)))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sequences = self.sequences[idx]
        embeddings = self.X[idx]
        labels = self.y[idx]

        sample = {
            'sequences': sequences, 
            'embeddings': torch.from_numpy(embeddings), 
            'labels': torch.Tensor([labels])
        }

        return sample if self.ret_dict else (sample['embeddings'], sample['labels'])
    
class SubsetDiversitySampler(Sampler):
    """Samples elements given their diversity w.r.t. the rest of the dataset"""
    
    def __init__(self, set_indices : list, diversity_path : str, diversity_cutoff : float, max_size: int, strategy : str = "maximize", seed : int = 123) -> None:
        """
        Args:
            set_indices (list): list of dataset indices
            diversity_path (string): Path to the csv with sequences and diversity.
            diversity_cutoff (float): value for a diversity cutoff
            max_size (int): maximum sample size
            strategy (str): Maximize or minize diversity. Default "maximize"
            seed (int): random seed. Default 123
        """
        
        self.diversity_path = diversity_path
        self.diversity_data = pd.read_csv(self.diversity_path, index_col=0)
        self.diversity_data = self.diversity_data[self.diversity_data.index.isin(set_indices)]
        
        if strategy == "maximize":
            sorting_order = False
        elif strategy == "minimize":
            sorting_order = True
        else: 
            raise ValueError(f"Strategy {strategy} is not supported")
        
        self.diversity_data = self.diversity_data.sort_values(
            by='diversity', 
            ascending=sorting_order
        )
        
        self.cutoff = diversity_cutoff
        self.indices = []
        self.max_size = max_size
        self.set_indices = set_indices
        self.strategy = strategy
        self.seed = seed
        
        if strategy == 'maximize':
            self.cutoff_lambda = lambda x, cutoff : x < cutoff
        elif strategy == 'minimize':
            self.cutoff_lambda = lambda x, cutoff : x > cutoff
            
        self.stopped_by = self.subset_by_cutoff(self.cutoff, self.cutoff_lambda)
        self.stopped_by_bin = 1 if self.stopped_by == "CUTOFF" else 0
        
        
    def subset_by_cutoff(self, cutoff, cutoff_func) -> None:
        stopped_by = ""
        indices = []
        for row in self.diversity_data.itertuples():
            is_cutoff = cutoff_func(row.diversity, cutoff)
            if is_cutoff:
                stopped_by = "CUTOFF"
                break
            elif len(indices) >= self.max_size:
                stopped_by = "MAX SIZE REACHED"
                break
            else:
                indices.append(row.Index)
                
        self.indices = [self.set_indices.index(subset_idx) for subset_idx in indices]
                
        return stopped_by
        
    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        return iter(rng.permutation(self.indices))

    def __len__(self) -> int:
        return len(self.indices)