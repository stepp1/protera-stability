from typing import Iterator, List
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np


class SubsetDiversitySampler(torch.utils.data.Sampler):
    """Samples elements given their diversity w.r.t. the rest of the dataset"""

    def __init__(
        self,
        set_sequences: List[str],
        diversity_path: str,
        diversity_cutoff: float,
        max_size: int,
        strategy: str = "maximize",
        seed: int = None,
    ) -> None:
        """
        Args:
            set_indices List[str]: list of dataset indices
            diversity_path (str): Path to the csv with sequences and diversity.
            diversity_cutoff (float): value for a diversity cutoff
            max_size (int): maximum sample size
            strategy (str): Maximize or minize diversity. Default "maximize"
            seed (int): random seed. Default 123
        """

        self.diversity_path = diversity_path
        self.diversity_data = pd.read_csv(self.diversity_path)
        self.diversity_data = self.diversity_data[
            self.diversity_data.sequence.isin(set_sequences)
        ]

        if strategy == "maximize":
            sorting_order = False
        elif strategy == "minimize":
            sorting_order = True
        else:
            raise ValueError(f"Strategy {strategy} is not supported")

        self.indices = []
        self.sequences = []

        self.max_size = max_size
        self.set_sequences = set_sequences
        self.strategy = strategy
        self.seed = seed if seed is not None else pl.seed_everything()

        self.cutoff = diversity_cutoff

        self.stopped_by = self.subset_by_cutoff(
            self.cutoff, self.cutoff_func, sorting_order
        )

    def cutoff_func(self, x, cutoff, strategy):
        if strategy == "maximize":
            return x < cutoff
        elif strategy == "minimize":
            return x > cutoff

    def subset_by_cutoff(self, cutoff, cutoff_func, sorting_order) -> None:
        sorted_df = self.diversity_data.sort_values(
            by="diversity", ascending=sorting_order
        )

        indices = []
        sequences = []
        stopped_by = ""
        for row in sorted_df.itertuples():
            is_cutoff = cutoff_func(row.diversity, cutoff, self.strategy)
            if is_cutoff:
                stopped_by = "CUTOFF"
                break
            elif len(sequences) >= self.max_size:
                stopped_by = "MAX SIZE REACHED"
                break
            else:
                indices.append(row.Index)
                sequences.append(row.sequence)

        self.sequences = sequences
        self.indices = indices
        return stopped_by

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        return iter(rng.permutation(self.indices))

    def __len__(self) -> int:
        return len(self.indices)
