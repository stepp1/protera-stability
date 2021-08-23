from typing import Iterator
import sys

import torch
from torch.nn import functional as F
import torchmetrics

import fsspec  # imported first because of pl import error

import pytorch_lightning as pl
from sklearn import preprocessing
import pandas as pd
import numpy as np
import h5py
import dill


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class LitProteins(pl.LightningModule):
    """Training Protein Stability Regression Model"""

    # See for ddp: https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html#logging-torchmetrics
    def __init__(self, model, hparams):
        super(LitProteins, self).__init__()
        self.model = model
        self.r2 = torchmetrics.R2Score()

        self.save_hyperparameters(hparams)
        self.conf = hparams
        # self.logger.log_hyperparams(params=hparams, metrics={})

    def forward(self, x):
        pred_stability = self.model(x)
        return pred_stability

    def do_step(self, batch, stage):
        X, y = batch
        y_hat = self.model(X)
        loss = F.mse_loss(y_hat, y)

        self.log(f"{stage}_r2_step", self.r2(y_hat, y))
        return y_hat, loss

    def step_log(self, loss, stage):
        self.log(f"{stage}_loss_step", loss, prog_bar=False, on_epoch=False)

    def epoch_log(self, avg_loss, stage):
        self.log(f"{stage}_r2_epoch", self.r2.compute(), prog_bar=True)
        self.log(
            f"{stage}_loss_epoch", avg_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def training_step(self, batch, batch_idx):
        y_hat, loss = self.do_step(batch, "train")
        self.step_log(loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, loss = self.do_step(batch, "valid")
        self.step_log(loss, "valid")
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.epoch_log(avg_loss, "train")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([out for out in outputs]).mean()
        self.epoch_log(avg_loss, "valid")

    def configure_optimizers(self):
        optimizer = self.conf["optimizer"]["object"](
            self.model.parameters(), **self.conf["optimizer"]["params"]
        )

        schedulers = [
            (sched["object"](optimizer, **sched["params"]), sched["name"])
            for sched in self.conf["optimizer"]["schedulers"]
        ]

        lr_schedulers = []
        for schedule, name in schedulers:
            scheduler_dict = {
                "scheduler": schedule,
                "monitor": "valid_loss_epoch",
                "name": name,
            }
            lr_schedulers.append(scheduler_dict)

        return [optimizer], lr_schedulers


class ProteinStabilityDataset(torch.utils.data.Dataset):
    """Protein1D Stability Dataset."""

    def __init__(self, proteins_path, ret_dict=False):
        """
        Args:
            proteins_path (string): Path to the H5Py file that contains sequences and embeddings.
            ret_dict (bool): If True, it will return a dictionary as batch. Otherwise, X and y tensors will be returned.
        """
        self.stability_path = proteins_path
        self.ret_dict = ret_dict
        self.x_scaler = preprocessing.StandardScaler()
        self.y_scaler = preprocessing.StandardScaler()

        with h5py.File(str(self.stability_path), "r") as dset:
            self.sequences = dset["sequences"][:]
            X = dset["embeddings"][:]
            y = dset["labels"][:]

            self.X = self.x_scaler.fit_transform(X)
            self.y = self.y_scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

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
            "sequences": sequences,
            "embeddings": torch.from_numpy(embeddings),
            "labels": torch.Tensor([labels]),
        }

        return sample if self.ret_dict else (sample["embeddings"], sample["labels"])


class SubsetDiversitySampler(torch.utils.data.Sampler):
    """Samples elements given their diversity w.r.t. the rest of the dataset"""

    def __init__(
        self,
        set_indices: list,
        diversity_path: str,
        diversity_cutoff: float,
        max_size: int,
        strategy: str = "maximize",
        seed: int = 123,
    ) -> None:
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
        self.diversity_data = self.diversity_data[
            self.diversity_data.index.isin(set_indices)
        ]

        if strategy == "maximize":
            sorting_order = False
        elif strategy == "minimize":
            sorting_order = True
        else:
            raise ValueError(f"Strategy {strategy} is not supported")

        self.diversity_data = self.diversity_data.sort_values(
            by="diversity", ascending=sorting_order
        )

        self.cutoff = diversity_cutoff
        self.indices = []
        self.max_size = max_size
        self.set_indices = set_indices
        self.strategy = strategy
        self.seed = seed

        if strategy == "maximize":
            self.cutoff_lambda = lambda x, cutoff: x < cutoff
        elif strategy == "minimize":
            self.cutoff_lambda = lambda x, cutoff: x > cutoff

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
