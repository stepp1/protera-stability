import argparse
from copy import copy

import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from protera_stability.config.common.data import (
    base_dataloader,
    base_dataset,
    base_sampler,
    get_train_val_indices,
)
from protera_stability.config.instantiate import instantiate
from protera_stability.config.lazy import LazyCall as L
from protera_stability.models import ProteinMLP
from protera_stability.engine.default import DefaultTrainer, get_cfg


# TODO: SHOULD THIS BE IN configs/....py?
def setup_diversity(
    cfg,
    diversity_cutoff: float,
    random_percent: float,
    sampling_method: str = None,
    experiment_name: str = "base",
):
    """
    Default Setup for Protein Diversity Experiments.

    Sets up the experiment and model keys of a cfg.

    Args:
        diversity_cutoff (float): The diversity value to cutoff the dataset.
        random_percent (float): A percentage of the dataset to be sampled randomly
        sampling_method (str): One of ["", "diversity", "random"]
        experiment_name (str): The experiment name, a sort of prefix id.
    """
    cfg.experiment = OmegaConf.create()
    cfg.experiment.sampling_method = (
        sampling_method if sampling_method is not None else cfg.sampling_method
    )
    cfg.experiment.diversity_cutoff = diversity_cutoff
    cfg.experiment.random_percent = random_percent
    cfg.experiment.random_split = cfg.random_split

    if sampling_method == "random":
        exp_kind = f"{sampling_method}_{random_percent}"
    elif sampling_method == "random":
        exp_kind = f"{sampling_method}_{diversity_cutoff}"
    else:
        exp_kind = f"all-data"

    cfg.experiment.name = f"{experiment_name}_{exp_kind}"
    cfg.model = L(ProteinMLP)(
        n_in=1280,
        n_units=2048,
        n_layers=3,
        act=L(torch.nn.GELU)(),
        drop_p=0.7,
        last_drop=False,
    )
    return cfg


# TODO: SHOULD THIS BE IN configs/....py?
def setup_data(
    cfg, base_sampler=base_sampler, base_dl=base_dataloader
):
    """
    Default Data Configuration for Protein Diversity Experiments.

    Sets up the dataloader key of a cfg.
    """
    dataset = instantiate(base_dataloader.train)
    train_idx, valid_idx = get_train_val_indices(dataset, cfg.experiment.random_split)

    # Check training sampling
    train_sampler = copy(base_sampler)
    if cfg.experiment.sampling_method == "":
        # this is because we aren't using a "special" sampling method, therefore we directly pass the indices
        train_sampler.random.indices.set_indices = train_idx
        train_sampler.random.indices.random_percent = 1.0
        train_sampler = train_sampler.random

    elif cfg.experiment.sampling_method == "diversity":
        train_sampler.diversity.set_sequences.dataset = dataset
        train_sampler.diversity.set_sequences.set_indices = train_idx
        train_sampler.diversity.max_size = int(len(dataset) * 0.8)
        train_sampler = train_sampler.diversity

    elif cfg.experiment.sampling_method == "random":
        train_sampler.random.indices.dataset = dataset
        train_sampler.random.indices.set_indices = train_idx
        train_sampler = train_sampler.random

    else:
        raise ValueError(
            f"Sampling Method {cfg.experiment.sampling_method} is not valid."
        )

    # just pass the indices to the sampler
    valid_sampler = copy(base_sampler)
    valid_sampler.random.indices.set_indices = valid_idx
    valid_sampler.random.indices.random_percent = 1.0
    valid_sampler = valid_sampler.random

    # dataloaders
    dataloder = copy(base_dl)
    dataloder.train.sampler = train_sampler
    dataloder.valid.sampler = valid_sampler

    cfg.dataloader = dataloder
    return cfg


def setup(args={}):
    cfg = get_cfg(args)
    cfg = setup_diversity(cfg)
    cfg = setup_data(cfg)
    return cfg


def do_train(cfg):
    """
    Instantiates trainer and fits it.

    Sets up custom experiment parameters, in this case a specific callbacks, etc.

    We currently expect the cfg to have a trainer_params key. See config/common/train.py
    """
    # Add experiment specific callbacks
    stop_r2_reached = L(EarlyStopping)(
        monitor="valid/r2",
        patience=1,
        check_on_train_epoch_end=False,
        stopping_threshold=0.72,
        mode="max",
    )
    cfg.trainer_params["callbacks"] = [stop_r2_reached]

    trainer = DefaultTrainer(cfg)
    train_dl = trainer.data_module.train_dataloader()
    print(f"=== USING {cfg.experiment.sampling_method} as Sampling Method ===")
    print(
        f"=== USING {len(train_dl.sampler)} out of {len(train_dl.dataset)} samples ==="
    )

    if cfg.experiment.sampling_method == "diversity":
        print(f"=== SIZE WAS DETERMINED BY {train_dl.sampler.stopped_by} ===")

    elif cfg.experiment.sampling_method == "random":
        print(
            f"=== SIZE WAS DETERMINED BY RANDOM PERCENT OF {cfg.experiment.random_percent} ==="
        )

    # fit and return
    trainer.fit()
    return trainer


def do_test(cfg, trainer):
    trainer.test(datamodule=trainer.datamodule)
    return cfg, trainer


def main(args):
    cfg = setup(args)
    cfg, trainer = do_train(cfg)
    cfg, trainer = do_test(cfg, trainer)

    return cfg, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
