from copy import copy
from pathlib import Path
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import torch
from protera_stability.config.common.train import base_train
from protera_stability.config.common.scheduler import CosineLR
from protera_stability.config.common.optim import AdamW
from protera_stability.config.instantiate import instantiate
from protera_stability.config.lazy import LazyCall as L
from protera_stability.engine.lightning_train import DataModule, TrainingPl, default_cbs

# TODO: parse args recursively
def get_cfg(args):
    cfg = DictConfig(base_train)
    return cfg


def setup_train(cfg):
    """
    Default Training Configuration for Protein Diversity Experiments.

    Sets up optim, scheduler, trainer_params.callbacks, trainer_params.logger and trainer keys of a cfg.
    """
    # Wasn't setup before, then use default
    if not "optim" in cfg.keys():
        cfg.optim = copy(AdamW)

    # Was given a custom lr
    if "lr" in cfg.keys():
        cfg.optim.lr = cfg.lr

    # Wasn't setup before, then use default
    if not "scheduler" in cfg.keys():
        cfg.scheduler = copy(CosineLR)

    # setup directories
    log_dir = Path(f"{cfg.output_dir}/{cfg.experiment.name}")
    ckpt_dir = log_dir / "models"

    if not log_dir.exists():
        # shutil.rmtree(log_dir)
        log_dir.mkdir()
        ckpt_dir.mkdir()

    # default callbacks
    cbs = default_cbs(ckpt_dir, lazy=True)
    if "callbacks" in cfg.trainer_params.keys():
        cfg.trainer_params["callbacks"] = cfg.trainer_params["callbacks"] + cbs
    else:
        cfg.trainer_params["callbacks"] = cbs
    cfg.trainer_params["logger"] = L(pl_loggers.TensorBoardLogger)(save_dir=log_dir)
    return cfg


class DefaultTrainer(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = setup_train(cfg)

        # build trainer
        trainer_params = {k: instantiate(v) for k, v in cfg.trainer_params.items()}
        self.trainer = Trainer(**trainer_params)

        # build model
        self.model = instantiate(cfg.model)

        # build optim
        cfg.optim.params.model = self.model
        self.optimizer = instantiate(cfg.optim)

        # build scheduler
        cfg.scheduler.optimizer = self.optimizer
        self.scheduler = instantiate(cfg.scheduler)

        # build modules
        self.module = TrainingPl(
            cfg, model=self.model, optimizer=self.optimizer, schedulers=[self.scheduler]
        )
        self.data_module = DataModule(cfg)

    def fit(self, train_loader=None, valid_dataloader=None, kwargs={}):
        """
        Fits this Trainer with the defined lightning module and datamodule.

        It also allows for custom training and validation dataloaders.

        Returns:
            self
        """
        if train_loader is None and valid_dataloader is None:
            self.trainer.fit(self.module, datamodule=self.data_module, **kwargs)
            return self

        # only check if valid dataloader is none and let pl handle the train loader
        elif valid_dataloader is None:
            self.trainer.fit(self.module, train_loader, **kwargs)
            return self

        else:
            self.trainer.fit(self.module, train_loader, valid_dataloader, **kwargs)
            return self

    def test(self, dataloaders=None, kwargs={}):
        """
        Test this Trainer with the defined lightning module and datamodule.

        It also allows for custom test dataloaders.

        Returns:
            self
        """

        if dataloaders is None:
            self.trainer.test(datamodule=self.data_module, **kwargs)
            return self
        else:
            self.trainer.test(dataloaders=dataloaders, **kwargs)
