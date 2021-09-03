import copy
from typing import Any, Dict, List

import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from IPython.display import clear_output, display
from protera_stability.config.instantiate import instantiate
from protera_stability.config.lazy import LazyCall as L
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from torch.nn import functional as F


class TrainingLit(pl.LightningModule):
    """Training Protein Stability Regression Model"""

    # For ddp see: https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html#logging-torchmetrics
    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        schedulers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        super(TrainingLit, self).__init__()

        self.cfg = cfg

        self.model = model
        self.optim = optimizer
        self.lr_scheds = schedulers

        self.train_r2 = torchmetrics.R2Score()
        self.valid_r2 = torchmetrics.R2Score().clone()
        self.test_r2 = torchmetrics.R2Score().clone()

        self.r2 = {
            "train": self.train_r2,
            "valid": self.valid_r2,
            "test": self.test_r2,
        }

        hparams = self.cfg["experiment"]
        self.save_hyperparameters(hparams)
        # self.logger.log_hyperparams(params=hparams, metrics={})

    def forward(self, x):
        pred_stability = self.model(x)
        return pred_stability

    def do_step(self, batch, stage):
        X, y = batch
        y_hat = self.model(X)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss, "y_hat": y_hat.detach(), "y": y}

    def step_log(self, outputs, stage):
        self.r2[stage](outputs["y_hat"], outputs["y"])
        self.log(f"{stage}/r2_step", self.r2[stage])
        self.log(f"{stage}/loss_step", outputs["loss"], prog_bar=False, on_epoch=False)

    def epoch_log(self, avg_loss, stage):
        self.log(f"{stage}/r2", self.r2[stage].compute(), prog_bar=True)
        self.log(f"{stage}/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        outputs = self.do_step(batch, "train")
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.do_step(batch, "valid")
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.do_step(batch, "test")
        return outputs

    def training_step_end(self, outputs) -> None:
        self.step_log(outputs, "train")

    def validation_step_end(self, outputs) -> None:
        self.step_log(outputs, "valid")

    def test_step_end(self, outputs) -> None:
        self.step_log(outputs, "test")

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.epoch_log(avg_loss, "train")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.epoch_log(avg_loss, "valid")

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.epoch_log(avg_loss, "test")

    def configure_optimizers(self):
        optimizer = self.optim
        # lr_schedulers = []
        # for scheduler in self.lr_scheds:
        #     print(scheduler)
        #     scheduler_dict = {
        #         "scheduler": scheduler["scheduler"],
        #         "monitor": "valid/loss",
        #         "interval": scheduler["interval"]
        #     }
        #     lr_schedulers.append(scheduler_dict)
        return [optimizer], self.lr_scheds


LitProteins = TrainingLit


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        return instantiate(self.cfg.dataloader.train)

    def val_dataloader(self):
        return instantiate(self.cfg.dataloader.valid)

    def test_dataloader(self):
        return instantiate(self.cfg.dataloader.test)


def unwrap(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


class PrintCallback(Callback):
    def __init__(self):
        self.metrics = []
        self.cur_epoch = 0

    def on_epoch_end(self, trainer, pl_module):
        clear_output(wait=True)
        metrics_dict = copy.deepcopy(trainer.callback_metrics)

        # del metrics_dict["loss"]
        metrics_dict = {k: unwrap(v) for k, v in metrics_dict.items()}
        metrics_dict["epoch"] = self.cur_epoch
        self.cur_epoch += 1
        self.metrics.append(metrics_dict)

        del metrics_dict

        # column-names should be modified as per your usage
        metrics_df = pd.DataFrame.from_records(
            self.metrics,
            columns=[
                "epoch",
                "train_loss",
                "valid_loss",
                "valid_r2",
            ],
        )
        display(metrics_df)


def default_cbs(ckpt_dir, lazy = False):
    ckpt_cb = L(ModelCheckpoint)(dirpath=ckpt_dir)
    stop_valid = L(EarlyStopping)(
        monitor="valid/loss", patience=20, check_on_train_epoch_end=False,
    )
    monitor_lr = L(LearningRateMonitor)(logging_interval="epoch")

    callbacks = [ckpt_cb, stop_valid, monitor_lr]
    if not lazy:
        callbacks = [instantiate(cb) for cb in callbacks]
    return callbacks
