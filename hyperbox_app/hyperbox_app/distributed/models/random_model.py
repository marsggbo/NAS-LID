import pdb
from typing import Any, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification.accuracy import Accuracy

from hyperbox.utils.logger import get_logger
from hyperbox.models.base_model import BaseModel

logger = get_logger(__name__)


class RandomModel(BaseModel):
    """(Random Search)
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        scheduler_cfg: Optional[Union[DictConfig, dict]] = None,
        is_sync: bool = True,
        is_net_parallel: bool = True,
        num_subnets: int = 1,
        sample_interval: int = 1,
        set_to_none: bool = False,
        **kwargs
    ):
        r"""Random NAS model
        Args:
            network [DictConfig, dict, torch.nn.Module]:
            mutator [DictConfig, dict, BaseMutator]:
            optimizer [DictConfig, dict, torch.optim.Optimizer]:
            loss Optional[DictConfig, dict, Callable]: loss function or DictConfig of loss function
            metric: metric function, such as Accuracy, Precision, etc.
        """
        super().__init__(
            network_cfg,
            mutator_cfg,
            optimizer_cfg,
            loss_cfg,
            metric_cfg,
            scheduler_cfg,
            **kwargs
        )
        self.automatic_optimization = False
        self.num_subnets = num_subnets
        self.sample_interval = sample_interval
        self.is_sync = is_sync
        self.set_to_none = set_to_none
        # self.network = self.network.to(self.device)

    def sample_search(self):
        super().sample_search(self.is_sync, self.is_net_parallel)

    def forward(self, x):
        return self.network(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=self.set_to_none)
        self.network.train()
        self.mutator.eval()
        loss = 0.
        acc = 0.
        if hasattr(self.mutator, 'num_path'):
            num_subnets = self.mutator.num_path
        else:
            num_subnets = self.num_subnets
        for i in range(num_subnets):
            if batch_idx % self.sample_interval == 0:
                self.sample_search()
            loss_subnet, preds_subnet, targets_subnet = self.step(batch)
            loss_subnet.backward()
            acc_subnet = self.train_metric(preds_subnet, targets_subnet)
            loss += loss_subnet
            acc += acc_subnet
        opt.step()
        loss /= num_subnets
        acc /= num_subnets
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        return {'loss': loss}
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        # return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        if not self.automatic_optimization and self.lr_schedulers():
            self.lr_schedulers().step()
        acc_epoch = self.trainer.callback_metrics['train/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['train/loss_epoch'].item()
        logger.info(f'Train epoch{self.trainer.current_epoch} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')

    # def on_validation_start(self):
    #     if self.is_network_search:
    #         try:
    #             # if not self.mutator._cache:
    #             #     self.mutator.reset()
    #             # self.reset_running_statistics(subset_size=256, subset_batch_size=64)
    #             pass
    #         except Exception as e:
    #             print(e)
    #             print('you should reset the mutator before validation in search mode.')

    # def on_validation_epoch_start(self):
    #     if self.is_network_search:
    #         try:
    #             # if not self.mutator._cache:
    #             #     self.mutator.reset()
    #             self.reset_running_statistics(subset_size=64, subset_batch_size=32)
    #         except Exception as e:
    #             print(e)
    #             print('you should reset the mutator before validation in search mode.')

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_metric(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc_epoch = self.trainer.callback_metrics['val/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['val/loss_epoch'].item()
        logger.info(f'Val epoch{self.trainer.current_epoch} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        acc = self.trainer.callback_metrics['test/acc'].item()
        loss = self.trainer.callback_metrics['test/loss'].item()
        logger.info(f'Test epoch{self.trainer.current_epoch} acc={acc:.4f} loss={loss:.4f}')

