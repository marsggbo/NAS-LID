import json
import os
from glob import glob
from types import MethodType

import numpy as np
import torch
from hydra.utils import instantiate
from hyperbox.engine.base_engine import BaseEngine
from hyperbox.mutator import EAMutator, EvolutionMutator
from hyperbox.mutator.utils import NonDominatedSorting
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import load_json
from omegaconf import DictConfig
from scipy.stats import kendalltau

log = get_logger(__name__)


def query_api(self, arch):
    '''
    self -> mutator
    self.model -> pytorch model
    self.pl_model -> pytorch-lightning module
    '''
    if hasattr(self, 'supernet_masks'):
        for i, supernet_mask in enumerate(self.supernet_masks):
            if self.is_subnet_in_supernet(arch, supernet_mask):
                ckpt_path = self.supernet_mask_paths[i].replace('mask.json', 'latest.ckpt')
                self.model.load_from_ckpt(ckpt_path)
    if hasattr(self.model, 'query_by_key'):
        real = self.model.query_by_key(self.query_key)
    else:
        real = -1
    proxy = self.__dict__['trainer'].validate(
        model=self.__dict__['pl_model'], datamodule=self.__dict__['datamodule'], verbose=False)[0][self.metric_key]
    encoding = self.arch2encoding(arch)
    info = self.vis_dict[encoding]
    info['real_perf'] = real
    info['proxy_perf'] = proxy
    return proxy


class EASearchEngine(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        supernet_mask_path_pattern: str=None,
        sample_iterations: int=1000,
        metric_key: str='val/acc_epoch',
        query_key: str='test_acc'
    ):
        super().__init__(trainer, model, datamodule, cfg)
        # assert isinstance(model.mutator, (EAMutator, EvolutionMutator))
        self.mutator = model.mutator
        self.metric_key = metric_key
        self.query_key = query_key
        self.sample_iterations = sample_iterations
        if supernet_mask_path_pattern is not None:
            self.supernet_mask_paths = glob(supernet_mask_path_pattern)
            self.supernet_masks = []
            for path in self.supernet_mask_paths:
                self.supernet_masks.append(load_json(path))
            self.mutator.supernet_mask_paths = self.supernet_mask_paths
            self.mutator.supernet_masks = self.supernet_masks
        self.performance_history = {}

    def reset_idx(self):
        self.idx = 0

    def parse_arch(self, pl_module):
        if hasattr(pl_module, 'arch_encoding'):
            arch = pl_module.arch_encoding
        elif hasattr(pl_module.network, 'arch_encoding'):
            arch = pl_module.network.arch_encoding
        else:
            raise NotImplementedError
        return arch

    def convert_list2tensor(self, src):
        for key, val in src.items():
            if isinstance(val, list):
                src[key] = torch.tensor(val)
        return src

    def run(self):
        # self.trainer.limit_val_batches = 50
        self.datamodule.setup()
        if 'CIFAR10DataModule' in self.datamodule.__class__.__name__:
            dataset = 'cifar10'
        elif 'CIFAR100DataModule' in self.datamodule.__class__.__name__:
            dataset = 'cifar100'
        self.performance_history = {}

        mutator = self.mutator
        self.mutator.__dict__['trainer'] = self.trainer
        self.mutator.__dict__['pl_model'] = self.model
        self.mutator.__dict__['datamodule'] = self.datamodule
        self.mutator.query_api = MethodType(query_api, self.mutator)
        self.mutator.metric_key = self.metric_key
        self.mutator.query_key = self.query_key
        try:
            self.mutator.search()

            reals = [info['real_perf'] for encoding, info in self.mutator.vis_dict.items() if 'real_perf' in info]
            proxies = [info['proxy_perf'] for encoding, info in self.mutator.vis_dict.items() if 'proxy_perf' in info]
            tau_visited, p_visited = kendalltau(reals, proxies)
            self.mutator.plot_real_proxy_metrics(
                real_metrics=reals,
                proxy_metrics=proxies,
                figname='evolution_real_proxy_metrics.png')


            reals = [info['real_perf'] for info in self.mutator.keep_top_k[self.mutator.topk] if 'real_perf' in info]
            proxies = [info['proxy_perf'] for info in self.mutator.keep_top_k[self.mutator.topk] if 'proxy_perf' in info]
            tau_topk, p_topk = kendalltau(reals, proxies)
            self.mutator.plot_real_proxy_metrics(
                real_metrics=reals,
                proxy_metrics=proxies,
                figname=f'evolution_top{self.mutator.topk}_real_proxy_metrics.png')

            results = {
                'tau_visited': tau_visited,
                'p_visited': p_visited,
                'tau_topk': tau_topk,
                'p_topk': p_topk,
            } 
            log.info(results)
            return results
        except Exception as e:
            self.mutator.save_checkpoint()
            return {'error_info': str(e)}
