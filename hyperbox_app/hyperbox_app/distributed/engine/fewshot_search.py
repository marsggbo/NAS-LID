import itertools
import json
import os
import pickle
import random
import types
from copy import deepcopy
from glob import glob
from typing import List, Optional, Union

import hydra
import networkx as nx
import numpy as np
import skdim
import wandb
import torch
import torch.multiprocessing as mp
from sklearn.mixture import GaussianMixture
from hydra.utils import instantiate
from hyperbox.engine.base_engine import BaseEngine
from hyperbox.mutables.spaces import OperationSpace
from hyperbox.mutator import RandomMutator
from hyperbox.utils import logger, utils
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import hparams_wrapper, load_json, save_arch_to_json
from hyperbox_app.distributed.networks.nasbench201.nasbench201 import \
    NASBench201Network
from hyperbox_app.distributed.utils.twonn import twonn1, twonn2
from ipdb import set_trace
from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_lightning.callbacks import Callback
from scipy.stats import kendalltau
from skdim.id import ESS
from sklearn.cluster import SpectralClustering

log = get_logger(__name__)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


@hparams_wrapper
class FewshotSearch(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        warmup_epochs: Optional[Union[List[int], int]]=[20, 40],
        finetune_epoch: int=10,
        load_from_parent: bool=False,
        split_criterion: str='ID', # 'ID' or 'grad'
        split_num: int=2,
        ID_method: str='lid', # 'lid' or 'ess' or 'mle' or 'twonn'
        split_method: str='spectral_cluster', # 'spectral_cluster' or 'mincut'
        similarity_method: str='cosine', # 'corre' or 'cosine'
        is_single_path: bool=False,
        repeat_num: int=1,
        supernet_masks_path: str=None,
        global_pool_path: str=None,
    ):
        super().__init__(trainer, model, datamodule, cfg)
        datamodule_cfg = deepcopy(self.cfg.datamodule)
        datamodule_cfg.batch_size = 128
        datamodule = instantiate(datamodule_cfg)
        datamodule.setup()
        self.dataloader = datamodule.train_dataloader()
        self.mutator = self.model.mutator
        self.supernet_masks_path = supernet_masks_path # e.g., /path/to/*json
        self.global_pool_path = global_pool_path # e.g., /path/to/*pt

        network_name = cfg.model.network_cfg._target_.split('.')[-1]
        project = f"{network_name}_{split_criterion}_{split_method}_{similarity_method}"
        offline = cfg.logger.wandb.get('offline', False)
        # if offline:
        #     os.environ["WANDB_MODE"] = "offline"
        # wandb.init(project=project)

    def run(self):
        trainer = self.trainer
        model = self.model
        datamodule = self.datamodule
        config = self.cfg

        if self.global_pool_path is not None:
            global_pools = torch.load(self.global_pool_path)
            labels = np.array([v['label'] for v in global_pools.values()])
            processes = []
            # finetune all supernets
            try:
                mp.set_start_method('spawn', force=True)
                log.info('Start spawning all supernets')
            except Exception as e:
                log.info('Already spawned')
                raise e
            for rank, label in enumerate(set(labels)):
                indices = np.where(labels == label)[0]
                mask_pool = [global_pools[str(i)]['mask'] for i in indices]
                finetune_epoch = self.hparams.get('finetune_epoch', 20)
                p = mp.Process(target=finetune_mp_from_pool, args=(config, rank, mask_pool, finetune_epoch))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            log.info('Finished finetuning all supernets of the pool')
        elif self.supernet_masks_path is None:
            # split search space
            split_mask_history = {}
            supernet_mask = {m.key: torch.ones_like(m.mask) for m in self.mutator.mutables}
            all_supernet_settings = [
                [
                    [trainer, model, [supernet_mask], None],                    
                ],
            ]
            # warmup and split the supernet
            for level, warmup_epoch in enumerate(self.warmup_epochs):
                log.info(f'Level {level}: warmup for {warmup_epoch} epochs')
                supernet_settings = all_supernet_settings[level]
                new_supernet_settings = []

                # train all supernets at current level, and split them along the way
                for idx, supernet_setting in enumerate(supernet_settings):
                    flag = f'{level}_{idx}'
                    parent_trainer, parent_model, supernet_masks, best_edge_key = supernet_setting
                    if not isinstance(supernet_masks, list):
                        supernet_masks = [supernet_masks]
                    for supernet_mask in supernet_masks:
                        # warmup
                        trainer, model = self.warmup(
                            parent_trainer, parent_model, datamodule, config,
                            warmup_epoch, supernet_mask)

                        # split current search space (supernet)
                        splitted_supernet_masks, best_infos, best_edge_key = self.split_supernet(
                            trainer, model, datamodule, config,
                            supernet_mask, level, self.hparams
                        )
                        new_supernet_settings.append([trainer, model, deepcopy(splitted_supernet_masks), best_edge_key])

                if len(new_supernet_settings) > 0:
                    split_mask_history[level] = [deepcopy(s[2]) for s in new_supernet_settings]
                    all_supernet_settings.append(new_supernet_settings)

            level = -1
            save_root_path = os.path.join(os.getcwd(), f'checkpoints')
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)
            path = os.path.join(save_root_path, f'split_mask_history.json')
            save_arch_to_json(split_mask_history, path)
            # save supernet masks
            for idx, supernet_setting in enumerate(all_supernet_settings[level]):
                parent_trainer, parent_model, supernet_masks, best_edge_key = supernet_setting
                for idy, supernet_mask in enumerate(supernet_masks):
                    flag = f"level[{level}]-[{idx}-{idy}]-Edge[{best_edge_key}]-subSupernet"
                    mask_path = os.path.join(save_root_path, f"{flag}_mask.json")
                    save_arch_to_json(supernet_mask, mask_path)

            # finetune all supernets
            try:
                mp.set_start_method('spawn', force=True)
                log.info('Start spawning all supernets')
            except Exception as e:
                log.info('Already spawned')
                raise e
            group_size = torch.cuda.device_count()
            num_processes = 0
            for idx, supernet_setting in enumerate(all_supernet_settings[level]):
                parent_trainer, parent_model, supernet_masks, best_edge_key = supernet_setting
                for rank, supernet_mask in enumerate(supernet_masks):
                    num_processes += 1
            num_unprocessed = num_processes
            processes = []
            parent_ckpt_paths = []
            global_rank = 0
            load_from_parent = self.hparams.get('load_from_parent', False)
            for idx, supernet_setting in enumerate(all_supernet_settings[level]):
                parent_trainer, parent_model, supernet_masks, best_edge_key = supernet_setting
                for rank, supernet_mask in enumerate(supernet_masks):
                    flag = f"level[{level}]-[{idx}-{rank}]-Edge[{best_edge_key}]-subSupernet"
                    log.info(f"Fintune {flag} with mask {supernet_mask}")
                    
                    parent_ckpt_path = f'{os.getcwd()}/temp_{idx}_{rank}.ckpt'
                    parent_ckpt_paths.append(parent_ckpt_path)
                    if load_from_parent:
                        parent_trainer.save_checkpoint(parent_ckpt_path)
                    finetune_epoch = self.hparams.get('finetune_epoch', 20)
                    p = mp.Process(target=finetune_mp, args=(
                        config, global_rank, None, load_from_parent,
                        supernet_mask, finetune_epoch, parent_ckpt_path, flag
                        )
                    )
                    p.start()
                    processes.append(p)
                    num_unprocessed -= 1
                    global_rank += 1
                
                if len(processes) >= group_size or num_unprocessed <= 0:
                    for p in processes:
                        p.join()
                    log.info(f"Finetune level-{idx}: {len(processes)}  finished. \n\n")
                    log.info(f"="*20)
                    processes = []
            if load_from_parent:
                try:
                    for parent_ckpt_path in parent_ckpt_paths:
                        os.remove(parent_ckpt_path)
                except Exception as e:
                    log.info(f"{e}")
        else:
            # load supernet masks
            supernet_masks_path = glob(self.supernet_masks_path)
            supernet_masks = [load_json(path) for path in supernet_masks_path]
            
            processes = []
            # finetune all supernets
            try:
                mp.set_start_method('spawn', force=True)
                log.info('Start spawning all supernets')
            except Exception as e:
                log.info('Already spawned')
                raise e
            model.share_memory()
            group_size = torch.cuda.device_count()
            num_processes = len(supernet_masks_path)
            rank_id_list = list(range(num_processes))
            list_of_groups = zip(*(iter(rank_id_list),) * group_size)
            for group in list_of_groups:
                # for local_rank, rank in enumerate(group):
                for rank in group:
                    mask_path = supernet_masks_path[rank]
                    ckpt_path = mask_path.replace('mask.json', 'latest.ckpt')
                    # if os.path.exists(ckpt_path):
                    #     continue
                    supernet_mask = load_json(mask_path)
                    log.info(f"Fintune [{rank}]-th sub-supernet with mask {supernet_mask}")
                    load_from_parent = self.hparams.get('load_from_parent', False)
                    load_from_parent = False
                    parent_ckpt_path = f'./temp_{rank}.ckpt'
                    finetune_epoch = self.hparams.get('finetune_epoch', 20)
                    p = mp.Process(target=finetune_mp, args=(
                        config, rank, mask_path, load_from_parent,
                        supernet_mask, finetune_epoch, parent_ckpt_path
                        )
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
                
        return {}

    def warmup(
        self,
        parent_trainer, parent_model, datamodule, config,
        warmup_epoch: int=20, supernet_mask: dict=None):
        '''
        Warmup the model by training for a few epochs.
        '''
        # rebuild a model with the weights of parent_model
        model_cfg = deepcopy(config.model)
        model = instantiate(model_cfg, _recursive_=False)
        model.load_state_dict(parent_model.state_dict())
        # model = deepcopy(parent_model)
        mutator = model.mutator
        mutator.supernet_mask = deepcopy(supernet_mask)

        trainer_cfg = deepcopy(config.trainer)
        trainer_cfg.max_epochs = warmup_epoch
        to_resume = False
        try:
            parent_trainer.save_checkpoint('./temp.ckpt')
            trainer_cfg.resume_from_checkpoint = './temp.ckpt'
            to_resume = True
        except Exception as e:
            pass
        callbacks = parent_trainer.callbacks
        trainer = instantiate(trainer_cfg, callbacks=callbacks,
            logger=parent_trainer.logger, _convert_="partial")
        trainer.fit(model, datamodule)
        if to_resume:
            os.system('rm ./temp.ckpt')
        return trainer, model

    def split_supernet(
        self, trainer, model, datamodule, config,
        supernet_mask: dict, level, hparams: dict=None
    ):
        '''
        Split the supernet into sub-supernets.
        '''
        base_mask = deepcopy(supernet_mask)
        edge_keys = list(supernet_mask.keys())
        mutator = model.mutator
        is_single_path = hparams.get('is_single_path', False)
        repeat_num = hparams.get('repeat_num', 1)
        split_criterion = hparams.get('split_criterion', 'grad')
        split_method = hparams.get('split_method', 'spectral_cluster')
        split_num = hparams.get('split_num', 2)
        if isinstance(split_num, (list, ListConfig)):
            if len(split_num) == 1:
                split_num = split_num[0]
            else:
                split_num = split_num[level]
        ID_method = hparams.get('ID_method', 'lid')
        similarity_method = hparams.get('similarity_method', 'cosine')

        # Todo: implement the following methods
        if split_criterion=='random':
            return self.split_supernet_random(
                trainer, model, datamodule, config, supernet_mask, level, hparams)

        # best_value = -1e10 if split_method == 'spectral_cluster' else 1e10
        best_value = 0
        best_partition = []
        best_edge_key = None
        best_infos = None
        # enumerate all edges in the supernet
        for edge_key in edge_keys:
            if not base_mask[edge_key].all():
                # if an edge has already been split, skip it
                # [1,1,1,1,1] indicates that the edge has not been split
                # [1,0,1,0,0] indicates that the edge has been split
                continue
            num_ops = len(base_mask[edge_key])
            # enumerate all enabled operations of the current edge
            similarity_avg = 0
            for batch_idx, data in enumerate(self.dataloader):
                if batch_idx == repeat_num:
                    break
                
                crt_mask = deepcopy(base_mask)
                for k, v in base_mask.items():
                    if k == edge_key:
                        crt_mask[k] = torch.zeros_like(v)
                    if is_single_path:
                        # sample a single path each time,
                        # each path differs from only one operation in the current edge
                        indices = torch.where(v != 0)[0]
                        gen_index = random.choice(indices)
                        crt_mask[k] = torch.nn.functional.one_hot(gen_index, len(v)).float()
                
                # get gradients
                infos = {}
                for op_idx in range(num_ops):
                    crt_mask[edge_key] = torch.zeros_like(base_mask[edge_key])
                    crt_mask[edge_key][op_idx] = 1
                    crt_mask = {k: v.bool() for k, v in crt_mask.items()}
                    mutator.sample_by_mask(crt_mask)
                    flag = (edge_key, op_idx)
                    info = {}
                    if split_criterion == 'grad':
                        grads = calc_grads(model, data, edge_key, crt_mask).cpu().numpy()
                        info = {
                            'mask': crt_mask,
                            'grads': grads,
                            'criterion': grads,
                            'edge_key': edge_key,
                            'op_idx': op_idx,
                        }
                    elif split_criterion == 'ID':
                        IDs = calc_IDs(model.network, data, 2, ID_method)
                        # IDs = calc_IDs(model.network, self.dataloader, 2)
                        info = {
                            'mask': crt_mask,
                            'IDs': IDs,
                            'ID': IDs.mean(0),
                            'criterion': IDs.mean(0),
                            'edge_key': edge_key,
                            'op_idx': op_idx,
                        }
                    if flag not in infos:
                        infos[flag] = info
                    else:
                        criterion = infos[flag]['criterion']
                        criterion += info['criterion']
                        criterion /= 2
                        infos[flag]['criterion'] = criterion

                # calculate the similarity between all edges
                if split_criterion == 'grad':
                    similarity = self.calc_similarity(infos, method=similarity_method)
                elif split_criterion == 'ID':
                    # similarity = self.calc_similarity(infos, method='correcoef')
                    similarity = self.calc_similarity(infos, method=similarity_method)
                similarity = np.nan_to_num(similarity) + 1
                # log.info(f"edge_key={edge_key} similarity={similarity}")
                similarity_avg += similarity
            similarity_avg /= repeat_num
            similarity = similarity_avg - 1
            log.info(f"{edge_key} similarity:\n {similarity}")
            # try:
            #     name = f"{trainer.current_epoch}_{edge_key}_similarity"
            #     x_labels = y_labels = list(range(len(similarity)))
            #     wandb.log({name: wandb.plots.HeatMap(x_labels, y_labels, similarity)})
            # except Exception as e:
            #     log.error(e)
            
            if split_method == 'GM':
                criterions = np.stack([info['criterion'] for info in infos.values()])
                gm = GaussianMixture(n_components=split_num, random_state=0).fit(criterions)
                labels = np.array(gm.predict(criterions))
                u_labels = np.unique(labels)
                cut_value = 0.
                for label in u_labels:
                    mask = labels == label
                    sim = np.nan_to_num(similarity)[mask][:, mask]
                    cut_value += sim.mean()
                cut_value /= len(u_labels)
                partition = [np.where(labels==i)[0] for i in set(labels)]
                if cut_value > best_value:
                    best_value = cut_value
                    best_edge_key = edge_key
                    best_partition = partition
                    best_infos = infos
                    log.info(f"Edge {edge_key}: Best cluster={best_partition} with cut value {best_value:4f}")   
            elif split_method == 'spectral_cluster':
                log.info("Split the supernet into clusters...")
                cluster, cluster_sim_avg = self.gen_cluster(similarity, split_num)
                log.info(f"Cluster={cluster} with averaged similarity {cluster_sim_avg:4f}")

                if cluster_sim_avg > best_value:
                    best_value = cluster_sim_avg
                    best_edge_key = edge_key
                    labels = cluster.labels_
                    best_partition = [np.where(labels==i)[0] for i in set(labels)]
                    best_infos = infos
                    log.info(f"Edge {edge_key}: Best cluster={best_partition} with averaged similarity {best_value:4f}")
            elif split_method == 'mincut':
                cut_value, partition = mincut(similarity, split_num)
                if cut_value > best_value:
                    best_value = cut_value
                    best_edge_key = edge_key
                    best_partition = partition
                    best_infos = infos
                    log.info(f"Edge {edge_key}: Best cluster={best_partition} with cut value {best_value:4f}")
            elif split_method == 'stoer_wagner':
                G = nx.from_numpy_matrix(similarity)
                cut_value, partition = nx.stoer_wagner(G)
                if cut_value > best_value:
                    best_value = cut_value
                    best_edge_key = edge_key
                    best_partition = partition
                    best_infos = infos
                    log.info(f"Edge {edge_key}: Best cluster={best_partition} with cut value {best_value:4f}")

        supernet_masks = []
        for indices in best_partition:
            crt_mask = deepcopy(base_mask)
            crt_mask[best_edge_key] = torch.zeros_like(base_mask[best_edge_key])
            keys = list(best_infos.keys())
            for idx in range(len(indices)):
                crt_mask[best_edge_key][indices[idx]] = 1
            supernet_masks.append(crt_mask)
        return supernet_masks, best_infos, best_edge_key

    # Todo: implement the following methods
    def split_supernet_random(
        self, trainer, model, datamodule, config,
        supernet_mask: dict, level, hparams: dict=None
    ):
        edge_keys = list(supernet_mask.keys())
        best_edge_key = random.choice(edge_keys)

        base_mask = deepcopy(supernet_mask)
        while not base_mask[best_edge_key].all():
            log.info(f"{best_edge_key} has been split. select another edge...")
            best_edge_key = random.choice(edge_keys)
        supernet_masks = []
        split_num = hparams.get('split_num', 2)
        if isinstance(split_num, (list, ListConfig)):
            if len(split_num) == 1:
                split_num = split_num[0]
            else:
                split_num = split_num[level]
        partition = list(range(len(base_mask[best_edge_key])))
        num_partition = len(partition)
        random.shuffle(partition)

        split_indices = np.random.choice(list(range(1,num_partition-1)), split_num-1, replace=False)
        split_indices = sorted(split_indices)
        best_partition = [partition[i:j] for i, j in zip([0]+split_indices, split_indices+[None])]

        for indices in best_partition:
            crt_mask = deepcopy(base_mask)
            crt_mask[best_edge_key] = torch.zeros_like(base_mask[best_edge_key])
            for idx in range(len(indices)):
                crt_mask[best_edge_key][indices[idx]] = 1
            supernet_masks.append(crt_mask)
        return supernet_masks, None, best_edge_key

    def finetune(
        self,
        parent_trainer, parent_model, datamodule, config,
        finetune_epoch: int, supernet_mask: dict, hparams: dict):
        '''Finetune the model by training for a few epochs.
        '''
        model_cfg = deepcopy(config.model)
        model = instantiate(model_cfg, _recursive_=False)
        load_from_parent = hparams.get('load_from_parent', False)
        if load_from_parent:
            model.load_state_dict(parent_model.state_dict())
        # model = deepcopy(parent_model)
        mutator = model.mutator
        mutator.supernet_mask = deepcopy(supernet_mask)

        trainer_cfg = deepcopy(config.trainer)
        trainer_cfg.max_epochs = hparams.finetune_epoch
        if load_from_parent:
            try:
                parent_trainer.save_checkpoint('./temp.ckpt')
                trainer_cfg.resume_from_checkpoint = './temp.ckpt'
            except Exception as e:
                pass
        callbacks = parent_trainer.callbacks
        trainer = instantiate(trainer_cfg, callbacks=callbacks,
            logger=parent_trainer.logger, _convert_="partial")
        trainer.fit(model, datamodule)
        return trainer, model

    def calc_similarity(self, infos, method='correcoef'):
        '''Calculate the similarity between all edges (infos).
        Args:
            infos: a dict of edge info, with the following keys:
                criterion: the criterion of the edge, e.g., ID, grad
            method: the similarity method, can be 'correcoef' or 'cosine'
        
        Returns:
            similarity: a matrix of similarity between all edges
        '''
        criterions = [info['criterion'] for info in infos.values()]
        if isinstance(criterions[0], torch.Tensor):
            criterions = torch.stack(criterions).numpy()
        else:
            criterions = np.vstack(criterions)

        if method == 'correcoef':
            similarity = np.corrcoef(criterions)
        elif method == 'cosine':
            num = criterions.shape[0]
            similarity = np.zeros((num, num))
            for i in range(num):
                for j in range(i, num):
                    similarity[i, j] = np.dot(criterions[i, :], criterions[j, :]) / (
                        np.linalg.norm(criterions[i]) * np.linalg.norm(criterions[j]) + 1e-6
                    )
            tril_indices = np.tril_indices(num, -1)
            similarity[tril_indices] = similarity.T[tril_indices]
        elif method == 'mse':
            criterions = torch.tensor(criterions)
            distance = torch.cdist(criterions, criterions, p=2).numpy()
            similarity = 1 / (distance + 1 + 1e-6)
        return similarity

    def gen_cluster(self, similarity: np.array, n_clusters: int=None):
        x = similarity
        x = np.nan_to_num(x) + 1 
        best_cluster = None
        best_n_clusters = 0
        best_sim = 0
        best_labels = None
        if n_clusters is not None:
            n_cluster_list = [n_clusters]
        else:
            n_cluster_list = list(range(2, 10))
        for n_clusters in n_cluster_list:
            cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', affinity='precomputed', random_state=0)
            cluster.fit_predict(x)
            labels = cluster.labels_
            u_labels = np.unique(labels)
            sim_avg = 0
            for label in u_labels:
                mask = labels == label
                sim = np.nan_to_num(similarity)[mask][:, mask]
                # log.info(f"label={label}: {sum(labels==label)} {sim.mean()}")
                sim_avg += sim.mean()
            sim_avg /= len(u_labels)
            if sim_avg > best_sim:
                best_sim = sim_avg
                best_cluster = cluster
                best_n_cluster = n_clusters
                best_labels = labels
            log.info(f"n_clusters={n_clusters} avg similarity={sim_avg}")
        log.info(f"Best settings: {best_cluster} with sim {best_sim}")
        # return best_cluster, best_n_clusters, best_sim, best_labels
        return best_cluster, best_sim

    def save_pkl(self, data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


# def mincut(dist_avg, split_num): # note: this is not strictly mincut, but it's fine for 201
#     # assert split_num == 2, 'always split into 2 groups for 201 (when using gradient to split)'
#     assert isinstance(dist_avg, np.ndarray)
#     dist_avg = dist_avg - np.tril(dist_avg)
#     best_dist, best_groups, best_edge_score = float('inf'), [], 0
#     for opid1 in range(dist_avg.shape[0]):
#         for opid2 in range(opid1 + 1, dist_avg.shape[0]):
#             group1 = np.array([opid1, opid2]) # always 2
#             group2 = np.setdiff1d(np.array(list(range(dist_avg.shape[0]))), group1)
#             dist = dist_avg[group1[0], group1[1]] + dist_avg[group2[0], group2[1]]
#             if group2.shape[0] > 2:
#                 dist += dist_avg[group2[0], group2[2]] + dist_avg[group2[1], group2[2]]
#             if dist < best_dist:
#                 best_dist = dist
#                 best_groups = [group1, group2]
#                 best_edge_score = dist_avg.sum() - best_dist # dist_avg should be upper-triangular
#     return best_edge_score, best_groups

def init_callbacks(config):
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))
    return callbacks

def init_logger(config):
    loggers = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(instantiate(lg_conf))
    return loggers

def finetune_mp(config, rank: int, mask_path: str, load_from_parent: bool,
    supernet_mask: dict, finetune_epoch: int, parent_ckpt_path: str, flag: str=None):
    log_mp = get_logger(f"finetune_mp_{rank}", is_rank_zero=False)
    if mask_path is not None:
        log_mp.info(f"rank={rank} mask_path={mask_path}")
    pid = os.getpid()
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    log_mp.info(f'start finetune in process {pid}, local_rank {local_rank} global_rank {rank}')
    # log_mp.info(f'start finetune in process {pid}, global rank {rank}')
    datamodule_cfg = config['datamodule']
    datamodule_cfg.num_workers = 2
    datamodule = instantiate(config["datamodule"])

    model_cfg = deepcopy(config.model)
    model = instantiate(model_cfg, _recursive_=False)

    mutator = model.mutator
    mutator.supernet_mask = deepcopy(supernet_mask)

    trainer_cfg = deepcopy(config.trainer)
    trainer_cfg.gpus = [local_rank]
    trainer_cfg.max_epochs = finetune_epoch
    # each process uses 1 GPU for finetuning
    # "ddp" will throw unexcepted errors
    trainer_cfg.strategy = "dp"    
    if load_from_parent and parent_ckpt_path:
        trainer_cfg.resume_from_checkpoint = parent_ckpt_path
        ckpt_epoch = torch.load(parent_ckpt_path, map_location='cpu')['epoch']
        trainer_cfg.max_epochs = finetune_epoch + ckpt_epoch
    
    callbacks = init_callbacks(config)
    logger = init_logger(config)
    # trainer = instantiate(trainer_cfg, callbacks=callbacks, logger=logger, _convert_="partial")
    trainer = instantiate(trainer_cfg, callbacks=callbacks, logger=[], _convert_="partial")
    trainer.fit(model, datamodule)
    results = trainer.callback_metrics
    log_mp.info(f"pid={pid} Fintune Done: [{rank}]-th sub-supernet with mask {supernet_mask} \nresults: {results}")
    
    level = -1
    if mask_path is not None:
        flag = mask_path.replace('mask.json', 'latest.ckpt')
        ckpt_path = flag
    else:
        ckpt_path = os.path.join(os.getcwd(), f'checkpoints/{flag}_latest.ckpt')
    trainer.save_checkpoint(f"{ckpt_path}")
    log_mp.info(f"pid={pid} Saved [{rank}]-th sub-Supernet checkpoint to {ckpt_path}")
    return trainer, model


def mincut(sim_avg, split_num): # note: this is not strictly mincut, but it's fine for 201
    # assert split_num == 2, 'always split into 2 groups for 201 (when using gradient to split)'
    assert isinstance(sim_avg, np.ndarray)
    if split_num==2:
        sim_avg = sim_avg - np.tril(sim_avg)
        best_sim, best_groups, best_edge_score = -1*float('inf'), [], 0
        for opid1 in range(sim_avg.shape[0]):
            for opid2 in range(opid1 + 1, sim_avg.shape[0]):
                group1 = np.array([opid1, opid2]) # always 2
                group2 = np.setdiff1d(np.array(list(range(sim_avg.shape[0]))), group1)
                sim = sim_avg[group1[0], group1[1]] + sim_avg[group2[0], group2[1]]
                if group2.shape[0] > 2:
                    sim += sim_avg[group2[0], group2[2]] + sim_avg[group2[1], group2[2]]
                if sim > best_sim:
                    best_sim = sim
                    best_groups = [group1, group2]
                    best_edge_score = sim_avg.sum() - best_sim # sim_avg should be upper-triangular
        return best_edge_score, best_groups
    # Todo: implement mincut for 3 groups
    elif split_num==3:
        vertex = [i for i in range(sim_avg.shape[0])]
        best_sim, best_groups, best_edge_score = -1*float('inf'), [], 0
        sim = 0
        for p in itertools.permutations(vertex):
            p_list = list(p)
            for i in range(1, len(vertex) // 2 + 1):
                for j in range(i+2, len(vertex)-1):
                    for edge in itertools.combinations(vertex, 2):
                        if (edge[0] in p_list[0:i+1] and edge[1] in p_list[0:i+1]) or \
                                (edge[0] in p_list[i+1:j+1] and edge[1] in p_list[i+1:j+1]) or \
                                (edge[0] in p_list[j+1:] and edge[1] in p_list[j+1:]):
                            group1 = np.array(p_list[0:i+1])
                            group2 = np.array(p_list[i+1:j+1])
                            group3 = np.array(p_list[j+1:])
                            sim += sim_avg[edge[0], edge[1]]
                    if sim > best_sim:
                        best_sim = sim
                        sim = 0
                        best_groups = [group1, group2, group3]
                        best_edge_score = sim_avg.sum() - best_sim # the smaller the better
        return best_edge_score, best_groups


def apply_along_axis(function, axis, x):
    return torch.stack([
        function(x_i) for x_i in x
    ], dim=axis)

def lid_term_torch(X, batch, k=20):
    eps = 1e-6
    X = torch.tensor(X).float()
    batch = torch.tensor(batch).float()
    f = lambda v: - k / (torch.sum(torch.log(v / (v[-1]+eps)))+eps)
    distances = torch.cdist(X, batch)
    # print(distances)

    # get the closest k neighbours
    sort_indices = torch.argsort(distances, dim=1)[:, 1:k + 1]
    # print(sort_indices)
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices

    # sorted matrix
    distances_ = distances[tuple(idx)]
    # print(distances_)
    lids = apply_along_axis(f, axis=-1, x=distances_)
    # print(lids)
    return lids.mean()


# def calc_IDs(supernet, dataloader, num_batches=3):
def calc_IDs(supernet, data_batch, num_batches=3, ID_method='lid'):
    supernet = supernet.to(device)
    # device = next(supernet.parameters()).device
    IDs = []
    # for batch_idx, batch in enumerate(dataloader):
    #     id_batch = []
    #     if batch_idx+1 > num_batches:
    #         break
    #     imgs, labels = batch
    id_batch = []
    for imgs, labels in [data_batch]:
        imgs, labels = imgs.to(device), labels.to(device)
        bs = imgs.shape[0]
        with torch.no_grad():
            y = supernet(imgs)
        features = supernet.features
        for idx, feat in enumerate(features):
            if isinstance(feat, torch.Tensor):
                feat = feat.view(bs, -1).detach()
            else:
                feat = feat.reshape(bs, -1)
            try:
                if ID_method == 'lid':
                    _id = lid_term_torch(feat, feat).item()
                else:
                    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-20)
                    feat = feat.cpu().numpy()
                    if ID_method == 'twonn':
                        # _id = twonn2(feat)
                        _id = skdim.id.TwoNN().fit_transform(X=feat)
                    elif ID_method == 'fishers':
                        _id = skdim.id.FisherS().fit_transform(X=feat)
                    elif ID_method == 'ess':
                        _id = skdim.id.ESS().fit_transform(X=feat)
                    elif ID_method == 'mle':
                        _id = skdim.id.MLE().fit_transform(X=feat)
                    if _id is np.nan:
                        set_trace()
                        log.info(_id)
            except Exception as e:
                set_trace()
                log.info(idx, str(e))
                log.info(feat.shape, feat.max(), feat.mean(), feat.min())
            id_batch.append(_id)
        # id_batch_1 = np.array(id_batch)[1:]-np.array(id_batch)[:-1]
        # id_batch_2 = np.array(id_batch_1)[1:]-np.array(id_batch_1)[:-1]
        # id_batch = id_batch_1.tolist() + id_batch_2.tolist()
        IDs.append(id_batch)
    IDs = np.vstack(IDs)
    return IDs

def calc_grads(model, data_batch, split_edge_key, crt_mask):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    supernet = model.network.to(device)
    mutator = model.mutator
    device = next(supernet.parameters()).device

    model.configure_optimizers()
    optimizer = model.optimizers()
    if isinstance(optimizer, tuple):
        optimizer = optimizer[0]
    optimizer.zero_grad()
    imgs, labels = data_batch
    imgs, labels = imgs.to(device), labels.to(device)
    bs = imgs.shape[0]
    logits = model(imgs)
    loss = model.criterion(logits, labels)
    loss.backward()
    grads = get_splitted_grads(supernet, mutator, split_edge_key, crt_mask)
    grads = [g.clone().detach().reshape(-1) for g in grads]
    grads = torch.cat(grads, 0)
    return grads    

def get_splitted_grads(model, mutator, split_edge_key, crt_mask):
    params = []
    for name, module in model.named_modules():
        if isinstance(module, OperationSpace):
            if module.key != split_edge_key:
                op_indices = torch.where(crt_mask[module.key]!=0)[0]
                for op_index in op_indices:
                    op = module.candidates[op_index]
                    params += list(op.parameters())
    if hasattr(model, 'classifier'):
        params += list(model.classifier.parameters())
    param_grads = [p.grad for p in params if p.grad is not None]
    return param_grads


def sample_func_from_pool(self):
    assert hasattr(self, 'mask_pool'), 'No mask_pool found'
    result = random.choice(self.mask_pool)
    for key in result:
        result[key] = torch.tensor(result[key]).bool()
    return result


def finetune_mp_from_pool(config, rank: int, mask_pool: dict, finetune_epoch: int,
        load_from_parent: bool=False, parent_ckpt_path: str=None, flag: str=None):
    log_mp = get_logger(f"finetune_mp_{rank}", is_rank_zero=False)
    pid = os.getpid()
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    log_mp.info(f'start finetune in process {pid}, local_rank {local_rank} global_rank {rank}')
    log_mp.info(f'pool size={len(mask_pool)}')
    # log_mp.info(f'start finetune in process {pid}, global rank {rank}')
    datamodule_cfg = config['datamodule']
    datamodule_cfg.num_workers = 2
    datamodule = instantiate(config["datamodule"])

    model_cfg = deepcopy(config.model)
    model = instantiate(model_cfg, _recursive_=False)

    mutator = model.mutator
    mutator.mask_pool = deepcopy(mask_pool)
    mutator.sample_func = types.MethodType(sample_func_from_pool, mutator)

    trainer_cfg = deepcopy(config.trainer)
    trainer_cfg.gpus = [local_rank]
    trainer_cfg.max_epochs = finetune_epoch
    if load_from_parent and parent_ckpt_path:
        trainer_cfg.resume_from_checkpoint = parent_ckpt_path
        ckpt_epoch = torch.load(parent_ckpt_path, map_location='cpu')['epoch']
        trainer_cfg.max_epochs = finetune_epoch + ckpt_epoch
    
    callbacks = init_callbacks(config)
    logger = init_logger(config)
    # trainer = instantiate(trainer_cfg, callbacks=callbacks, logger=logger, _convert_="partial")
    trainer = instantiate(trainer_cfg, callbacks=callbacks, logger=[], _convert_="partial")
    trainer.fit(model, datamodule)
    results = trainer.callback_metrics
    log_mp.info(f"pid={pid} Fintune Done: [{rank}]-th sub-supernet \nresults: {results}")
    
    flag = f'global_cluster_rank[{rank}]'
    ckpt_path = os.path.join(os.getcwd(), f'checkpoints/{flag}_latest.ckpt')
    trainer.save_checkpoint(f"{ckpt_path}")
    log_mp.info(f"pid={pid} Saved [{rank}]-th sub-Supernet checkpoint to {ckpt_path}")
    return trainer, model

