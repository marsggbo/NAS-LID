# naslid

## 1. Environment

### 1.1 Install `Hyperbox`

Our project is based on AutoML Framework [`hyperbox`](https://github.com/marsggbo/hyperbox), so you need to first install `hyperbox`.

There are two ways to install hyperbox:

- install via pip

```bash
pip install hyperbox
```

- install via git (recommended)
```bash
git clone https://github.com/marsggbo/hyperbox
cd hyperbox
python setup.py develop
python install -r requirements.txt
pip install pytorch-lightning==1.5.8 # higher version are not tested
```

### 1.2 Prepare enviroment for this project

```
git clone https://github.com/marsggbo/NAS-LID
cd hyperbox_app
python setup.py develop
pip install -r requirements.txt
```

## 2. Code Structure

- `hyperbox_app/hyperbox_app/distributed/Imagenet_train`: finetune the final searched models on the ImageNet. Please see `hyperbox_app/hyperbox_app/distributed/Imagenet_train/scripts` for details of finetuning.
- The other folders and files in `hyperbox_app/hyperbox_app/distributed` are used for Supernet Partition and Evolution-based Search.
- All configurations are managed by `hydra` library, you can read or modify them in `hyperbox_app/hyperbox_app/distributed/configs`. An experiment needs to specify multiple modules, such as `datamodule`, `engine`, `model`, `trainer`, and `logger`. Specifically, the yaml files in `hyperbox_app/hyperbox_app/distributed/configs/experiment` integrate all above modules. For example, `hyperbox_app/hyperbox_app/distributed/configs/experiment/fewshot_search_nb201.yaml` is used for experiment of splitting NB201 search space. You can create you own yaml file to run a different experiment.

## 3. LID/Gradient-based Supernet Partition

Splitting NAS-Bench-201 

### 3.1 Single-GPU
```
bash ./scripts/fewshot/fewshot_search_nb201.sh 0 nb201_c10_ID_mincut_sp_16splits \
"logger.wandb.offline=True trainer.strategy=null engine.split_criterion=ID engine.split_method=mincut"
```

- `nb201_c10_ID_mincut_sp_16splits` is the experiment name, which is used for creating experiment log folder.
- replacing `engine.split_criterion=ID` by `engine.split_criterion=grad` will split the Supernet via gradient.

### 3.2 Multi-GPU (e.g., 4 GPUs)

```
bash ./scripts/fewshot/fewshot_search_nb201.sh 4 nb201_c10_ID_mincut_sp_16splits \
"logger.wandb.offline=True trainer.strategy=null engine.split_criterion=ID engine.split_method=mincut +trainer.strategy=ddp"
```



### 3.3 For debug:

```
bash ./scripts/fewshot/fewshot_search_nb201.sh 0 debug_nb201_c10_ID_mincut_sp_4splits \
"ipdb_debug=False logger.wandb.offline=True trainer.strategy=null +trainer.limit_val_batches=10 engine.split_criterion=ID engine.split_method=mincut engine.is_single_path=True +trainer.fast_dev_run=True engine.warmup_epochs=[1,2] engine.repeat_num=1 engine.finetune_epoch=1"
```

## 4. Evolutionary Algorithm-based Search

Simialrly, the following line can generate a series of running commands for evolution-based search.

The following two paths are important:
- 1. the configuration template of ea-based search, e.g., `/path/to/scripts/fewshot/ea_search_cfgs` (absolute path is preferred)
- 2. the log path of the previous experiment in Step 3, e.g., `/path/to/logs/runs/nb201_c10_ID_mincut_sp_16splits/2022-12-10_07-50-42`


```bash
path1=/path/to/scripts/fewshot/ea_search_cfgs
path2=/path/to/logs/runs/nb201_c10_ID_mincut_sp_16splits/2022-12-10_07-50-42
bash scripts/fewshot/ea_search.sh [0] eaSearch_exp_name 1.0 ${path1} " ++engine.supernet_mask_path_pattern=${path2}/checkpoints/*mask.json ipdb_debug=False trainer.fast_dev_run=False model.mutator_cfg.evolution_epochs=100  model.mutator_cfg.population_num=50"
```

- Command for debug

```bash
path1=/path/to/scripts/fewshot/ea_search_cfgs
path2=/path/to/logs/runs/nb201_c10_ID_mincut_sp_16splits/2022-12-10_07-50-42
bash scripts/fewshot/ea_search.sh [0] eaSearch_exp_name 1.0 ${path1} " ++engine.supernet_mask_path_pattern=${path2}/checkpoints/*mask.json ipdb_debug=True trainer.fast_dev_run=True model.mutator_cfg.evolution_epochs=2  model.mutator_cfg.population_num=10"
```

## 5. Evaluation on Searched Models

The way of reproducing the results on OFA and ProxylessNAS search space is detailed in the following folder

```bash
cd hyperbox_app/hyperbox_app/distributed/Imagenet_train/scripts
```

You may need to modify the path arguments to your local path in each bash scripts before evaluations.
