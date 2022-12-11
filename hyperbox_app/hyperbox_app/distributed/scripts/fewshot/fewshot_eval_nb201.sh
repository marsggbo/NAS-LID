cfgPath=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/scripts/fewshot/.hydra
gpu=$1
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
VB=$2
suffix=$3
name=evalTau_Fewshot_${suffix}
others=$4

echo $name

# nb201
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
--config-path=${cfgPath} \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
trainer.limit_val_batches=${VB} \
hydra.job.name=$name \
++hydra.job.chdir=True \
logger.wandb.name=$name \
engine.supernet_mask_path_pattern=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/fewshot_search_nb201_c10_grad_spectral_cluster_sp/2022-06-19_04-27-59/checkpoints/*json \
engine.search_space_path=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/networks/nasbench201/top1percent_models.json \
engine.ckpts_path_pattern=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/fewshot_search_nb201_c10_grad_mincut_sp/2022-06-18_02-10-54/checkpoints/*subSup*ckpt \
logger.wandb.offline=True \
$others


# engine.supernet_mask_path_pattern=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/test_supernet_masks/*json \
# engine.search_space_path=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/networks/nasbench201/top1percent_models.json \
# engine.ckpts_path_pattern=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/fewshot_search_ID_nb201/2022-06-16_04-01-41/checkpoints/*subSup*ckpt \
# logger.wandb.offline=True \