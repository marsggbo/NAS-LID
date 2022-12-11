
gpu=$1
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
suffix=$2
VB=$3
cfgPath=$4
others=$5
name=ea_search_${suffix}

echo $name


# nb201
# CUDA_VISIBLE_DEVICES=$gpu 
python -m hyperbox.run \
--config-path=${cfgPath} \
hydra.searchpath=[pkg://hyperbox_app.distributed.configs] \
hydra.job.name=$name \
logger.wandb.name=$name \
trainer.limit_val_batches=${VB} \
trainer.gpus=$gpu \
$others
