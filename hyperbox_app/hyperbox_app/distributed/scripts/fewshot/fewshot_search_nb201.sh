gpu=$1 # gpu number
suffix=$2 # suffix of the experiemnt name
others=$3 # other hyperparameters

name=fewshot_search_${suffix}
echo $name

python -m hyperbox.run \
hydra.searchpath=[pkg://hyperbox_app.distributed.configs] \
experiment=fewshot_search_nb201.yaml \
hydra.job.name=$name \
logger.wandb.name=$name \
+trainer.gpus=$gpu \
+model.is_net_parallel=False \
$others

# hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
# bash scripts/fewshot_search_nb201.sh 0 nb201 "trainer.fast_dev_run=True ipdb_debug=True"