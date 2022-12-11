PATH=/path/to/hyperbox_app/hyperbox_app/distributed/Imagenet_train
bash ${PATH}/distributed_train.sh 4 \
-c ${PATH}/scripts/ofa/args.yaml \
--mask ${PATH}/scripts/ofa/arch.json \
--opt lookahead_rmsproptf --opt-eps .001  --knowledge_distill --kd_ratio 9.0 --teacher_name D-Net-big224 \
--resume ${PATH}/scripts/ofa/ofa_acc80.46.tar \
--evalonly 1