PATH=/path/to/hyperbox_app/hyperbox_app/distributed/Imagenet_train
bash ${PATH}/distributed_train.sh 4 \
-c ${PATH}/scripts/proxylessv1/args.yaml \
--mask ${PATH}/scripts/proxylessv1/arch.json \
--knowledge_distill --kd_ratio 9.0 --teacher_name D-Net-big224 \
--teacher_path ./scripts/dnet.pth
# --resume ${PATH}/scripts/proxylessv1/proxyless1.4_v1_acc77.15.tar