PATH=/path/to/hyperbox_app/hyperbox_app/distributed/Imagenet_train
bash ${PATH}/distributed_train.sh 4 \
-c ${PATH}/scripts/proxylessv2/args.yaml \
--mask ${PATH}/scripts/proxylessv2/arch.json \
--knowledge_distill --kd_ratio 9.0 --teacher_name D-Net-big224 \
--teacher_path ./scripts/dnet.pth
# --resume ${PATH}/scripts/proxylessv2/proxyless1.4_v2_acc77.21.tar