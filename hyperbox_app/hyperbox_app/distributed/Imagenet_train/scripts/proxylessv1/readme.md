# Download pretrained weights of ProxylessNAS Supernet


Download the pretrained weights of ProxylessNAS Supernet from onedrive cloud: [hyperbox_proxy_mobile_w1.4.pth](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18481086_life_hkbu_edu_hk/EV5I3Nqz7_lCuZGIR81bF18BIkvKlVxMZcoZdJp4arRd4A)

# Download Checkpoint

1. Download the checkpoint from onedrive cloud: [proxyless1.4_v1_acc77.15.tar](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18481086_life_hkbu_edu_hk/EVWkDyP3hxBGnNTOFaAgDAQBnPa6gVoXZQbrVFLie3xOWg?e=KhEJKO)

2. Save the checkpoint to `scripts/proxylessv1/proxyless1.4_v2_acc77.15.tar`

# Download Teacher pretrained weights

1. Download the weights of teacher network [dnet.pth](https://drive.google.com/file/d/17azGLyfcCCP0IfGVoDpBVaPAqbdVyl42/view?usp=sharing)
2. mv `dnet.pth` to `./scripts/dnet.pt`


# Train
```bash
bash scripts/proxylessv1/run.sh
```


# Eval
```bash
bash scripts/proxylessv1/eval.sh
```
