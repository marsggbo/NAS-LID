# Download pretrained weights of ProxylessNAS Supernet


Download the pretrained weights of ProxylessNAS Supernet from onedrive cloud: [hyperbox_proxy_mobile_w1.4.pth](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18481086_life_hkbu_edu_hk/EV5I3Nqz7_lCuZGIR81bF18BIkvKlVxMZcoZdJp4arRd4A)


# Download Checkpoint

1. Download the checkpoint from onedrive cloud: [proxyless1.4_v2_acc77.21.tar](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18481086_life_hkbu_edu_hk/EZyF93CRnS5Cjxdew5QytUcBHnTZpwDT7hH8Iq4-NKjRCg)

2. Save the checkpoint to `scripts/proxylessv2/proxyless1.4_v2_acc77.21.tar`

# Download Teacher pretrained weights

1. Download the weights of teacher network [dnet.pth](https://drive.google.com/file/d/17azGLyfcCCP0IfGVoDpBVaPAqbdVyl42/view?usp=sharing)
2. mv `dnet.pth` to `./scripts/dnet.pt`


# Train
```bash
bash scripts/proxylessv2/run.sh
```


# Eval
```bash
bash scripts/proxylessv2/eval.sh
```
