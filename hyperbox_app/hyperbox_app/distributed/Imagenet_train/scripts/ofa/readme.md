# Download pretrained weights of OFA Supernet


Download the pretrained weights of OFA Supernet from onedrive cloud: [hyperbox_OFA_MBV3_k357_d234_e346_w1.2.pth](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18481086_life_hkbu_edu_hk/EcZiX-B5pxhOmCOiL3kYNkgBmvEgY_ELEu21uKrEWQMnKw?e=hdwu39)

# Download Checkpoint

Download the checkpoint from onedrive cloud: [ofa_acc80.46.tar](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18481086_life_hkbu_edu_hk/EWi9reXFMupLimTQCKaXeecBPAOJ-vhXTyzJ_8O8sYXmuw?e=X179oq)

# Download Teacher pretrained weights

1. Download the weights of teacher network [dnet.pth](https://drive.google.com/file/d/17azGLyfcCCP0IfGVoDpBVaPAqbdVyl42/view?usp=sharing)
2. mv `dnet.pth` to `./scripts/dnet.pt`



# Train
```bash
bash scripts/ofa/run.sh
```


# Eval
```bash
bash scripts/ofa/eval.sh
```

- top1@acc=80.46
- top5@acc=94.97
