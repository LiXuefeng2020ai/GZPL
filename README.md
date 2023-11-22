# GZPL

Code for our ACL2023 findings paper: [Generative Zero-Shot Prompt Learning for Cross-Domain Slot Filling with Inverse Prompting](https://arxiv.org/abs/2307.02830)

## Requirement

```
# python==3.8
# torch==1.9.0
# openprompt==1.0
transformers==4.10.0
tqdm==4.62.2
yacs==0.1.8
protobuf==3.19.1
datasets==1.16.0
tensorboardX==2.4.1
sentencepiece==0.1.96
scikit-learn==0.24.2
nltk==3.6.5
```
直接cd根目录下，执行
```
pip install -r requirements.txt
```
这里对于原始的openprompt进行了一定的改动，安装只需要cd根目录下，执行
```
pip install .
```

## Datasets
prompt_sf/snips/data目录下缺少emb文件夹，可以前往[emb](https://github.com/W-lw/PCLC/tree/main/data/snips/emb)进行下载

## Train&&Test
cd prompt_sf目录下，执行test.sh脚本即可


## Citation
If you use any source codes or ideas included in this repository for your work, please cite the following paper.
```
@article{li2023generative,
  title={Generative zero-shot prompt learning for cross-domain slot filling with inverse prompting},
  author={Li, Xuefeng and Wang, Liwen and Dong, Guanting and He, Keqing and Zhao, Jinzheng and Lei, Hao and Liu, Jiachi and Xu, Weiran},
  journal={arXiv preprint arXiv:2307.02830},
  year={2023}
}
```
